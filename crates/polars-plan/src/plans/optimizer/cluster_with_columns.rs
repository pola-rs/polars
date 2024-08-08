use std::sync::Arc;

use arrow::bitmap::MutableBitmap;
use polars_core::schema::Schema;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::vec::inplace_zip_filtermap;

use super::aexpr::AExpr;
use super::ir::IR;
use super::{aexpr_to_leaf_names_iter, ColumnName};

type ColumnMap = PlHashMap<ColumnName, usize>;

fn column_map_finalize_bitset(bitset: &mut MutableBitmap, column_map: &ColumnMap) {
    assert!(bitset.len() <= column_map.len());

    let size = bitset.len();
    bitset.extend_constant(column_map.len() - size, false);
}

fn column_map_set(bitset: &mut MutableBitmap, column_map: &mut ColumnMap, column: ColumnName) {
    let size = column_map.len();
    column_map
        .entry(column)
        .and_modify(|idx| bitset.set(*idx, true))
        .or_insert_with(|| {
            bitset.push(true);
            size
        });
}

pub fn optimize(root: Node, lp_arena: &mut Arena<IR>, expr_arena: &Arena<AExpr>) {
    let mut ir_stack = Vec::with_capacity(16);
    ir_stack.push(root);

    // We define these here to reuse the allocations across the loops
    let mut column_map = ColumnMap::with_capacity(8);
    let mut input_genset = MutableBitmap::with_capacity(16);
    let mut current_expr_livesets: Vec<MutableBitmap> = Vec::with_capacity(16);
    let mut current_liveset = MutableBitmap::with_capacity(16);
    let mut pushable = MutableBitmap::with_capacity(16);
    let mut potential_pushable = Vec::with_capacity(4);

    while let Some(current) = ir_stack.pop() {
        let current_ir = lp_arena.get(current);
        current_ir.copy_inputs(&mut ir_stack);
        let IR::HStack { input, .. } = current_ir else {
            continue;
        };
        let input = *input;

        let [current_ir, input_ir] = lp_arena.get_many_mut([current, input]);

        let IR::HStack {
            input: ref mut current_input,
            exprs: ref mut current_exprs,
            schema: ref mut current_schema,
            options: ref mut current_options,
        } = current_ir
        else {
            unreachable!();
        };
        let IR::HStack {
            input: ref mut input_input,
            exprs: ref mut input_exprs,
            schema: ref mut input_schema,
            options: ref mut input_options,
        } = input_ir
        else {
            continue;
        };

        let column_map = &mut column_map;

        // Reuse the allocations of the previous loop
        column_map.clear();
        input_genset.clear();
        current_expr_livesets.clear();
        current_liveset.clear();
        pushable.clear();
        potential_pushable.clear();

        pushable.reserve(current_exprs.len());
        potential_pushable.reserve(current_exprs.len());

        // @NOTE
        // We can pushdown any column that utilizes no live columns that are generated in the
        // input.

        for input_expr in input_exprs.iter() {
            column_map_set(
                &mut input_genset,
                column_map,
                input_expr.output_name_arc().clone(),
            );
        }

        for expr in current_exprs.iter() {
            let mut liveset = MutableBitmap::from_len_zeroed(column_map.len());

            for live in aexpr_to_leaf_names_iter(expr.node(), expr_arena) {
                column_map_set(&mut liveset, column_map, live.clone());
            }

            current_expr_livesets.push(liveset);
        }

        // Force that column_map is not further mutated from this point on
        let column_map = column_map as &_;

        column_map_finalize_bitset(&mut input_genset, column_map);

        current_liveset.extend_constant(column_map.len(), false);
        for expr_liveset in &mut current_expr_livesets {
            use std::ops::BitOrAssign;
            column_map_finalize_bitset(expr_liveset, column_map);
            (&mut current_liveset).bitor_assign(expr_liveset as &_);
        }

        // Check for every expression in the current WITH_COLUMNS node whether it can be pushed
        // down or pruned.
        inplace_zip_filtermap(
            current_exprs,
            &mut current_expr_livesets,
            |mut expr, liveset| {
                let does_input_assign_column_that_expr_used =
                    input_genset.intersects_with(&liveset);

                if does_input_assign_column_that_expr_used {
                    pushable.push(false);
                    return Some((expr, liveset));
                }

                let column_name = expr.output_name_arc();
                let is_pushable = if let Some(idx) = column_map.get(column_name) {
                    let does_input_alias_also_expr = input_genset.get(*idx);
                    let is_alias_live_in_current = current_liveset.get(*idx);

                    if does_input_alias_also_expr && !is_alias_live_in_current {
                        let column_name = column_name.as_ref();

                        // @NOTE: Pruning of re-assigned columns
                        //
                        // We checked if this expression output is also assigned by the input and
                        // that that assignment is not used in the current WITH_COLUMNS.
                        // Consequently, we are free to prune the input's assignment to the output.
                        //
                        // We immediately prune here to simplify the later code.
                        //
                        // @NOTE: Expressions in a `WITH_COLUMNS` cannot alias to the same column.
                        // Otherwise, this would be faulty and would panic.
                        let input_expr = input_exprs
                            .iter_mut()
                            .find(|input_expr| column_name == input_expr.output_name())
                            .expect("No assigning expression for generated column");

                        // @NOTE
                        // Since we are reassigning a column and we are pushing to the input, we do
                        // not need to change the schema of the current or input nodes.
                        std::mem::swap(&mut expr, input_expr);
                        return None;
                    }

                    // We cannot have multiple assignments to the same column in one WITH_COLUMNS
                    // and we need to make sure that we are not changing the column value that
                    // neighbouring expressions are seeing.

                    // @NOTE: In this case it might be possible to push this down if all the
                    // expressions that use the output are also being pushed down.
                    if !does_input_alias_also_expr && is_alias_live_in_current {
                        potential_pushable.push(pushable.len());
                        pushable.push(false);
                        return Some((expr, liveset));
                    }

                    !does_input_alias_also_expr && !is_alias_live_in_current
                } else {
                    true
                };

                pushable.push(is_pushable);
                Some((expr, liveset))
            },
        );

        debug_assert_eq!(pushable.len(), current_exprs.len());

        // Here we do a last check for expressions to push down.
        // This will pushdown the expressions that "has an output column that is mentioned by
        // neighbour columns, but all those neighbours were being pushed down".
        for candidate in potential_pushable.iter().copied() {
            let column_name = current_exprs[candidate].output_name_arc();
            let column_idx = column_map.get(column_name).unwrap();

            current_liveset.clear();
            current_liveset.extend_constant(column_map.len(), false);
            for (i, expr_liveset) in current_expr_livesets.iter().enumerate() {
                if pushable.get(i) || i == candidate {
                    continue;
                }
                use std::ops::BitOrAssign;
                (&mut current_liveset).bitor_assign(expr_liveset as &_);
            }

            if !current_liveset.get(*column_idx) {
                pushable.set(candidate, true);
            }
        }

        let pushable_set_bits = pushable.set_bits();

        // If all columns are pushable, we can merge the input into the current. This should be
        // a relatively common case.
        if pushable_set_bits == pushable.len() {
            // @NOTE: To keep the schema correct, we reverse the order here. As a
            // `WITH_COLUMNS` higher up produces later columns. This also allows us not to
            // have to deal with schemas.
            input_exprs.extend(std::mem::take(current_exprs));
            std::mem::swap(current_exprs, input_exprs);

            // Here, we perform the trick where we switch the inputs. This makes it possible to
            // change the essentially remove the `current` node without knowing the parent of
            // `current`. Essentially, we move the input node to the current node.
            *current_input = *input_input;
            *current_options = current_options.merge_options(input_options);

            // Let us just make this node invalid so we can detect when someone tries to
            // mention it later.
            lp_arena.take(input);

            // Since we merged the current and input nodes and the input node might have
            // optimizations with their input, we loop again on this node.
            ir_stack.pop();
            ir_stack.push(current);
            continue;
        }

        // There is nothing to push down. Move on.
        if pushable_set_bits == 0 {
            continue;
        }

        let input_schema_inner = Arc::make_mut(input_schema);

        // @NOTE: We don't have to insert a SimpleProjection or redo the `current_schema` if
        // `pushable` contains only 0..N for some N. We use these two variables to keep track
        // of this.
        let mut has_seen_unpushable = false;
        let mut needs_simple_projection = false;

        input_schema_inner.reserve(pushable_set_bits);
        input_exprs.reserve(pushable_set_bits);
        *current_exprs = std::mem::take(current_exprs)
            .into_iter()
            .zip(pushable.iter())
            .filter_map(|(expr, do_pushdown)| {
                if do_pushdown {
                    needs_simple_projection = has_seen_unpushable;

                    let column = expr.output_name_arc().as_ref();
                    // @NOTE: we cannot just use the index here, as there might be renames that sit
                    // earlier in the schema
                    let datatype = current_schema.get(column).unwrap();
                    input_schema_inner.with_column(column.into(), datatype.clone());
                    input_exprs.push(expr);

                    None
                } else {
                    has_seen_unpushable = true;
                    Some(expr)
                }
            })
            .collect();

        let options = current_options.merge_options(input_options);
        *current_options = options;
        *input_options = options;

        // @NOTE: Here we add a simple projection to make sure that the output still
        // has the right schema.
        if needs_simple_projection {
            // @NOTE: This may seem stupid, but this way we prioritize the input columns and then
            // the existing columns which is exactly what we want.
            let mut new_current_schema = Schema::with_capacity(current_schema.len());
            new_current_schema.merge_from_ref(input_schema.as_ref());
            new_current_schema.merge_from_ref(current_schema.as_ref());

            debug_assert_eq!(new_current_schema.len(), current_schema.len());

            let proj_schema = std::mem::replace(current_schema, Arc::new(new_current_schema));

            let moved_current = lp_arena.add(IR::Invalid);
            let projection = IR::SimpleProjection {
                input: moved_current,
                columns: proj_schema,
            };
            let current = lp_arena.replace(current, projection);
            lp_arena.replace(moved_current, current);
        }
    }
}
