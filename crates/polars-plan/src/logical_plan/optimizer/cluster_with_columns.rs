use std::sync::Arc;

use arrow::bitmap::MutableBitmap;
use polars_core::schema::Schema;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::arena::{Arena, Node};

use super::aexpr::AExpr;
use super::alp::IR;
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
    let mut current_livesets: Vec<MutableBitmap> = Vec::with_capacity(16);
    let mut pushable = MutableBitmap::with_capacity(16);

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
        current_livesets.clear();
        pushable.clear();

        // @NOTE
        // We can pushdown any column that utilizes no live columns that are generated in the
        // input.

        for input_expr in input_exprs.as_exprs() {
            column_map_set(
                &mut input_genset,
                column_map,
                input_expr.output_name_arc().clone(),
            );
        }

        for expr in current_exprs.as_exprs() {
            let mut liveset = MutableBitmap::from_len_zeroed(column_map.len());

            for live in aexpr_to_leaf_names_iter(expr.node(), expr_arena) {
                column_map_set(&mut liveset, column_map, live.clone());
            }

            current_livesets.push(liveset);
        }

        // Force that column_map is not further mutated from this point on
        let column_map = column_map as &_;

        column_map_finalize_bitset(&mut input_genset, column_map);

        // Check for every expression in the current WITH_COLUMNS node whether it can be pushed
        // down.
        for expr_liveset in &mut current_livesets {
            column_map_finalize_bitset(expr_liveset, column_map);

            let has_intersection = input_genset.intersects_with(expr_liveset);
            let is_pushable = !has_intersection;

            pushable.push(is_pushable);
        }

        let pushable_set_bits = pushable.set_bits();

        // If all columns are pushable, we can merge the input into the current. This should be
        // a relatively common case.
        if pushable_set_bits == pushable.len() {
            // @NOTE: To keep the schema correct, we reverse the order here. As a
            // `WITH_COLUMNS` higher up produces later columns. This also allows us not to
            // have to deal with schemas.
            input_exprs
                .exprs_mut()
                .extend(std::mem::take(current_exprs.exprs_mut()));
            std::mem::swap(current_exprs.exprs_mut(), input_exprs.exprs_mut());

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
        input_exprs.exprs_mut().reserve(pushable_set_bits);
        *current_exprs.exprs_mut() = std::mem::take(current_exprs.exprs_mut())
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
                    input_exprs.exprs_mut().push(expr);

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
