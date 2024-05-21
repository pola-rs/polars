use std::sync::Arc;

use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_core::schema::SchemaRef;
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};

use super::aexpr::AExpr;
use super::alp::IR;
use super::expr_ir::OutputName;
use super::{aexpr_to_leaf_names_iter, ColumnName, ProjectionOptions};
use crate::logical_plan::projection_expr::ProjectionExprs;

type ColumnMap = PlHashMap<ColumnName, usize>;

/// Utility structure to contain all the information of a [`IR::HStack`] i.e. a `WITH_COLUMNS` node
struct WithColumnsMut<'a> {
    input: &'a mut Node,
    exprs: &'a mut ProjectionExprs,
    schema: &'a mut SchemaRef,
    options: &'a mut ProjectionOptions,
}

/// A bitset that contains which columns where read from i.e. are live
struct LiveSet(MutableBitmap);

impl<'a> WithColumnsMut<'a> {
    fn from_ir(ir: &'a mut IR) -> Option<Self> {
        let IR::HStack {
            input,
            exprs,
            schema,
            options,
        } = ir
        else {
            return None;
        };

        Some(Self {
            input,
            exprs,
            schema,
            options,
        })
    }
}

fn column_map_idx(column_map: &mut ColumnMap, column: ColumnName) -> usize {
    let size = column_map.len();
    *column_map.entry(column).or_insert_with(|| size)
}

impl LiveSet {
    fn new(expr: Node, expr_arena: &Arena<AExpr>, column_map: &mut ColumnMap) -> Self {
        let mut liveset = MutableBitmap::from_len_zeroed(column_map.len());

        for live in aexpr_to_leaf_names_iter(expr, expr_arena) {
            let size = column_map.len();
            column_map
                .entry(live)
                .and_modify(|idx| {
                    liveset.set(*idx, true);
                })
                .or_insert_with(|| {
                    liveset.push(true);
                    size
                });
        }

        Self(liveset)
    }

    /// Finalize the the [`LiveSet`] into a [`Bitmap`] which has a status bit-flag for each column
    fn freeze(self, column_map: &ColumnMap) -> Bitmap {
        let Self(mut liveset) = self;

        assert!(liveset.len() <= column_map.len());

        let additional = column_map.len() - liveset.len();
        liveset.extend_constant(additional, false);
        liveset.freeze()
    }
}

pub fn optimize(root: Node, lp_arena: &mut Arena<IR>, expr_arena: &Arena<AExpr>) {
    let mut ir_stack = Vec::with_capacity(16);
    ir_stack.push(root);

    loop {
        let Some(current) = ir_stack.pop() else {
            break;
        };

        while let IR::HStack { input, .. } = lp_arena.get(current) {
            let input = *input;

            let [current_ir, input_ir] = lp_arena.get_many_mut([current, input]);

            let Some(current_wc) = WithColumnsMut::from_ir(current_ir) else {
                unreachable!();
            };
            let Some(input_wc) = WithColumnsMut::from_ir(input_ir) else {
                break;
            };

            // @NOTE
            // We can pushdown any column that utilizes no live columns that are generated in the
            // input.

            let mut column_map = ColumnMap::default();

            let input_gen_columns: Vec<usize> = input_wc
                .exprs
                .as_exprs()
                .iter()
                .filter_map(|expr| match expr.output_name_inner() {
                    OutputName::None => None,
                    OutputName::LiteralLhs(s) | OutputName::ColumnLhs(s) => {
                        Some(column_map_idx(&mut column_map, s.clone()))
                    },
                    OutputName::Alias(s) => Some(column_map_idx(&mut column_map, s.clone())),
                })
                .collect();

            let current_livesets: Vec<LiveSet> = current_wc
                .exprs
                .as_exprs()
                .iter()
                .map(|expr| LiveSet::new(expr.node(), expr_arena, &mut column_map))
                .collect();
            let current_livesets: Vec<Bitmap> = current_livesets
                .into_iter()
                .map(|s| s.freeze(&column_map))
                .collect();

            // @NOTE: This satisfies the borrow checker
            let num_names = column_map.len();
            drop(column_map);

            let mut input_generate = MutableBitmap::from_len_zeroed(num_names);
            for idx in &input_gen_columns {
                input_generate.set(*idx, true);
            }
            let input_generate = input_generate.freeze();

            // Check for every expression in the current WITH_COLUMNS node whether it can be pushed
            // down.
            let mut pushable = MutableBitmap::with_capacity(input_wc.exprs.len());
            for expr_liveset in &current_livesets {
                let has_intersection = input_generate.intersect_with(expr_liveset);
                let is_pushable = !has_intersection;
                pushable.push(is_pushable);
            }
            let pushable = pushable.freeze();

            // There is nothing to push down. Move on.
            if pushable.set_bits() == 0 {
                break;
            }

            // If all columns are pushable, we can merge the input into the current. This should be
            // a relatively common case.
            if pushable.set_bits() == pushable.len() {
                // @NOTE: To keep the schema correct, we reverse the order here. As a
                // `WITH_COLUMNS` higher up produces later columns. This also allows us not to
                // have to deal with schemas.
                input_wc
                    .exprs
                    .exprs_mut()
                    .extend(std::mem::take(current_wc.exprs.exprs_mut()));
                std::mem::swap(current_wc.exprs.exprs_mut(), input_wc.exprs.exprs_mut());

                *current_wc.input = *input_wc.input;
                *current_wc.options = current_wc.options.merge_options(input_wc.options);

                // Let us just make this node invalid so we can detect when someone tries to
                // mention it later.
                lp_arena.take(input);

                // Since we merged the current and input nodes and the input node might have
                // optimizations with their input, we loop again on this node.
                continue;
            }

            let mut current_schema = current_wc.schema.as_ref().clone();
            let mut input_schema = input_wc.schema.as_ref().clone();

            // @NOTE: We don't have to insert a SimpleProjection or redo the `current_schema` if
            // `pushable` contains only 0..N for some N. We use these two variables to keep track
            // of this.
            let mut has_seen_unpushable = false;
            let mut needs_simple_projection = false;

            let current_exprs = std::mem::take(current_wc.exprs.exprs_mut())
                .into_iter()
                .zip(pushable)
                .enumerate()
                .filter_map(|(i, (expr, do_pushdown))| {
                    if do_pushdown {
                        needs_simple_projection = has_seen_unpushable;

                        // @NOTE: It would be nice to have a `remove_at_index` here.
                        let (column, _) = current_wc.schema.get_at_index(i).unwrap();

                        input_wc.exprs.exprs_mut().push(expr);
                        let datatype = current_schema.remove(column).unwrap();
                        input_schema.with_column(column.clone(), datatype);

                        None
                    } else {
                        has_seen_unpushable = true;
                        Some(expr)
                    }
                })
                .collect();

            *current_wc.exprs.exprs_mut() = current_exprs;

            let options = current_wc.options.merge_options(input_wc.options);
            *current_wc.options = options;
            *input_wc.options = options;

            // @NOTE: Here we add a simple projection to make sure that the output still
            // has the right schema.
            if needs_simple_projection {
                current_schema.merge(input_schema.clone());
                *input_wc.schema = Arc::new(input_schema);
                let proj_schema = current_wc.schema.clone();
                *current_wc.schema = Arc::new(current_schema);

                let moved_current = lp_arena.add(IR::Invalid);
                let projection = IR::SimpleProjection {
                    input: moved_current,
                    columns: proj_schema,
                };
                let current = lp_arena.replace(current, projection);
                lp_arena.replace(moved_current, current);
            } else {
                *input_wc.schema = Arc::new(input_schema);
            }

            // We know that this node is done optimizing
            break;
        }

        lp_arena.get(current).copy_inputs(&mut ir_stack);
    }
}
