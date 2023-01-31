use polars_core::prelude::*;

use crate::prelude::*;
use crate::utils::aexpr_is_simple_projection;

pub(super) struct SlicePushDown {
    streaming: bool,
}

#[derive(Copy, Clone)]
struct State {
    offset: i64,
    len: IdxSize,
}

impl SlicePushDown {
    pub(super) fn new(streaming: bool) -> Self {
        Self { streaming }
    }

    // slice will be done at this node if we found any
    // we also stop optimization
    fn no_pushdown_finish_opt(
        &self,
        lp: ALogicalPlan,
        state: Option<State>,
        lp_arena: &mut Arena<ALogicalPlan>,
    ) -> PolarsResult<ALogicalPlan> {
        match state {
            Some(state) => {
                let input = lp_arena.add(lp);

                let lp = ALogicalPlan::Slice {
                    input,
                    offset: state.offset,
                    len: state.len,
                };
                Ok(lp)
            }
            None => Ok(lp),
        }
    }

    /// slice will be done at this node, but we continue optimization
    fn no_pushdown_restart_opt(
        &self,
        lp: ALogicalPlan,
        state: Option<State>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<ALogicalPlan> {
        let inputs = lp.get_inputs();
        let exprs = lp.get_exprs();

        let new_inputs = inputs
            .iter()
            .map(|&node| {
                let alp = lp_arena.take(node);
                // No state, so we do not push down the slice here.
                let state = None;
                let alp = self.pushdown(alp, state, lp_arena, expr_arena)?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let lp = lp.with_exprs_and_input(exprs, new_inputs);

        self.no_pushdown_finish_opt(lp, state, lp_arena)
    }

    /// slice will be pushed down.
    fn pushdown_and_continue(
        &self,
        lp: ALogicalPlan,
        state: Option<State>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<ALogicalPlan> {
        let inputs = lp.get_inputs();
        let exprs = lp.get_exprs();

        let new_inputs = inputs
            .iter()
            .map(|&node| {
                let alp = lp_arena.take(node);
                let alp = self.pushdown(alp, state, lp_arena, expr_arena)?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(lp.with_exprs_and_input(exprs, new_inputs))
    }

    fn pushdown(
        &self,
        lp: ALogicalPlan,
        state: Option<State>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<ALogicalPlan> {
        use ALogicalPlan::*;

        match (lp, state) {
            (AnonymousScan {
                function,
                file_info,
                output_schema,
                predicate,
                options,
            },
                // TODO! we currently skip slice pushdown if there is a predicate.
                // we can modify the readers to only limit after predicates have been applied
                Some(state)) if state.offset == 0 && predicate.is_none() => {
                let mut options = options;
                options.n_rows = Some(state.len as usize);
                let lp = AnonymousScan {
                    function,
                    file_info,
                    output_schema,
                    predicate,
                    options,
                };

                Ok(lp)
            },

            #[cfg(feature = "parquet")]
            (ParquetScan {
                path,
                file_info,
                output_schema,
                predicate,
                options,
                cloud_options,

            },
                // TODO! we currently skip slice pushdown if there is a predicate.
                // we can modify the readers to only limit after predicates have been applied
                Some(state)) if state.offset == 0 && predicate.is_none() => {
                let mut options = options;
                options.n_rows = Some(state.len as usize);
                let lp = ParquetScan {
                    path,
                    file_info,
                    output_schema,
                    predicate,
                    options,
                    cloud_options,
                };

                Ok(lp)
            },
            #[cfg(feature = "ipc")]
            (IpcScan {path,
            file_info,
                output_schema,
                predicate,
                options
            }, Some(state)) if state.offset == 0 && predicate.is_none() => {
                let mut options = options;
                options.n_rows = Some(state.len as usize);
                let lp = IpcScan {
                    path,
                    file_info,
                    output_schema,
                    predicate,
                    options
                };

                Ok(lp)

            }

            #[cfg(feature = "csv-file")]
            (CsvScan {
                path,
                file_info,
                output_schema,
                options,
                predicate,
            }, Some(state)) if state.offset >= 0 && predicate.is_none() => {
                let mut options = options;
                options.skip_rows += state.offset as usize;
                options.n_rows = Some(state.len as usize);

                let lp = CsvScan {
                    path,
                    file_info,
                    output_schema,
                    options,
                    predicate,
                };
                Ok(lp)
            }

            (Union {inputs, mut options }, Some(state)) => {
                options.slice = true;
                options.slice_offset = state.offset;
                options.slice_len = state.len;
                Ok(Union {inputs, options})
            },
            (Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                mut options
            }, Some(state)) if !self.streaming => {
                // first restart optimization in both inputs and get the updated LP
                let lp_left = lp_arena.take(input_left);
                let lp_left = self.pushdown(lp_left, None, lp_arena, expr_arena)?;
                let input_left = lp_arena.add(lp_left);

                let lp_right = lp_arena.take(input_right);
                let lp_right = self.pushdown(lp_right, None, lp_arena, expr_arena)?;
                let input_right = lp_arena.add(lp_right);

                // then assign the slice state to the join operation

                options.slice = Some((state.offset, state.len as usize));

                Ok(Join {
                    input_left,
                    input_right,
                    schema,
                    left_on,
                    right_on,
                    options
                })
            }
            (Aggregate { input, keys, aggs, schema, apply, maintain_order, mut options }, Some(state)) => {
                // first restart optimization in inputs and get the updated LP
                let input_lp = lp_arena.take(input);
                let input_lp = self.pushdown(input_lp, None, lp_arena, expr_arena)?;
                let input= lp_arena.add(input_lp);

                options.slice = Some((state.offset, state.len as usize));

                Ok(Aggregate {
                    input,
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    options
                })
            }
            (Sort {input, by_column, mut args}, Some(state)) => {
                // first restart optimization in inputs and get the updated LP
                let input_lp = lp_arena.take(input);
                let input_lp = self.pushdown(input_lp, None, lp_arena, expr_arena)?;
                let input= lp_arena.add(input_lp);

                args.slice = Some((state.offset, state.len as usize));
                Ok(Sort {
                    input,
                    by_column,
                    args
                })
            }
            (Slice {
                input,
                offset,
                len
            }, Some(previous_state)) => {
                let alp = lp_arena.take(input);
                let state = Some(State {
                    offset,
                    len
                });
                let lp = self.pushdown(alp, state, lp_arena, expr_arena)?;
                let input = lp_arena.add(lp);
                Ok(Slice {
                    input,
                    offset: previous_state.offset,
                    len: previous_state.len
                })
            }
            (Slice {
                input,
                offset,
                len
            }, None) => {
                let alp = lp_arena.take(input);
                let state = Some(State {
                    offset,
                    len
                });
                self.pushdown(alp, state, lp_arena, expr_arena)
            }
            // [Do not pushdown] boundary
            // here we do not pushdown.
            // we reset the state and then start the optimization again
            m @ (Selection { .. }, _)
            // let's be conservative. projections may do aggregations and a pushed down slice
            // will lead to incorrect aggregations
            | m @ (LocalProjection {..},_)
            // other blocking nodes
            | m @ (DataFrameScan {..}, _)
            | m @ (Sort {..}, _)
            | m @ (Explode {..}, _)
            | m @ (Melt {..}, _)
            | m @ (Cache {..}, _)
            | m @ (Distinct {..}, _)
            | m @ (HStack {..},_)
            | m @ (Aggregate{..},_)
            // blocking in streaming
            | m @ (Join{..},_)
            => {
                let (lp, state) = m;
                self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
            }
            // [Pushdown]
            (MapFunction {input, function}, _) if function.allow_predicate_pd() => {
                let lp = MapFunction {input, function};
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            },
            // [NO Pushdown]
            m @ (MapFunction {..}, _) => {
                let (lp, state) = m;
                self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
            }
            // [Pushdown]
            // these nodes will be pushed down.
             // State is None, we can continue
             m @(Projection{..}, None)
            => {
                let (lp, state) = m;
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            }
            // there is state, inspect the projection to determine how to deal with it
            (Projection {input, mut expr, schema}, Some(State{offset, len})) => {
                // The slice operation may only pass on simple projections. col("foo").alias("bar")
                if expr.iter().all(|root|  {
                    aexpr_is_simple_projection(*root, expr_arena)
                }) {
                    let lp = Projection {input, expr, schema};
                    self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
                }
                // we add a slice node to the projections
                else {
                    let offset_node = to_aexpr(lit(offset), expr_arena);
                    let length_node = to_aexpr(lit(len), expr_arena);
                    expr.iter_mut().for_each(|node| {
                        let aexpr = AExpr::Slice {
                            input: *node,
                            offset: offset_node,
                            length: length_node
                        };
                        *node = expr_arena.add(aexpr)
                    });
                    let lp = Projection {input, expr, schema};

                    self.pushdown_and_continue(lp, None, lp_arena, expr_arena)
                }
            }
            (catch_all, state) => {
                self.no_pushdown_finish_opt(catch_all, state, lp_arena)
            }

        }
    }

    pub fn optimize(
        &self,
        logical_plan: ALogicalPlan,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<ALogicalPlan> {
        self.pushdown(logical_plan, None, lp_arena, expr_arena)
    }
}
