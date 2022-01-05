use crate::prelude::*;
use polars_core::prelude::*;

pub(crate) struct SlicePushDown {}

#[derive(Copy, Clone)]
struct State {
    offset: i64,
    len: u32,
}

impl SlicePushDown {
    fn push_down(
        &self,
        lp: ALogicalPlan,
        state: Option<State>,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match (lp, state) {


            #[cfg(feature = "parquet")]
            (ParquetScan {
                path,
                schema,
                output_schema,
                predicate,
                aggregate,
                options,

            }, Some(state)) if state.offset == 0 => {
                let mut options = options;
                options.n_rows = Some(state.len as usize);
                let lp = ParquetScan {
                    path,
                    schema,
                    output_schema,
                    predicate,
                    aggregate,
                    options
                };

                Ok(lp)
            },
            #[cfg(feature = "ipc")]
            (IpcScan {path,
            schema,
                output_schema,
                predicate,
                aggregate,
                options
            }, Some(state)) if state.offset == 0 => {
                let mut options = options;
                options.n_rows = Some(state.len as usize);
                let lp = IpcScan {
                    path,
                    schema,
                    output_schema,
                    predicate,
                    aggregate,
                    options
                };

                Ok(lp)

            }

            #[cfg(feature = "csv-file")]
            (CsvScan {
                path,
                schema,
                output_schema,
                options,
                predicate,
                aggregate,
            }, Some(state)) if state.offset > 0 => {
                let mut options = options;
                options.skip_rows = state.offset as usize;
                options.n_rows = Some(state.len as usize);

                let lp = CsvScan {
                    path,
                    schema,
                    output_schema,
                    options,
                    predicate,
                    aggregate
                };
                Ok(lp)
            }

            (Union {inputs, .. }, Some(state)) => {
                let options = UnionOptions {
                    slice: true,
                    slice_offset: state.offset,
                    slice_len: state.len,
                };
                Ok(Union {inputs, options})
            },
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
                let lp = self.push_down(alp, state, lp_arena, expr_arena)?;
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
                self.push_down(alp, state, lp_arena, expr_arena)
            }

            // [Do not pushdown] boundary
            // here we do not pushdown.
            // we reset the state and then start the optimization again
            m @ (Selection { .. }, _)
            | m @ (Join { .. }, _)
            | m @ (Aggregate {..}, _)
            | m @ (DataFrameScan {..}, _)
            | m @ (Sort {..}, _)
            | m @ (Explode {..}, _)
            | m @ (Cache {..}, _)
            | m @ (Distinct {..}, _)
            | m @ (Udf {predicate_pd: false, ..}, _)
            // let's be conservative. projections may do aggregations and a pushed down slice
            // will lead to incorrect aggregations
            | m @ (Projection {..}, _)
            | m @ (LocalProjection {..},_)
            | m @ (HStack {..},_)
            => {
                let (lp, state) = m;
                let inputs = lp.get_inputs();
                let exprs = lp.get_exprs();

                let new_inputs = inputs
                    .iter()
                    .map(|&node| {
                        let alp = lp_arena.take(node);
                        // No state, so we do not push down the slice here.
                        let state = None;
                        let alp = self.push_down(alp, state, lp_arena, expr_arena)?;
                        lp_arena.replace(node, alp);
                        Ok(node)

                    })
                    .collect::<Result<Vec<_>>>()?;
                let lp = lp.from_exprs_and_input(exprs, new_inputs);

                if let Some(state) = state {
                    let input = lp_arena.add(lp);

                    let lp = ALogicalPlan::Slice {
                        input,
                        offset: state.offset,
                        len: state.len,
                    };
                    Ok(lp)

                } else {
                    Ok(lp)
                }

            }
            // [Pushdown]
            // these nodes will be pushed down.
            m @ (Melt { .. },_)
            | m @(Udf{predicate_pd: true, ..}, _)

            => {
                let (lp, state) = m;
                let inputs = lp.get_inputs();
                let exprs = lp.get_exprs();

                let new_inputs = inputs
                    .iter()
                    .map(|&node| {
                        let alp = lp_arena.take(node);
                        let alp = self.push_down(alp, state, lp_arena, expr_arena)?;
                        lp_arena.replace(node, alp);
                        Ok(node)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(lp.from_exprs_and_input(exprs, new_inputs))
            }
            (catch_all, state) => {
                assert!(state.is_none());
                Ok(catch_all)

            }

        }
    }

    pub fn optimize(
        &self,
        logical_plan: ALogicalPlan,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        self.push_down(logical_plan, None, lp_arena, expr_arena)
    }
}
