use polars_core::prelude::*;
use recursive::recursive;

use crate::prelude::*;

pub(super) struct SlicePushDown {
    streaming: bool,
    pub scratch: Vec<Node>,
}

#[derive(Copy, Clone)]
struct State {
    offset: i64,
    len: IdxSize,
}

/// Can push down slice when:
/// * all projections are elementwise
/// * at least 1 projection is based on a column (for height broadcast)
/// * projections not based on any column project as scalars
///
/// Returns (all_elementwise, all_elementwise_and_any_expr_has_column)
fn can_pushdown_slice_past_projections(exprs: &[ExprIR], arena: &Arena<AExpr>) -> (bool, bool) {
    let mut all_elementwise_and_any_expr_has_column = false;
    for expr_ir in exprs.iter() {
        // `select(c = Literal([1, 2, 3])).slice(0, 0)` must block slice pushdown,
        // because `c` projects to a height independent from the input height. We check
        // this by observing that `c` does not have any columns in its input nodes.
        //
        // TODO: Simply checking that a column node is present does not handle e.g.:
        // `select(c = Literal([1, 2, 3]).is_in(col(a)))`, for functions like `is_in`,
        // `str.contains`, `str.contains_many` etc. - observe a column node is present
        // but the output height is not dependent on it.
        let is_elementwise = is_streamable(expr_ir.node(), arena, Context::Default);
        let (has_column, literals_all_scalar) = arena.iter(expr_ir.node()).fold(
            (false, true),
            |(has_column, lit_scalar), (_node, ae)| {
                (
                    has_column | matches!(ae, AExpr::Column(_)),
                    lit_scalar
                        & if let AExpr::Literal(v) = ae {
                            v.projects_as_scalar()
                        } else {
                            true
                        },
                )
            },
        );

        // If there is no column then all literals must be scalar
        if !is_elementwise || !(has_column || literals_all_scalar) {
            return (false, false);
        }

        all_elementwise_and_any_expr_has_column |= has_column
    }

    (true, all_elementwise_and_any_expr_has_column)
}

impl SlicePushDown {
    pub(super) fn new(streaming: bool) -> Self {
        Self {
            streaming,
            scratch: vec![],
        }
    }

    // slice will be done at this node if we found any
    // we also stop optimization
    fn no_pushdown_finish_opt(
        &self,
        lp: IR,
        state: Option<State>,
        lp_arena: &mut Arena<IR>,
    ) -> PolarsResult<IR> {
        match state {
            Some(state) => {
                let input = lp_arena.add(lp);

                let lp = IR::Slice {
                    input,
                    offset: state.offset,
                    len: state.len,
                };
                Ok(lp)
            },
            None => Ok(lp),
        }
    }

    /// slice will be done at this node, but we continue optimization
    fn no_pushdown_restart_opt(
        &self,
        lp: IR,
        state: Option<State>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
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
        lp: IR,
        state: Option<State>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
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

    #[recursive]
    fn pushdown(
        &self,
        lp: IR,
        state: Option<State>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        use IR::*;

        match (lp, state) {
            #[cfg(feature = "python")]
            (PythonScan {
                mut options,
            },
            // TODO! we currently skip slice pushdown if there is a predicate.
            // we can modify the readers to only limit after predicates have been applied
                Some(state)) if state.offset == 0 && matches!(options.predicate, PythonPredicate::None) => {
                options.n_rows = Some(state.len as usize);
                let lp = PythonScan {
                    options,
                };
                Ok(lp)
            }
            #[cfg(feature = "csv")]
            (Scan {
                paths,
                file_info,
                hive_parts,
                output_schema,
                mut file_options,
                predicate,
                scan_type: FileScan::Csv { options, cloud_options },
            }, Some(state)) if predicate.is_none() && state.offset >= 0 =>  {
                file_options.slice = Some((0, state.offset as usize + state.len as usize));

                let lp = Scan {
                    paths,
                    file_info,
                    hive_parts,
                    output_schema,
                    scan_type: FileScan::Csv { options, cloud_options },
                    file_options,
                    predicate,
                };

                self.no_pushdown_finish_opt(lp, Some(state), lp_arena)
            },
            #[cfg(feature = "parquet")]
            (Scan {
                paths,
                file_info,
                hive_parts,
                output_schema,
                mut file_options,
                predicate,
                scan_type: scan_type @ FileScan::Parquet { .. },
            }, Some(state)) if predicate.is_none() =>  {
                file_options.slice = Some((state.offset, state.len as usize));

                let lp = Scan {
                    paths,
                    file_info,
                    hive_parts,
                    output_schema,
                    scan_type,
                    file_options,
                    predicate,
                };

                Ok(lp)
            },
            // TODO! we currently skip slice pushdown if there is a predicate.
            (Scan {
                paths,
                file_info,
                hive_parts,
                output_schema,
                file_options: mut options,
                predicate,
                scan_type
            }, Some(state)) if state.offset == 0 && predicate.is_none() => {
                options.slice = Some((0, state.len as usize));

                let lp = Scan {
                    paths,
                    file_info,
                    hive_parts,
                    output_schema,
                    predicate,
                    file_options: options,
                    scan_type
                };

                Ok(lp)
            },
            (DataFrameScan {df, schema, output_schema, filter, }, Some(state)) if filter.is_none() => {
                let df = df.slice(state.offset, state.len as usize);
                let lp = DataFrameScan {
                    df: Arc::new(df),
                    schema,
                    output_schema,
                    filter
                };
                Ok(lp)
            }
            (Union {mut inputs, mut options }, Some(state)) => {
                if state.offset == 0 {
                    for input in &mut inputs {
                        let input_lp = lp_arena.take(*input);
                        let input_lp = self.pushdown(input_lp, Some(state), lp_arena, expr_arena)?;
                        lp_arena.replace(*input, input_lp);
                    }
                }
                // The in-memory union node is slice aware.
                // We still set this information, but the streaming engine will ignore it.
                options.slice = Some((state.offset, state.len as usize));
                let lp = Union {inputs, options};

                if self.streaming {
                    // Ensure the slice node remains.
                    self.no_pushdown_finish_opt(lp, Some(state), lp_arena)
                } else {
                    Ok(lp)
                }
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

                let mut_options = Arc::make_mut(&mut options);
                mut_options.args.slice = Some((state.offset, state.len as usize));

                Ok(Join {
                    input_left,
                    input_right,
                    schema,
                    left_on,
                    right_on,
                    options
                })
            }
            (GroupBy { input, keys, aggs, schema, apply, maintain_order, mut options }, Some(state)) => {
                // first restart optimization in inputs and get the updated LP
                let input_lp = lp_arena.take(input);
                let input_lp = self.pushdown(input_lp, None, lp_arena, expr_arena)?;
                let input= lp_arena.add(input_lp);

                let mut_options= Arc::make_mut(&mut options);
                mut_options.slice = Some((state.offset, state.len as usize));

                Ok(GroupBy {
                    input,
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    options
                })
            }
            (Distinct {input, mut options}, Some(state)) => {
                // first restart optimization in inputs and get the updated LP
                let input_lp = lp_arena.take(input);
                let input_lp = self.pushdown(input_lp, None, lp_arena, expr_arena)?;
                let input= lp_arena.add(input_lp);
                options.slice = Some((state.offset, state.len as usize));
                Ok(Distinct {
                    input,
                    options,
                })
            }
            (Sort {input, by_column, mut slice,
                sort_options}, Some(state)) => {
                // first restart optimization in inputs and get the updated LP
                let input_lp = lp_arena.take(input);
                let input_lp = self.pushdown(input_lp, None, lp_arena, expr_arena)?;
                let input= lp_arena.add(input_lp);

                slice = Some((state.offset, state.len as usize));
                Ok(Sort {
                    input,
                    by_column,
                    slice,
                    sort_options
                })
            }
            (Slice {
                input,
                offset,
                len
            }, Some(previous_state)) => {
                let alp = lp_arena.take(input);
                let state = Some(if previous_state.offset == offset  {
                    State {
                        offset,
                        len: std::cmp::min(len, previous_state.len)
                    }
                } else {
                    State {
                        offset,
                        len
                    }
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
            m @ (Filter { .. }, _)
            // other blocking nodes
            | m @ (DataFrameScan {..}, _)
            | m @ (Sort {..}, _)
            | m @ (MapFunction {function: FunctionIR::Explode {..}, ..}, _)
            | m @ (Cache {..}, _)
            | m @ (Distinct {..}, _)
            | m @ (GroupBy{..},_)
            // blocking in streaming
            | m @ (Join{..},_)
            => {
                let (lp, state) = m;
                self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
            },
            #[cfg(feature = "pivot")]
             m @ (MapFunction {function: FunctionIR::Unpivot {..}, ..}, _) => {
                let (lp, state) = m;
                self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
            },
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
            m @(Select {..}, None) |
            m @ (SimpleProjection {..}, _)
            => {
                let (lp, state) = m;
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            }
            // there is state, inspect the projection to determine how to deal with it
            (Select {input, expr, schema, options}, Some(_)) => {
                if can_pushdown_slice_past_projections(&expr, expr_arena).1 {
                    let lp = Select {input, expr, schema, options};
                    self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
                }
                // don't push down slice, but restart optimization
                else {
                    let lp = Select {input, expr, schema, options};
                    self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
                }
            }
            (HStack {input, exprs, schema, options}, _) => {
                let (can_pushdown, all_elementwise_and_any_expr_has_column) = can_pushdown_slice_past_projections(&exprs, expr_arena);

                if (
                    // If the schema length is greater than an input column is being projected, so
                    // the exprs in with_columns do not need to have an input column name.
                    schema.len() > exprs.len() && can_pushdown
                )
                || all_elementwise_and_any_expr_has_column // e.g. select(c).with_columns(c = c + 1)
                {
                    let lp = HStack {input, exprs, schema, options};
                    self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
                }
                // don't push down slice, but restart optimization
                else {
                    let lp = HStack {input, exprs, schema, options};
                    self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
                }
            }
            (HConcat {inputs, schema, options}, _) => {
                // Slice can always be pushed down for horizontal concatenation
                let lp = HConcat {inputs, schema, options};
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            }
            (catch_all, state) => {
                self.no_pushdown_finish_opt(catch_all, state, lp_arena)
            }
        }
    }

    pub fn optimize(
        &self,
        logical_plan: IR,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        self.pushdown(logical_plan, None, lp_arena, expr_arena)
    }
}
