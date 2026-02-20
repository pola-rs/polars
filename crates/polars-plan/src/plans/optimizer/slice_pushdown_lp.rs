use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::slice_enum::Slice;
use recursive::recursive;

use crate::prelude::*;

mod inner {
    use polars_utils::arena::Node;
    use polars_utils::idx_vec::UnitVec;
    use polars_utils::unitvec;

    pub struct SlicePushDown {
        scratch: UnitVec<Node>,
        pub(super) maintain_errors: bool,
    }

    impl SlicePushDown {
        pub fn new(maintain_errors: bool) -> Self {
            Self {
                scratch: unitvec![],
                maintain_errors,
            }
        }

        /// Returns shared scratch space after clearing.
        pub fn empty_nodes_scratch_mut(&mut self) -> &mut UnitVec<Node> {
            self.scratch.clear();
            &mut self.scratch
        }
    }
}

pub(super) use inner::SlicePushDown;

#[derive(Copy, Clone, Debug)]
struct State {
    offset: i64,
    len: IdxSize,
}

impl State {
    fn to_slice_enum(self) -> Slice {
        let offset = self.offset;
        let len: usize = usize::try_from(self.len).unwrap();

        (offset, len).into()
    }
}

/// Can push down slice when:
/// * all projections are elementwise
/// * at least 1 projection is based on a column (for height broadcast)
/// * projections not based on any column project as scalars
///
/// Returns (can_pushdown, can_pushdown_and_any_expr_has_column)
fn can_pushdown_slice_past_projections(
    exprs: &[ExprIR],
    arena: &Arena<AExpr>,
    scratch: &mut UnitVec<Node>,
    maintain_errors: bool,
) -> (bool, bool) {
    scratch.clear();

    let mut can_pushdown_and_any_expr_has_column = false;

    for expr_ir in exprs.iter() {
        scratch.push(expr_ir.node());

        // # "has_column"
        // `select(c = Literal([1, 2, 3])).slice(0, 0)` must block slice pushdown,
        // because `c` projects to a height independent from the input height. We check
        // this by observing that `c` does not have any columns in its input nodes.
        //
        // TODO: Simply checking that a column node is present does not handle e.g.:
        // `select(c = Literal([1, 2, 3]).is_in(col(a)))`, for functions like `is_in`,
        // `str.contains`, `str.contains_any` etc. - observe a column node is present
        // but the output height is not dependent on it.
        let mut has_column = false;
        let mut literals_all_scalar = true;

        let mut pd_group = ExprPushdownGroup::Pushable;

        while let Some(node) = scratch.pop() {
            let ae = arena.get(node);

            // We re-use the logic from predicate pushdown, as slices can be seen as a form of filtering.
            // But we also do some bookkeeping here specific to slice pushdown.

            match ae {
                AExpr::Column(_) => has_column = true,
                AExpr::Literal(v) => literals_all_scalar &= v.is_scalar(),
                _ => {},
            }

            if pd_group
                .update_with_expr(scratch, ae, arena)
                .blocks_pushdown(maintain_errors)
            {
                return (false, false);
            }
        }

        // If there is no column then all literals must be scalar
        if !(has_column || literals_all_scalar) {
            return (false, false);
        }

        can_pushdown_and_any_expr_has_column |= has_column
    }

    (true, can_pushdown_and_any_expr_has_column)
}

impl SlicePushDown {
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
        &mut self,
        lp: IR,
        state: Option<State>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let inputs = lp.get_inputs();

        let new_inputs = inputs
            .into_iter()
            .map(|node| {
                let alp = lp_arena.take(node);
                // No state, so we do not push down the slice here.
                let state = None;
                let alp = self.pushdown(alp, state, lp_arena, expr_arena)?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<PolarsResult<UnitVec<_>>>()?;
        let lp = lp.with_inputs(new_inputs);

        self.no_pushdown_finish_opt(lp, state, lp_arena)
    }

    /// slice will be pushed down.
    fn pushdown_and_continue(
        &mut self,
        lp: IR,
        state: Option<State>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let inputs = lp.get_inputs();

        let new_inputs = inputs
            .into_iter()
            .map(|node| {
                let alp = lp_arena.take(node);
                let alp = self.pushdown(alp, state, lp_arena, expr_arena)?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<PolarsResult<UnitVec<_>>>()?;
        Ok(lp.with_inputs(new_inputs))
    }

    #[recursive]
    fn pushdown(
        &mut self,
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

            (Scan {
                sources,
                file_info,
                hive_parts,
                output_schema,
                mut unified_scan_args,
                predicate,
                predicate_file_skip_applied,
                scan_type,
            }, Some(state)) if predicate.is_none() && match &*scan_type {
                #[cfg(feature = "parquet")]
                FileScanIR::Parquet { .. } => true,

                #[cfg(feature = "ipc")]
                FileScanIR::Ipc { .. } => true,

                #[cfg(feature = "csv")]
                FileScanIR::Csv { .. } => true,

                #[cfg(feature = "json")]
                FileScanIR::NDJson { .. } => true,

                #[cfg(feature = "python")]
                FileScanIR::PythonDataset { .. } => true,

                #[cfg(feature = "scan_lines")]
                FileScanIR::Lines { .. } => true,

                // TODO: This can be `true` after Anonymous scan dispatches to new-streaming.
                FileScanIR::Anonymous { .. } => state.offset == 0,
            }  =>  {
                unified_scan_args.pre_slice = Some(state.to_slice_enum());

                let lp = Scan {
                    sources,
                    file_info,
                    hive_parts,
                    output_schema,
                    scan_type,
                    unified_scan_args,
                    predicate,
                    predicate_file_skip_applied,
                };

                Ok(lp)
            },

            (DataFrameScan {df, schema, output_schema, }, Some(state))  => {
                let df = df.slice(state.offset, state.len as usize);
                let lp = DataFrameScan {
                    df: Arc::new(df),
                    schema,
                    output_schema,
                };
                Ok(lp)
            }
            (Union {mut inputs, mut options }, opt_state) => {
                let subplan_slice: Option<State> = opt_state
                    .filter(|x| x.offset >= 0)
                    .and_then(|x| x.len.checked_add(x.offset.try_into().unwrap()))
                    .map(|len| State {
                        offset: 0,
                        len,
                    });

                for input in &mut inputs {
                    let input_lp = lp_arena.take(*input);
                    let input_lp = self.pushdown(input_lp, subplan_slice, lp_arena, expr_arena)?;
                    lp_arena.replace(*input, input_lp);
                }
                options.slice = opt_state.map(|x| (x.offset, x.len.try_into().unwrap()));
                let lp = Union {inputs, options};
                Ok(lp)
            },
            (Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                mut options
            }, Some(state)) if !matches!(options.options, Some(JoinTypeOptionsIR::CrossAndFilter { .. })) => {
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
            (Sort {input, by_column, slice, sort_options}, Some(state)) => {
                // The slice argument on Sort should be inserted by slice pushdown,
                // so it shouldn't exist yet (or be idempotently the same).
                let new_slice = Some((state.offset, state.len as usize, None));
                assert!(slice.is_none() || slice == new_slice);

                // first restart optimization in inputs and get the updated LP
                let input_lp = lp_arena.take(input);
                let input_lp = self.pushdown(input_lp, None, lp_arena, expr_arena)?;
                let input = lp_arena.add(input_lp);

                Ok(Sort {
                    input,
                    by_column,
                    slice: new_slice,
                    sort_options
                })
            }
            (Slice {
                input,
                offset,
                mut len
            }, Some(outer_slice)) => {
                let alp = lp_arena.take(input);

                // Both are positive, can combine into a single slice.
                if outer_slice.offset >= 0 && offset >= 0 {
                    let state = State {
                        offset: offset.checked_add(outer_slice.offset).unwrap(),
                        len: if len as i128 > outer_slice.offset as i128 {
                            (len - outer_slice.offset as IdxSize).min(outer_slice.len)
                        } else {
                            0
                        },
                    };
                    return self.pushdown(alp, Some(state), lp_arena, expr_arena);
                }

                // If offset is negative the length can never be greater than it.
                if offset < 0 {
                    #[allow(clippy::unnecessary_cast)] // Necessary when IdxSize = u64.
                    if len as u64 > offset.unsigned_abs() as u64 {
                        len = offset.unsigned_abs() as IdxSize;
                    }
                }

                // Both are negative, can also combine (but not so simply).
                if outer_slice.offset < 0 && offset < 0 {
                    // We use 128-bit arithmetic to avoid overflows, clamping at the end.
                    let inner_start_rel_end = offset as i128;
                    let inner_stop_rel_end = inner_start_rel_end + len as i128;
                    let naive_outer_start_rel_end = inner_stop_rel_end + outer_slice.offset as i128;
                    let naive_outer_stop_rel_end = naive_outer_start_rel_end + outer_slice.len as i128;
                    let clamped_outer_start_rel_end = naive_outer_start_rel_end.max(inner_start_rel_end);
                    let clamped_outer_stop_rel_end = naive_outer_stop_rel_end.max(clamped_outer_start_rel_end);

                    let state = State {
                        offset: clamped_outer_start_rel_end.clamp(i64::MIN as i128, i64::MAX as i128) as i64,
                        len: (clamped_outer_stop_rel_end - clamped_outer_start_rel_end).min(IdxSize::MAX as i128) as IdxSize,
                    };
                    return self.pushdown(alp, Some(state), lp_arena, expr_arena);
                }

                let inner_slice = Some(State { offset, len });
                let lp = self.pushdown(alp, inner_slice, lp_arena, expr_arena)?;
                let input = lp_arena.add(lp);
                Ok(Slice {
                    input,
                    offset: outer_slice.offset,
                    len: outer_slice.len
                })
            }
            (Slice {
                input,
                offset,
                mut len
            }, None) => {
                let alp = lp_arena.take(input);

                // If offset is negative the length can never be greater than it.
                if offset < 0 {
                    #[allow(clippy::unnecessary_cast)] // Necessary when IdxSize = u64.
                    if len as u64 > offset.unsigned_abs() as u64 {
                        len = offset.unsigned_abs() as IdxSize;
                    }
                }

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
            m @ (Select {..}, None)
            | m @ (HStack {..}, None)
            | m @ (SimpleProjection {..}, _)
            => {
                let (lp, state) = m;
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            }
            // there is state, inspect the projection to determine how to deal with it
            (Select {input, expr, schema, options}, Some(_)) => {
                let maintain_errors = self.maintain_errors;
                if can_pushdown_slice_past_projections(&expr, expr_arena, self.empty_nodes_scratch_mut(), maintain_errors).1 {
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
                let maintain_errors = self.maintain_errors;
                let (can_pushdown, can_pushdown_and_any_expr_has_column) = can_pushdown_slice_past_projections(&exprs, expr_arena, self.empty_nodes_scratch_mut(), maintain_errors);

                if can_pushdown_and_any_expr_has_column || (
                    // If the schema length is greater then an input column is being projected, so
                    // the exprs in with_columns do not need to have an input column name.
                    schema.len() > exprs.len() && can_pushdown
                )
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
            (lp @ Sink { .. }, _) | (lp @ SinkMultiple { .. }, _) => {
                // Slice can always be pushed down for sinks
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            }
            (catch_all, state) => {
                self.no_pushdown_finish_opt(catch_all, state, lp_arena)
            }
        }
    }

    pub fn optimize(
        &mut self,
        logical_plan: IR,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        self.pushdown(logical_plan, None, lp_arena, expr_arena)
    }
}
