use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::scratch_vec::{ScratchUnitVec, ScratchVec};
use polars_utils::slice_enum::Slice;
use polars_utils::unique_id::UniqueId;
use recursive::recursive;

use crate::plans::optimizer::slice_pushdown_expr;
use crate::prelude::*;

pub(super) struct SlicePushDown {
    nodes_scratch: ScratchUnitVec<Node>,
    maintain_errors: bool,
    pub(super) slice_node_in_optimized_plan: bool,
    pub(super) ae_nodes_scratch: ScratchVec<Node>,
    pub(super) ae_slice_pd_state_scratch: ScratchVec<slice_pushdown_expr::State>,
    optimized_cache_inputs: PlIndexMap<UniqueId, Node>,
}

impl SlicePushDown {
    pub(super) fn new() -> Self {
        Self {
            nodes_scratch: ScratchUnitVec::default(),
            // We don't maintain errors on slice to make the behavior predictable across engines.
            //
            // Even if we enable maintain_errors (thereby preventing the slice from being pushed),
            // the new-streaming engine still may not error due to early-stopping.
            maintain_errors: false,
            slice_node_in_optimized_plan: false,
            ae_nodes_scratch: ScratchVec::default(),
            ae_slice_pd_state_scratch: ScratchVec::default(),
            optimized_cache_inputs: PlIndexMap::default(),
        }
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
        &mut self,
        lp: IR,
        state: Option<Slice>,
        lp_arena: &mut Arena<IR>,
    ) -> PolarsResult<IR> {
        match state {
            Some(state) => {
                self.slice_node_in_optimized_plan = true;
                let input = lp_arena.add(lp);

                let lp = IR::Slice {
                    input,
                    slice: state,
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
        state: Option<Slice>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let inputs = lp.get_inputs();

        let new_inputs = inputs
            .into_iter()
            .map(|node| {
                // No state, so we do not push down the slice here.
                let state = None;
                let alp = self.pushdown(node, state, lp_arena, expr_arena)?;
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
        state: Option<Slice>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let inputs = lp.get_inputs();

        let new_inputs = inputs
            .into_iter()
            .map(|node| {
                let alp = self.pushdown(node, state, lp_arena, expr_arena)?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<PolarsResult<UnitVec<_>>>()?;
        Ok(lp.with_inputs(new_inputs))
    }

    /// This will take the `ir_node` from the `lp_arena`, replacing it with `IR::Invalid` (except if
    /// `ir_node` is a `IR::Cache`).
    #[recursive]
    fn pushdown(
        &mut self,
        ir_node: Node,
        state: Option<Slice>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        use IR::*;
        use polars_utils::slice_enum::Slice;

        // Cache nodes with the same cache id share one logical input.
        // Optimize that input once, then reuse it for later cache nodes
        // to avoid traversing a subtree that may already have been taken
        // from the arena.
        if let IR::Cache { input, id } = lp_arena.get(ir_node) {
            let (mut input, id) = (*input, *id);
            input = if let Some(input) = self.optimized_cache_inputs.get(&id) {
                *input
            } else {
                let alp = self.pushdown(input, None, lp_arena, expr_arena)?;
                lp_arena.replace(input, alp);
                self.optimized_cache_inputs.insert(id, input);
                input
            };

            return self.no_pushdown_finish_opt(IR::Cache { input, id }, state, lp_arena);
        }

        let maintain_errors = self.maintain_errors;

        let ir = lp_arena.get_mut(ir_node);

        let mut common_expr_col_slice = None;
        let mut col_hit_count: Option<usize> = Some(0);

        for eir in ir.exprs() {
            self.aexpr_slice_pushdown_rec(
                eir.node(),
                expr_arena,
                &mut common_expr_col_slice,
                &mut col_hit_count,
                maintain_errors,
            );
        }

        if col_hit_count != Some(0) {
            common_expr_col_slice = None;
        }

        'add_common_ir_slice: {
            let IR::Select { input, .. } = ir else {
                break 'add_common_ir_slice;
            };

            let Some((ir_slice, 0)) = common_expr_col_slice.zip(col_hit_count) else {
                break 'add_common_ir_slice;
            };

            let input = *input;
            let new_input_node = lp_arena.add(IR::Slice {
                input,
                slice: ir_slice,
            });

            let IR::Select { input, expr, .. } = lp_arena.get_mut(ir_node) else {
                unreachable!()
            };

            *input = new_input_node;

            // Every slice on a column in these expressions is contained in `ir_slice`.
            let col_slice_nodes: PlIndexSet<Node> = expr
                .iter()
                .flat_map(|e| expr_arena.iter(e.node()))
                .filter_map(|(node, ae)| match ae {
                    AExpr::Slice { input, .. }
                        if matches!(expr_arena.get(*input), AExpr::Column(_)) =>
                    {
                        Some(node)
                    },
                    _ => None,
                })
                .collect();

            for node in col_slice_nodes {
                use slice_pushdown_expr::ExprSlice;
                let AExpr::Slice {
                    input,
                    offset,
                    length,
                } = *expr_arena.get(node)
                else {
                    unreachable!()
                };

                let ExprSlice::Extracted(ae_slice) =
                    ExprSlice::from_nodes(offset, length, expr_arena)
                else {
                    unreachable!()
                };

                expr_arena.replace(node, expr_arena.get(input).clone());
                // `ir_slice` encloses every column slice, so `try_within` is `None` only for an
                // open-ended slice equal to `ir_slice`.
                if ae_slice != ir_slice {
                    let slice_within_ir_slice = ae_slice.try_within(ir_slice).unwrap();
                    ExprSlice::Extracted(slice_within_ir_slice).slice_arena_node(node, expr_arena);
                }
            }
        }

        match (lp_arena.take(ir_node), state) {
            #[cfg(feature = "python")]
            (
                PythonScan { mut options },
                // TODO! we currently skip slice pushdown if there is a predicate.
                // we can modify the readers to only limit after predicates have been applied
                Some(state),
            ) if matches!(state, Slice::Positive { offset: 0, .. })
                && matches!(options.predicate, PythonPredicate::None) =>
            {
                options.n_rows = state.len();
                let lp = PythonScan { options };
                Ok(lp)
            },

            (
                Scan {
                    sources,
                    file_info,
                    hive_parts,
                    output_schema,
                    mut unified_scan_args,
                    predicate,
                    predicate_file_skip_applied,
                    scan_type,
                    maintain_order,
                },
                Some(state),
            ) if predicate.is_none()
                && match &*scan_type {
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

                    FileScanIR::ExpandedPaths { .. } => false,

                    FileScanIR::ExternalReaderBuilder { .. } => true,

                    // TODO: This can be `true` after Anonymous scan dispatches to new-streaming.
                    FileScanIR::Anonymous { .. } => {
                        state.is_positive() && state.positive_offset() == 0
                    },
                } =>
            {
                if let Some(existing) = &mut unified_scan_args.pre_slice {
                    return if let Some(combined) = existing.reslice(state) {
                        *existing = combined;

                        let lp = Scan {
                            sources,
                            file_info,
                            hive_parts,
                            output_schema,
                            scan_type,
                            unified_scan_args,
                            predicate,
                            predicate_file_skip_applied,
                            maintain_order,
                        };

                        lp_arena.replace(ir_node, lp);
                        self.pushdown(ir_node, None, lp_arena, expr_arena)
                    } else {
                        let lp = Scan {
                            sources,
                            file_info,
                            hive_parts,
                            output_schema,
                            scan_type,
                            unified_scan_args,
                            predicate,
                            predicate_file_skip_applied,
                            maintain_order,
                        };

                        self.no_pushdown_restart_opt(lp, Some(state), lp_arena, expr_arena)
                    };
                }

                unified_scan_args.pre_slice = Some(state);

                let lp = Scan {
                    sources,
                    file_info,
                    hive_parts,
                    output_schema,
                    scan_type,
                    unified_scan_args,
                    predicate,
                    predicate_file_skip_applied,
                    maintain_order,
                };

                Ok(lp)
            },

            (
                DataFrameScan {
                    df,
                    schema,
                    output_schema,
                },
                Some(state),
            ) => {
                let df = df.slice(state);
                let lp = DataFrameScan {
                    df: Arc::new(df),
                    schema,
                    output_schema,
                };
                Ok(lp)
            },
            (
                Union {
                    mut inputs,
                    mut options,
                },
                opt_state,
            ) => {
                if let Some(existing_slice) = options.slice.as_mut() {
                    return if let Some(outer_slice) = opt_state
                        && let Some(combined) = existing_slice.reslice(outer_slice)
                    {
                        *existing_slice = combined;
                        let lp = Union { inputs, options };
                        self.pushdown_and_continue(lp, None, lp_arena, expr_arena)
                    } else {
                        let lp = Union { inputs, options };
                        self.no_pushdown_restart_opt(lp, opt_state, lp_arena, expr_arena)
                    };
                }

                let subplan_slice = opt_state
                    .filter(|x| x.is_positive())
                    .and_then(|x| x.bounding_limit());

                for input in &mut inputs {
                    let input_lp = self.pushdown(*input, subplan_slice, lp_arena, expr_arena)?;
                    lp_arena.replace(*input, input_lp);
                }
                options.slice = opt_state;
                let lp = Union { inputs, options };
                Ok(lp)
            },
            (
                Join {
                    input_left,
                    input_right,
                    schema,
                    mut options,
                },
                Some(state),
            ) if !matches!(options.options, JoinTypeOptionsIR::CrossAndFilter { .. })
                && !options.options.has_fused_predicate() =>
            {
                if let Some(existing_slice) = &mut Arc::make_mut(&mut options).args.slice {
                    return if let Some(combined) = existing_slice.reslice(state) {
                        *existing_slice = combined;
                        let lp = Join {
                            input_left,
                            input_right,
                            schema,
                            options,
                        };
                        self.pushdown_and_continue(lp, None, lp_arena, expr_arena)
                    } else {
                        let lp = Join {
                            input_left,
                            input_right,
                            schema,
                            options,
                        };
                        self.no_pushdown_restart_opt(lp, Some(state), lp_arena, expr_arena)
                    };
                }

                // For left/right/full joins we can push a limit.
                let input_limit_slice = state.bounding_limit();

                let order = options.args.maintain_order;
                let non_negative_offset = state.is_positive();
                let can_limit_left = match options.args.how {
                    JoinType::Left => !matches!(
                        order,
                        MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft
                    ),
                    JoinType::Full => {
                        non_negative_offset
                            && matches!(
                                order,
                                MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight
                            )
                    },
                    _ => false,
                };
                let can_limit_right = match options.args.how {
                    JoinType::Right => !matches!(
                        order,
                        MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight
                    ),
                    JoinType::Full => {
                        non_negative_offset
                            && matches!(
                                order,
                                MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft
                            )
                    },
                    _ => false,
                };

                let lp_left = self.pushdown(
                    input_left,
                    input_limit_slice.filter(|_| can_limit_left),
                    lp_arena,
                    expr_arena,
                )?;
                let input_left = lp_arena.add(lp_left);

                let lp_right = self.pushdown(
                    input_right,
                    input_limit_slice.filter(|_| can_limit_right),
                    lp_arena,
                    expr_arena,
                )?;
                let input_right = lp_arena.add(lp_right);

                // then assign the slice state to the join operation

                Arc::make_mut(&mut options).args.slice = Some(state);

                Ok(Join {
                    input_left,
                    input_right,
                    schema,
                    options,
                })
            },
            (
                GroupBy {
                    input,
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    mut options,
                },
                Some(state),
            ) => {
                // first restart optimization in inputs and get the updated LP
                let input_lp = self.pushdown(input, None, lp_arena, expr_arena)?;
                let input = lp_arena.add(input_lp);

                if let Some(existing_slice) = &mut Arc::make_mut(&mut options).slice {
                    return if let Some(combined) = existing_slice.reslice(state) {
                        *existing_slice = combined;
                        let lp = GroupBy {
                            input,
                            keys,
                            aggs,
                            schema,
                            apply,
                            maintain_order,
                            options,
                        };
                        self.pushdown_and_continue(lp, None, lp_arena, expr_arena)
                    } else {
                        let lp = GroupBy {
                            input,
                            keys,
                            aggs,
                            schema,
                            apply,
                            maintain_order,
                            options,
                        };
                        self.no_pushdown_restart_opt(lp, Some(state), lp_arena, expr_arena)
                    };
                }

                let mut_options = Arc::make_mut(&mut options);
                mut_options.slice = Some(state);

                Ok(GroupBy {
                    input,
                    keys,
                    aggs,
                    schema,
                    apply,
                    maintain_order,
                    options,
                })
            },
            (Distinct { input, mut options }, Some(state)) => {
                // first restart optimization in inputs and get the updated LP
                let input_lp = self.pushdown(input, None, lp_arena, expr_arena)?;
                let input = lp_arena.add(input_lp);

                if let Some(existing_slice) = &mut options.slice {
                    return if let Some(combined) = existing_slice.reslice(state) {
                        *existing_slice = combined;
                        let lp = Distinct { input, options };
                        self.pushdown_and_continue(lp, None, lp_arena, expr_arena)
                    } else {
                        let lp = Distinct { input, options };
                        self.no_pushdown_restart_opt(lp, Some(state), lp_arena, expr_arena)
                    };
                }

                options.slice = Some(state);
                Ok(Distinct { input, options })
            },
            (
                Sort {
                    input,
                    by_column,
                    mut slice,
                    sort_options,
                },
                Some(state),
            ) => {
                if let Some(existing_slice) = &mut slice {
                    let existing_dyn_predicate = &existing_slice.1;
                    return if existing_dyn_predicate.is_none()
                        && let Some(combined) = existing_slice.0.reslice(state)
                    {
                        *existing_slice = (combined, None);
                        let lp = Sort {
                            input,
                            by_column,
                            slice,
                            sort_options,
                        };
                        self.pushdown_and_continue(lp, None, lp_arena, expr_arena)
                    } else {
                        let lp = Sort {
                            input,
                            by_column,
                            slice,
                            sort_options,
                        };
                        self.no_pushdown_restart_opt(lp, Some(state), lp_arena, expr_arena)
                    };
                }

                let new_slice = Some((state, None));
                assert!(slice.is_none() || slice == new_slice);

                // first restart optimization in inputs and get the updated LP
                let input_lp = self.pushdown(input, None, lp_arena, expr_arena)?;
                let input = lp_arena.add(input_lp);

                Ok(Sort {
                    input,
                    by_column,
                    slice: new_slice,
                    sort_options,
                })
            },
            (IR::Slice { input, slice }, Some(outer_slice)) => {
                if let Some(combined) = slice.reslice(outer_slice) {
                    self.pushdown(input, Some(combined), lp_arena, expr_arena)
                } else {
                    let lp = self.pushdown(input, Some(slice), lp_arena, expr_arena)?;
                    let input = lp_arena.add(lp);
                    self.slice_node_in_optimized_plan = true;
                    Ok(IR::Slice {
                        input,
                        slice: outer_slice,
                    })
                }
            },
            (IR::Slice { input, slice }, None) => {
                self.pushdown(input, Some(slice), lp_arena, expr_arena)
            },
            m @ (Filter { .. }, _)
            | m @ (DataFrameScan { .. }, _)
            | m @ (Sort { .. }, _)
            | m @ (
                MapFunction {
                    function: FunctionIR::Explode { .. },
                    ..
                },
                _,
            )
            | m @ (Cache { .. }, _)
            | m @ (Distinct { .. }, _)
            | m @ (GroupBy { .. }, _)
            | m @ (Join { .. }, _) => {
                let (lp, state) = m;
                self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
            },
            #[cfg(feature = "pivot")]
            m @ (
                MapFunction {
                    function: FunctionIR::Unpivot { .. },
                    ..
                },
                _,
            ) => {
                let (lp, state) = m;
                self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
            },
            // A positive slice of a row index over a resolver: the resolver reads only the
            // slice, numbered from where the slice starts (else the in-memory engine reads the
            // whole source).
            (
                MapFunction {
                    input,
                    function:
                        FunctionIR::RowIndex {
                            name,
                            offset,
                            schema,
                        },
                },
                Some(state),
            ) if state.is_positive()
                && matches!(
                    lp_arena.get(input),
                    Resolver { resolved_ir: None, filters, .. } if filters.is_empty()
                ) =>
            {
                let shifted = IdxSize::try_from(state.positive_offset())
                    .ok()
                    .and_then(|o| offset.unwrap_or(0).checked_add(o));
                let lp = MapFunction {
                    input,
                    function: FunctionIR::RowIndex {
                        name,
                        offset: shifted.or(offset),
                        schema,
                    },
                };
                match shifted {
                    Some(_) => self.pushdown_and_continue(lp, Some(state), lp_arena, expr_arena),
                    None => self.no_pushdown_restart_opt(lp, Some(state), lp_arena, expr_arena),
                }
            },
            (MapFunction { input, function }, _) if function.allow_predicate_pd() => {
                let lp = MapFunction { input, function };
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            },
            m @ (MapFunction { .. }, _) => {
                let (lp, state) = m;
                self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
            },
            m @ (Select { .. }, None)
            | m @ (HStack { .. }, None)
            | m @ (SimpleProjection { .. }, _) => {
                let (lp, state) = m;
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            },
            (
                Select {
                    input,
                    expr,
                    schema,
                    options,
                },
                Some(_),
            ) => {
                let maintain_errors = self.maintain_errors;
                if can_pushdown_slice_past_projections(
                    &expr,
                    expr_arena,
                    self.nodes_scratch.get(),
                    maintain_errors,
                )
                .1
                {
                    let lp = Select {
                        input,
                        expr,
                        schema,
                        options,
                    };
                    self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
                }
                // don't push down slice, but restart optimization
                else {
                    let lp = Select {
                        input,
                        expr,
                        schema,
                        options,
                    };
                    self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
                }
            },
            (
                HStack {
                    input,
                    exprs,
                    schema,
                    options,
                },
                _,
            ) => {
                let maintain_errors = self.maintain_errors;
                let (can_pushdown, can_pushdown_and_any_expr_has_column) =
                    can_pushdown_slice_past_projections(
                        &exprs,
                        expr_arena,
                        self.nodes_scratch.get(),
                        maintain_errors,
                    );

                if can_pushdown_and_any_expr_has_column
                    || (
                        // If the schema length is greater then an input column is being projected, so
                        // the exprs in with_columns do not need to have an input column name.
                        schema.len() > exprs.len() && can_pushdown
                    )
                {
                    let lp = HStack {
                        input,
                        exprs,
                        schema,
                        options,
                    };
                    self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
                }
                // don't push down slice, but restart optimization
                else {
                    let lp = HStack {
                        input,
                        exprs,
                        schema,
                        options,
                    };
                    self.no_pushdown_restart_opt(lp, state, lp_arena, expr_arena)
                }
            },
            (
                HConcat {
                    inputs,
                    schema,
                    options,
                },
                Some(inner_state),
            ) if inner_state.is_negative() => {
                // Negative offset cannot push through hconcat, as this would
                // cause misalignment on inputs with non-equal lengths.
                // https://github.com/pola-rs/polars/issues/27552
                let lp = HConcat {
                    inputs,
                    schema,
                    options,
                };
                self.no_pushdown_finish_opt(lp, Some(inner_state), lp_arena)
            },
            (
                HConcat {
                    inputs,
                    schema,
                    options,
                },
                _,
            ) => {
                let lp = HConcat {
                    inputs,
                    schema,
                    options,
                };
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            },
            (lp @ Sink { .. }, _) | (lp @ SinkMultiple { .. }, _) => {
                // Slice can always be pushed down for sinks
                self.pushdown_and_continue(lp, state, lp_arena, expr_arena)
            },
            // Already resolved: `resolved_ir` is the single input of this node, so we
            // push the state into it through the normal input dispatch. Note that this
            // must go through `pushdown()` on the resolved node itself, so that node's
            // own rules apply (e.g. a `Sort` or `Filter` root must not be crossed).
            (
                lp @ Resolver {
                    resolved_ir: Some(_),
                    ..
                },
                state,
            ) => self.pushdown_and_continue(lp, state, lp_arena, expr_arena),
            (
                Resolver {
                    resolver,
                    resolver_schema,
                    projection,
                    mut slice,
                    filters,
                    filter_drop_columns_idx,
                    resolved_dsl,
                    resolved_ir,
                },
                Some(mut state),
            ) if filters.is_empty() => {
                state = if let Some(existing) = slice {
                    let Some(combined) = existing.reslice(state) else {
                        return self.no_pushdown_restart_opt(
                            Resolver {
                                resolver,
                                resolver_schema,
                                projection,
                                slice,
                                filters,
                                filter_drop_columns_idx,
                                resolved_dsl,
                                resolved_ir,
                            },
                            Some(state),
                            lp_arena,
                            expr_arena,
                        );
                    };
                    combined
                } else {
                    state
                };

                slice = Some(state);

                Ok(Resolver {
                    resolver,
                    resolver_schema,
                    projection,
                    slice,
                    filters,
                    filter_drop_columns_idx,
                    resolved_dsl,
                    resolved_ir,
                })
            },
            (catch_all, state) => self.no_pushdown_finish_opt(catch_all, state, lp_arena),
        }
    }

    pub fn optimize(
        &mut self,
        logical_plan: Node,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        self.pushdown(logical_plan, None, lp_arena, expr_arena)
    }
}
