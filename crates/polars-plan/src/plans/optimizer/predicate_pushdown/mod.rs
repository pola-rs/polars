mod dynamic;
mod group_by;
mod join;
mod keys;
mod utils;

pub use dynamic::{DynamicPred, PredicateExpr, TrivialPredicateExpr};
use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use recursive::recursive;
use utils::*;

use super::*;
use crate::plans::optimizer::predicate_pushdown::dynamic::new_dynamic_pred;
use crate::prelude::optimizer::predicate_pushdown::group_by::process_group_by;
use crate::prelude::optimizer::predicate_pushdown::join::process_join;
use crate::utils::{check_input_node, has_aexpr};

/// The struct is wrapped in a mod to prevent direct member access of `nodes_scratch`
mod inner {
    use polars_utils::arena::Node;
    use polars_utils::idx_vec::UnitVec;
    use polars_utils::unitvec;

    pub struct PredicatePushDown {
        // How many cache nodes a predicate may be pushed down to.
        // Normally this is 0. Only needed for CSPE.
        pub(super) caches_pass_allowance: u32,
        nodes_scratch: UnitVec<Node>,
        pub(super) new_streaming: bool,
        // Controls pushing filters past fallible projections
        pub(super) maintain_errors: bool,
    }

    impl PredicatePushDown {
        pub fn new(maintain_errors: bool, new_streaming: bool) -> Self {
            Self {
                caches_pass_allowance: 0,
                nodes_scratch: unitvec![],
                new_streaming,
                maintain_errors,
            }
        }

        /// Returns shared scratch space after clearing.
        pub(super) fn empty_nodes_scratch_mut(&mut self) -> &mut UnitVec<Node> {
            self.nodes_scratch.clear();
            &mut self.nodes_scratch
        }
    }
}

pub use inner::PredicatePushDown;

impl PredicatePushDown {
    pub(crate) fn block_at_cache(mut self, count: u32) -> Self {
        self.caches_pass_allowance = count;
        self
    }

    fn optional_apply_predicate(
        &mut self,
        lp: IR,
        local_predicates: Vec<ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> IR {
        if !local_predicates.is_empty() {
            let predicate = combine_predicates(local_predicates.into_iter(), expr_arena);
            let input = lp_arena.add(lp);

            IR::Filter { input, predicate }
        } else {
            lp
        }
    }

    fn pushdown_and_assign(
        &mut self,
        input: Node,
        acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<()> {
        let alp = lp_arena.take(input);
        let lp = self.push_down(alp, acc_predicates, lp_arena, expr_arena)?;
        lp_arena.replace(input, lp);
        Ok(())
    }

    /// Filter will be pushed down.
    fn pushdown_and_continue(
        &mut self,
        lp: IR,
        mut acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        has_projections: bool,
    ) -> PolarsResult<IR> {
        if has_projections {
            let input = {
                let mut inputs = lp.inputs();
                let input = inputs.next().unwrap();
                // projections should only have a single input.
                if inputs.next().is_some() {
                    // except for ExtContext
                    assert!(matches!(lp, IR::ExtContext { .. }));
                }
                input
            };

            let maintain_errors = self.maintain_errors;
            let (eligibility, alias_rename_map) = pushdown_eligibility(
                &lp.exprs().cloned().collect::<Vec<_>>(),
                &[],
                &acc_predicates,
                expr_arena,
                self.empty_nodes_scratch_mut(),
                maintain_errors,
                lp_arena.get(input),
            )?;

            let local_predicates = match eligibility {
                PushdownEligibility::Full => vec![],
                PushdownEligibility::Partial { to_local } => {
                    let mut out = Vec::with_capacity(to_local.len());
                    for key in to_local {
                        out.push(acc_predicates.remove(&key).unwrap());
                    }
                    out
                },
                PushdownEligibility::NoPushdown => {
                    return self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena);
                },
            };

            if !alias_rename_map.is_empty() {
                for (_, expr_ir) in acc_predicates.iter_mut() {
                    map_column_references(expr_ir, expr_arena, &alias_rename_map);
                }
            }

            let alp = lp_arena.take(input);
            let alp = self.push_down(alp, acc_predicates, lp_arena, expr_arena)?;
            lp_arena.replace(input, alp);

            Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
        } else {
            let mut local_predicates = Vec::with_capacity(acc_predicates.len());

            let inputs = lp.get_inputs();

            // determine new inputs by pushing down predicates
            let new_inputs = inputs
                .into_iter()
                .map(|node| {
                    // first we check if we are able to push down the predicate passed this node
                    // it could be that this node just added the column where we base the predicate on
                    let input_schema = lp_arena.get(node).schema(lp_arena);
                    let mut pushdown_predicates =
                        optimizer::init_hashmap(Some(acc_predicates.len()));
                    for (_, predicate) in acc_predicates.iter() {
                        // we can pushdown the predicate
                        if check_input_node(predicate.node(), &input_schema, expr_arena) {
                            insert_predicate_dedup(&mut pushdown_predicates, predicate, expr_arena)
                        }
                        // we cannot pushdown the predicate we do it here
                        else {
                            local_predicates.push(predicate.clone());
                        }
                    }

                    let alp = lp_arena.take(node);
                    let alp = self.push_down(alp, pushdown_predicates, lp_arena, expr_arena)?;
                    lp_arena.replace(node, alp);
                    Ok(node)
                })
                .collect::<PolarsResult<UnitVec<_>>>()?;

            let lp = lp.with_inputs(new_inputs);
            Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
        }
    }

    /// Filter will be done at this node, but we continue optimization
    fn no_pushdown_restart_opt(
        &mut self,
        lp: IR,
        acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let inputs = lp.inputs();

        let new_inputs = inputs
            .map(|node| {
                let alp = lp_arena.take(node);
                let alp = self.push_down(
                    alp,
                    init_hashmap(Some(acc_predicates.len())),
                    lp_arena,
                    expr_arena,
                )?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let lp = lp.with_inputs(new_inputs);

        // all predicates are done locally
        let local_predicates = acc_predicates.into_values().collect::<Vec<_>>();
        Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
    }

    fn no_pushdown(
        &mut self,
        lp: IR,
        acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        // all predicates are done locally
        let local_predicates = acc_predicates.into_values().collect::<Vec<_>>();
        Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
    }

    /// Predicate pushdown optimizer
    ///
    /// # Arguments
    ///
    /// * `IR` - Arena based logical plan tree representing the query.
    /// * `acc_predicates` - The predicates we accumulate during tree traversal.
    ///                      The hashmap maps from leaf-column name to predicates on that column.
    ///                      If the key is already taken we combine the predicate with a bitand operation.
    ///                      The `Node`s are indexes in the `expr_arena`
    /// * `lp_arena` - The local memory arena for the logical plan.
    /// * `expr_arena` - The local memory arena for the expressions.
    #[recursive]
    fn push_down(
        &mut self,
        lp: IR,
        mut acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        use IR::*;

        // Note: The logic within the match block should ensure `acc_predicates` is left in a state
        // where it contains only pushable exprs after it is done (although in some cases it may
        // contain a single fallible expression).

        match lp {
            Filter {
                // Note: We assume AND'ed predicates have already been split to separate IR filter
                // nodes during DSL conversion so we don't do that here.
                ref predicate,
                input,
            } => {
                // Use a tmp_key to avoid inadvertently combining predicates that otherwise would have
                // been partially pushed:
                //
                // (1) .filter(pl.count().over("key") == 1)
                // (2) .filter(pl.col("key") == 1)
                //
                // (2) can be pushed past (1) but they both have the same predicate
                // key name in the hashtable.
                let tmp_key = temporary_unique_key(&acc_predicates);
                acc_predicates.insert(tmp_key.clone(), predicate.clone());

                let maintain_errors = self.maintain_errors;

                let local_predicates = match pushdown_eligibility(
                    &[],
                    &[(&tmp_key, predicate.clone())],
                    &acc_predicates,
                    expr_arena,
                    self.empty_nodes_scratch_mut(),
                    maintain_errors,
                    lp_arena.get(input),
                )?
                .0
                {
                    PushdownEligibility::Full => vec![],
                    PushdownEligibility::Partial { to_local } => {
                        let mut out = Vec::with_capacity(to_local.len());
                        for key in to_local {
                            out.push(acc_predicates.remove(&key).unwrap());
                        }
                        out
                    },
                    PushdownEligibility::NoPushdown => {
                        let out = acc_predicates.drain().map(|t| t.1).collect();
                        acc_predicates.clear();
                        out
                    },
                };

                if let Some(predicate) = acc_predicates.remove(&tmp_key) {
                    insert_predicate_dedup(&mut acc_predicates, &predicate, expr_arena);
                }

                let alp = lp_arena.take(input);
                let new_input = self.push_down(alp, acc_predicates, lp_arena, expr_arena)?;

                // TODO!
                // If a predicates result would be influenced by earlier applied
                // predicates, we simply don't pushdown this one passed this node
                // However, we can do better and let it pass but store the order of the predicates
                // so that we can apply them in correct order at the deepest level
                Ok(
                    self.optional_apply_predicate(
                        new_input,
                        local_predicates,
                        lp_arena,
                        expr_arena,
                    ),
                )
            },
            DataFrameScan {
                df,
                schema,
                output_schema,
            } => {
                let selection = predicate_at_scan(acc_predicates, None, expr_arena);
                let mut lp = DataFrameScan {
                    df,
                    schema,
                    output_schema,
                };

                if let Some(predicate) = selection {
                    let input = lp_arena.add(lp);

                    lp = IR::Filter { input, predicate }
                }

                Ok(lp)
            },
            Scan {
                sources,
                file_info,
                hive_parts: scan_hive_parts,
                ref predicate,
                predicate_file_skip_applied,
                scan_type,
                unified_scan_args,
                output_schema,
            } => {
                let mut blocked_names = Vec::with_capacity(2);

                // TODO: Allow predicates on file names, this should be supported by new-streaming.
                if let Some(col) = unified_scan_args.include_file_paths.as_deref() {
                    blocked_names.push(col);
                }

                let local_predicates = if blocked_names.is_empty() {
                    vec![]
                } else {
                    transfer_to_local_by_name(expr_arena, &mut acc_predicates, |name| {
                        blocked_names.contains(&name.as_ref())
                    })
                };
                let predicate = predicate_at_scan(acc_predicates, predicate.clone(), expr_arena);

                let mut do_optimization = match &*scan_type {
                    #[cfg(feature = "csv")]
                    FileScanIR::Csv { .. } => unified_scan_args.pre_slice.is_none(),
                    FileScanIR::Anonymous { function, .. } => function.allows_predicate_pushdown(),
                    #[cfg(feature = "json")]
                    FileScanIR::NDJson { .. } => true,
                    #[allow(unreachable_patterns)]
                    _ => true,
                };
                do_optimization &= predicate.is_some();

                let hive_parts = scan_hive_parts;

                let lp = if do_optimization {
                    Scan {
                        sources,
                        file_info,
                        hive_parts,
                        predicate,
                        predicate_file_skip_applied,
                        unified_scan_args,
                        output_schema,
                        scan_type,
                    }
                } else {
                    let lp = Scan {
                        sources,
                        file_info,
                        hive_parts,
                        predicate: None,
                        predicate_file_skip_applied,
                        unified_scan_args,
                        output_schema,
                        scan_type,
                    };
                    if let Some(predicate) = predicate {
                        let input = lp_arena.add(lp);
                        Filter { input, predicate }
                    } else {
                        lp
                    }
                };

                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            },
            Distinct { input, options } => {
                let subset = if let Some(ref subset) = options.subset {
                    subset.as_ref()
                } else {
                    &[]
                };
                let mut names_set = PlHashSet::<PlSmallStr>::with_capacity(subset.len());
                for name in subset.iter() {
                    names_set.insert(name.clone());
                }

                let local_predicates = match options.keep_strategy {
                    UniqueKeepStrategy::Any => {
                        let condition = |e: &ExprIR| {
                            // if not elementwise -> to local
                            !is_elementwise_rec(e.node(), expr_arena)
                        };
                        transfer_to_local_by_expr_ir(expr_arena, &mut acc_predicates, condition)
                    },
                    UniqueKeepStrategy::First
                    | UniqueKeepStrategy::Last
                    | UniqueKeepStrategy::None => {
                        let condition = |name: &PlSmallStr| {
                            !subset.is_empty() && !names_set.contains(name.as_str())
                        };
                        transfer_to_local_by_name(expr_arena, &mut acc_predicates, condition)
                    },
                };

                self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;
                let lp = Distinct { input, options };
                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            },
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                schema,
                options,
            } => process_join(
                self,
                lp_arena,
                expr_arena,
                input_left,
                input_right,
                left_on,
                right_on,
                schema,
                options,
                acc_predicates,
                self.new_streaming,
            ),
            MapFunction { ref function, .. } => {
                if function.allow_predicate_pd() {
                    match function {
                        FunctionIR::Explode { columns, .. } => {
                            let condition = |name: &PlSmallStr| columns.iter().any(|s| s == name);

                            // first columns that refer to the exploded columns should be done here
                            let local_predicates = transfer_to_local_by_name(
                                expr_arena,
                                &mut acc_predicates,
                                condition,
                            );

                            let lp = self.pushdown_and_continue(
                                lp,
                                acc_predicates,
                                lp_arena,
                                expr_arena,
                                false,
                            )?;
                            Ok(self.optional_apply_predicate(
                                lp,
                                local_predicates,
                                lp_arena,
                                expr_arena,
                            ))
                        },
                        #[cfg(feature = "pivot")]
                        FunctionIR::Unpivot { args, .. } => {
                            // predicates that will be done at this level
                            let condition = |name: &PlSmallStr| {
                                name == &args.variable_name || name == &args.value_name
                            };
                            let local_predicates = transfer_to_local_by_name(
                                expr_arena,
                                &mut acc_predicates,
                                condition,
                            );

                            let lp = self.pushdown_and_continue(
                                lp,
                                acc_predicates,
                                lp_arena,
                                expr_arena,
                                false,
                            )?;
                            Ok(self.optional_apply_predicate(
                                lp,
                                local_predicates,
                                lp_arena,
                                expr_arena,
                            ))
                        },
                        FunctionIR::Unnest {
                            columns,
                            separator: _,
                        } => {
                            let exclude = columns.iter().cloned().collect::<PlHashSet<_>>();

                            let local_predicates =
                                transfer_to_local_by_name(expr_arena, &mut acc_predicates, |x| {
                                    exclude.contains(x)
                                });

                            let lp = self.pushdown_and_continue(
                                lp,
                                acc_predicates,
                                lp_arena,
                                expr_arena,
                                false,
                            )?;
                            Ok(self.optional_apply_predicate(
                                lp,
                                local_predicates,
                                lp_arena,
                                expr_arena,
                            ))
                        },
                        _ => self.pushdown_and_continue(
                            lp,
                            acc_predicates,
                            lp_arena,
                            expr_arena,
                            false,
                        ),
                    }
                } else {
                    self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena)
                }
            },
            GroupBy {
                input,
                keys,
                aggs,
                schema,
                apply,
                maintain_order,
                options,
            } => process_group_by(
                self,
                lp_arena,
                expr_arena,
                input,
                keys,
                aggs,
                schema,
                maintain_order,
                apply,
                options,
                acc_predicates,
            ),
            lp @ Union { .. } => {
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
            },
            Sort {
                input,
                by_column,
                mut slice,
                sort_options,
            } => {
                if let Some((offset, len, None)) = slice
                    && by_column.len() == 1
                {
                    let n = by_column[0].node();
                    if let AExpr::Column(_) = expr_arena.get(n) {
                        let (dyn_pred_node, pred) = new_dynamic_pred(n, expr_arena);
                        slice = Some((offset, len, Some(pred)));

                        let predicate = ExprIR::from_node(dyn_pred_node, expr_arena);
                        insert_predicate_dedup(&mut acc_predicates, &predicate, expr_arena);
                    }
                }

                let lp = Sort {
                    input,
                    by_column,
                    slice,
                    sort_options,
                };
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, true)
            },
            lp @ Sink { .. } | lp @ SinkMultiple { .. } => {
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
            },
            // Pushed down passed these nodes
            lp @ HStack { .. }
            | lp @ Select { .. }
            | lp @ SimpleProjection { .. }
            | lp @ ExtContext { .. } => {
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, true)
            },
            // NOT Pushed down passed these nodes
            // predicates influence slice sizes
            lp @ Slice { .. } => {
                self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena)
            },
            lp @ HConcat { .. } => {
                self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena)
            },
            // Caches will run predicate push-down in the `cache_states` run.
            Cache { .. } => {
                if self.caches_pass_allowance == 0 {
                    self.no_pushdown(lp, acc_predicates, lp_arena, expr_arena)
                } else {
                    self.caches_pass_allowance = self.caches_pass_allowance.saturating_sub(1);
                    self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
                }
            },
            #[cfg(feature = "python")]
            PythonScan { mut options } => {
                let predicate = predicate_at_scan(acc_predicates, None, expr_arena);
                if let Some(predicate) = predicate {
                    match ExprPushdownGroup::Pushable.update_with_expr_rec(
                        expr_arena.get(predicate.node()),
                        expr_arena,
                        None,
                    ) {
                        ExprPushdownGroup::Barrier => {
                            if cfg!(debug_assertions) {
                                // Expression should not be pushed here by the optimizer
                                panic!()
                            }

                            return Ok(self.optional_apply_predicate(
                                PythonScan { options },
                                vec![predicate],
                                lp_arena,
                                expr_arena,
                            ));
                        },

                        ExprPushdownGroup::Pushable | ExprPushdownGroup::Fallible => {
                            options.predicate = PythonPredicate::Polars(predicate);
                        },
                    }
                }

                Ok(PythonScan { options })
            },
            #[cfg(feature = "merge_sorted")]
            lp @ MergeSorted { .. } => {
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
            },
            Invalid => unreachable!(),
        }
    }

    pub(crate) fn optimize(
        &mut self,
        logical_plan: IR,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let acc_predicates = PlHashMap::new();
        self.push_down(logical_plan, acc_predicates, lp_arena, expr_arena)
    }
}
