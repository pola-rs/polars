mod group_by;
mod join;
mod keys;
mod rename;
mod utils;

use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;
use recursive::recursive;
use utils::*;

use super::*;
use crate::dsl::function_expr::FunctionExpr;
use crate::prelude::optimizer::predicate_pushdown::group_by::process_group_by;
use crate::prelude::optimizer::predicate_pushdown::join::process_join;
use crate::prelude::optimizer::predicate_pushdown::rename::process_rename;
use crate::utils::{check_input_node, has_aexpr};

pub type ExprEval<'a> =
    Option<&'a dyn Fn(&ExprIR, &Arena<AExpr>) -> Option<Arc<dyn PhysicalIoExpr>>>;

pub struct PredicatePushDown<'a> {
    expr_eval: ExprEval<'a>,
    verbose: bool,
    block_at_cache: bool,
}

impl<'a> PredicatePushDown<'a> {
    pub fn new(expr_eval: ExprEval<'a>) -> Self {
        Self {
            expr_eval,
            verbose: verbose(),
            block_at_cache: true,
        }
    }

    pub(crate) fn block_at_cache(mut self, toggle: bool) -> Self {
        self.block_at_cache = toggle;
        self
    }

    fn optional_apply_predicate(
        &self,
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
        &self,
        input: Node,
        acc_predicates: PlHashMap<Arc<str>, ExprIR>,
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
        &self,
        lp: IR,
        mut acc_predicates: PlHashMap<Arc<str>, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        has_projections: bool,
    ) -> PolarsResult<IR> {
        let inputs = lp.get_inputs_vec();
        let exprs = lp.get_exprs();

        if has_projections {
            // projections should only have a single input.
            if inputs.len() > 1 {
                // except for ExtContext
                assert!(matches!(lp, IR::ExtContext { .. }));
            }
            let input = inputs[inputs.len() - 1];

            let (eligibility, alias_rename_map) =
                pushdown_eligibility(&exprs, &[], &acc_predicates, expr_arena)?;

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
                    return self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena)
                },
            };

            if !alias_rename_map.is_empty() {
                for (_, e) in acc_predicates.iter_mut() {
                    let mut needs_rename = false;

                    for (_, ae) in (&*expr_arena).iter(e.node()) {
                        if let AExpr::Column(name) = ae {
                            needs_rename |= alias_rename_map.contains_key(name);

                            if needs_rename {
                                break;
                            }
                        }
                    }

                    if needs_rename {
                        // TODO! Do this directly on AExpr.
                        let mut new_expr = node_to_expr(e.node(), expr_arena);
                        new_expr = new_expr.map_expr(|e| match e {
                            Expr::Column(name) => {
                                if let Some(rename_to) = alias_rename_map.get(&*name) {
                                    Expr::Column(rename_to.clone())
                                } else {
                                    Expr::Column(name)
                                }
                            },
                            e => e,
                        });
                        let predicate = to_aexpr(new_expr, expr_arena)?;
                        e.set_node(predicate);
                    }
                }
            }

            let alp = lp_arena.take(input);
            let alp = self.push_down(alp, acc_predicates, lp_arena, expr_arena)?;
            lp_arena.replace(input, alp);

            let lp = lp.with_exprs_and_input(exprs, inputs);
            Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
        } else {
            let mut local_predicates = Vec::with_capacity(acc_predicates.len());

            // determine new inputs by pushing down predicates
            let new_inputs = inputs
                .iter()
                .map(|&node| {
                    // first we check if we are able to push down the predicate passed this node
                    // it could be that this node just added the column where we base the predicate on
                    let input_schema = lp_arena.get(node).schema(lp_arena);
                    let mut pushdown_predicates =
                        optimizer::init_hashmap(Some(acc_predicates.len()));
                    for (_, predicate) in acc_predicates.iter() {
                        // we can pushdown the predicate
                        if check_input_node(predicate.node(), &input_schema, expr_arena) {
                            insert_and_combine_predicate(
                                &mut pushdown_predicates,
                                predicate,
                                expr_arena,
                            )
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
                .collect::<PolarsResult<Vec<_>>>()?;

            let lp = lp.with_exprs_and_input(exprs, new_inputs);
            Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
        }
    }

    /// Filter will be done at this node, but we continue optimization
    fn no_pushdown_restart_opt(
        &self,
        lp: IR,
        acc_predicates: PlHashMap<Arc<str>, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let inputs = lp.get_inputs();
        let exprs = lp.get_exprs();

        let new_inputs = inputs
            .iter()
            .map(|&node| {
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
        let lp = lp.with_exprs_and_input(exprs, new_inputs);

        // all predicates are done locally
        let local_predicates = acc_predicates.into_values().collect::<Vec<_>>();
        Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
    }

    fn no_pushdown(
        &self,
        lp: IR,
        acc_predicates: PlHashMap<Arc<str>, ExprIR>,
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
        &self,
        lp: IR,
        mut acc_predicates: PlHashMap<Arc<str>, ExprIR>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        use IR::*;

        match lp {
            Filter {
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
                let tmp_key = Arc::<str>::from(&*temporary_unique_key(&acc_predicates));
                acc_predicates.insert(tmp_key.clone(), predicate.clone());

                let local_predicates = match pushdown_eligibility(
                    &[],
                    &[(tmp_key.clone(), predicate.clone())],
                    &acc_predicates,
                    expr_arena,
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
                    insert_and_combine_predicate(&mut acc_predicates, &predicate, expr_arena);
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
                filter: selection,
            } => {
                let selection = predicate_at_scan(acc_predicates, selection, expr_arena);
                let lp = DataFrameScan {
                    df,
                    schema,
                    output_schema,
                    filter: selection,
                };
                Ok(lp)
            },
            Scan {
                mut paths,
                file_info,
                hive_parts: mut scan_hive_parts,
                ref predicate,
                mut scan_type,
                file_options: options,
                output_schema,
            } => {
                let mut blocked_names = Vec::with_capacity(2);

                if let Some(col) = options.include_file_paths.as_deref() {
                    blocked_names.push(col);
                }

                match &scan_type {
                    #[cfg(feature = "parquet")]
                    FileScan::Parquet { .. } => {},
                    #[cfg(feature = "ipc")]
                    FileScan::Ipc { .. } => {},
                    _ => {
                        // Disallow row index pushdown of other scans as they may
                        // not update the row index properly before applying the
                        // predicate (e.g. FileScan::Csv doesn't).
                        if let Some(ref row_index) = options.row_index {
                            blocked_names.push(row_index.name.as_ref());
                        };
                    },
                };

                let local_predicates = if blocked_names.is_empty() {
                    vec![]
                } else {
                    transfer_to_local_by_name(expr_arena, &mut acc_predicates, |name| {
                        blocked_names.contains(&name.as_ref())
                    })
                };
                let predicate = predicate_at_scan(acc_predicates, predicate.clone(), expr_arena);

                if let (Some(hive_parts), Some(predicate)) = (&scan_hive_parts, &predicate) {
                    if let Some(io_expr) = self.expr_eval.unwrap()(predicate, expr_arena) {
                        if let Some(stats_evaluator) = io_expr.as_stats_evaluator() {
                            let mut new_paths = Vec::with_capacity(paths.len());
                            let mut new_hive_parts = Vec::with_capacity(paths.len());

                            for i in 0..paths.len() {
                                let path = &paths[i];
                                let hive_parts = &hive_parts[i];

                                if stats_evaluator.should_read(hive_parts.get_statistics())? {
                                    new_paths.push(path.clone());
                                    new_hive_parts.push(hive_parts.clone());
                                }
                            }

                            if paths.len() != new_paths.len() {
                                if self.verbose {
                                    eprintln!(
                                        "hive partitioning: skipped {} files, first file : {}",
                                        paths.len() - new_paths.len(),
                                        paths[0].display()
                                    )
                                }
                                scan_type.remove_metadata();
                            }
                            if new_paths.is_empty() {
                                let schema = output_schema.as_ref().unwrap_or(&file_info.schema);
                                let df = DataFrame::empty_with_schema(schema);

                                return Ok(DataFrameScan {
                                    df: Arc::new(df),
                                    schema: schema.clone(),
                                    output_schema: None,
                                    filter: None,
                                });
                            } else {
                                paths = Arc::from(new_paths);
                                scan_hive_parts = Some(Arc::from(new_hive_parts));
                            }
                        }
                    }
                }

                let mut do_optimization = match &scan_type {
                    #[cfg(feature = "csv")]
                    FileScan::Csv { .. } => options.slice.is_none(),
                    FileScan::Anonymous { function, .. } => function.allows_predicate_pushdown(),
                    #[cfg(feature = "json")]
                    FileScan::NDJson { .. } => true,
                    #[allow(unreachable_patterns)]
                    _ => true,
                };
                do_optimization &= predicate.is_some();

                let hive_parts = scan_hive_parts;

                let lp = if do_optimization {
                    Scan {
                        paths,
                        file_info,
                        hive_parts,
                        predicate,
                        file_options: options,
                        output_schema,
                        scan_type,
                    }
                } else {
                    let lp = Scan {
                        paths,
                        file_info,
                        hive_parts,
                        predicate: None,
                        file_options: options,
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
                if let Some(ref subset) = options.subset {
                    // Predicates on the subset can pass.
                    let subset = subset.clone();
                    let mut names_set = PlHashSet::<&str>::with_capacity(subset.len());
                    for name in subset.iter() {
                        names_set.insert(name.as_ref());
                    }

                    let condition = |name: Arc<str>| !names_set.contains(name.as_ref());
                    let local_predicates =
                        transfer_to_local_by_name(expr_arena, &mut acc_predicates, condition);

                    self.pushdown_and_assign(input, acc_predicates, lp_arena, expr_arena)?;
                    let lp = Distinct { input, options };
                    Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
                } else {
                    let lp = Distinct { input, options };
                    self.no_pushdown_restart_opt(lp, acc_predicates, lp_arena, expr_arena)
                }
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
            ),
            MapFunction { ref function, .. } => {
                if function.allow_predicate_pd() {
                    match function {
                        FunctionIR::Rename { existing, new, .. } => {
                            let local_predicates =
                                process_rename(&mut acc_predicates, expr_arena, existing, new)?;
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
                        FunctionIR::Explode { columns, .. } => {
                            let condition =
                                |name: Arc<str>| columns.iter().any(|s| s.as_ref() == &*name);

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
                            let variable_name = args.variable_name.as_deref().unwrap_or("variable");
                            let value_name = args.value_name.as_deref().unwrap_or("value");

                            // predicates that will be done at this level
                            let condition = |name: Arc<str>| {
                                let name = &*name;
                                name == variable_name
                                    || name == value_name
                                    || args.on.iter().any(|s| s.as_str() == name)
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
                let mut local_predicates = vec![];

                // a count is influenced by a Union/Vstack
                acc_predicates.retain(|_, predicate| {
                    if has_aexpr(predicate.node(), expr_arena, |ae| matches!(ae, AExpr::Len)) {
                        local_predicates.push(predicate.clone());
                        false
                    } else {
                        true
                    }
                });
                let lp =
                    self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)?;
                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            },
            lp @ Sort { .. } => {
                let mut local_predicates = vec![];
                acc_predicates.retain(|_, predicate| {
                    if predicate_is_sort_boundary(predicate.node(), expr_arena) {
                        local_predicates.push(predicate.clone());
                        false
                    } else {
                        true
                    }
                });
                let lp =
                    self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)?;
                Ok(self.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
            },
            // Pushed down passed these nodes
            lp @ Sink { .. } => {
                self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
            },
            lp @ HStack { .. }
            | lp @ Select { .. }
            | lp @ Reduce { .. }
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
                if self.block_at_cache {
                    self.no_pushdown(lp, acc_predicates, lp_arena, expr_arena)
                } else {
                    self.pushdown_and_continue(lp, acc_predicates, lp_arena, expr_arena, false)
                }
            },
            #[cfg(feature = "python")]
            PythonScan { mut options } => {
                let predicate = predicate_at_scan(acc_predicates, None, expr_arena);
                if let Some(predicate) = predicate {
                    // For IO plugins we only accept streamable expressions as
                    // we want to apply the predicates to the batches.
                    if !is_streamable(predicate.node(), expr_arena, Context::Default)
                        && matches!(options.python_source, PythonScanSource::IOPlugin)
                    {
                        let lp = PythonScan { options };
                        return Ok(self.optional_apply_predicate(
                            lp,
                            vec![predicate],
                            lp_arena,
                            expr_arena,
                        ));
                    }

                    options.predicate = PythonPredicate::Polars(predicate);
                }
                Ok(PythonScan { options })
            },
            Invalid => unreachable!(),
        }
    }

    pub(crate) fn optimize(
        &self,
        logical_plan: IR,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<IR> {
        let acc_predicates = PlHashMap::new();
        self.push_down(logical_plan, acc_predicates, lp_arena, expr_arena)
    }
}
