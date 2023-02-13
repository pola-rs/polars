mod generic;
mod groupby;
mod hstack;
mod joins;
mod melt;
mod projection;
mod rename;
#[cfg(feature = "semi_anti_join")]
mod semi_anti_join;

use polars_core::datatypes::PlHashSet;
use polars_core::prelude::*;
#[cfg(feature = "semi_anti_join")]
use semi_anti_join::process_semi_anti_join;

use crate::logical_plan::Context;
use crate::prelude::iterator::ArenaExprIter;
use crate::prelude::optimizer::projection_pushdown::generic::process_generic;
use crate::prelude::optimizer::projection_pushdown::groupby::process_groupby;
use crate::prelude::optimizer::projection_pushdown::hstack::process_hstack;
use crate::prelude::optimizer::projection_pushdown::joins::process_join;
use crate::prelude::optimizer::projection_pushdown::melt::process_melt;
use crate::prelude::optimizer::projection_pushdown::projection::process_projection;
use crate::prelude::optimizer::projection_pushdown::rename::process_rename;
use crate::prelude::*;
use crate::utils::{
    aexpr_assign_renamed_leaf, aexpr_to_leaf_names, aexpr_to_leaf_nodes, check_input_node,
    expr_is_projected_upstream,
};

fn init_vec() -> Vec<Node> {
    Vec::with_capacity(16)
}
fn init_set() -> PlHashSet<Arc<str>> {
    PlHashSet::with_capacity(32)
}

/// utility function to get names of the columns needed in projection at scan level
fn get_scan_columns(
    acc_projections: &mut Vec<Node>,
    expr_arena: &Arena<AExpr>,
) -> Option<Arc<Vec<String>>> {
    let mut with_columns = None;
    if !acc_projections.is_empty() {
        let mut columns = Vec::with_capacity(acc_projections.len());
        for expr in acc_projections {
            for name in aexpr_to_leaf_names(*expr, expr_arena) {
                columns.push((*name).to_owned())
            }
        }
        with_columns = Some(Arc::new(columns));
    }
    with_columns
}

/// split in a projection vec that can be pushed down and a projection vec that should be used
/// in this node
///
/// # Returns
/// accumulated_projections, local_projections, accumulated_names
///
/// - `expands_schema`. An unnest adds more columns to a schema, so we cannot use fast path
fn split_acc_projections(
    acc_projections: Vec<Node>,
    down_schema: &Schema,
    expr_arena: &mut Arena<AExpr>,
    expands_schema: bool,
) -> (Vec<Node>, Vec<Node>, PlHashSet<Arc<str>>) {
    // If node above has as many columns as the projection there is nothing to pushdown.
    if !expands_schema && down_schema.len() == acc_projections.len() {
        let local_projections = acc_projections;
        (vec![], local_projections, PlHashSet::new())
    } else {
        let (acc_projections, local_projections): (Vec<Node>, Vec<Node>) = acc_projections
            .into_iter()
            .partition(|expr| check_input_node(*expr, down_schema, expr_arena));
        let mut names = init_set();
        for proj in &acc_projections {
            for name in aexpr_to_leaf_names(*proj, expr_arena) {
                names.insert(name);
            }
        }
        (acc_projections, local_projections, names)
    }
}

/// utility function such that we can recurse all binary expressions in the expression tree
fn add_expr_to_accumulated(
    expr: Node,
    acc_projections: &mut Vec<Node>,
    projected_names: &mut PlHashSet<Arc<str>>,
    expr_arena: &mut Arena<AExpr>,
) {
    for root_node in aexpr_to_leaf_nodes(expr, expr_arena) {
        for name in aexpr_to_leaf_names(root_node, expr_arena) {
            if projected_names.insert(name) {
                acc_projections.push(root_node)
            }
        }
    }
}

fn add_str_to_accumulated(
    name: &str,
    acc_projections: &mut Vec<Node>,
    projected_names: &mut PlHashSet<Arc<str>>,
    expr_arena: &mut Arena<AExpr>,
) {
    // if empty: all columns are already projected.
    if !acc_projections.is_empty() {
        let node = expr_arena.add(AExpr::Column(Arc::from(name)));
        add_expr_to_accumulated(node, acc_projections, projected_names, expr_arena);
    }
}

fn update_scan_schema(
    acc_projections: &[Node],
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    // this is only needed for parsers that sort the projections
    // currently these are:
    // sorting parsers: csv,
    // non-sorting: parquet, ipc
    sort_projections: bool,
) -> PolarsResult<Schema> {
    let mut new_schema = Schema::with_capacity(acc_projections.len());
    let mut new_cols = Vec::with_capacity(acc_projections.len());
    for node in acc_projections.iter() {
        for name in aexpr_to_leaf_names(*node, expr_arena) {
            let item = schema.get_full(&name).ok_or_else(|| {
                PolarsError::ComputeError(
                    format!("column '{name}' not available in schema {schema:?}").into(),
                )
            })?;
            new_cols.push(item);
        }
    }
    // make sure that the projections are sorted by the schema.
    if sort_projections {
        new_cols.sort_unstable_by_key(|item| item.0);
    }
    for item in new_cols {
        new_schema.with_column(item.1.clone(), item.2.clone());
    }
    Ok(new_schema)
}

pub struct ProjectionPushDown {
    pub(crate) has_joins_or_unions: bool,
    pub(crate) has_cache: bool,
}

impl ProjectionPushDown {
    pub(super) fn new() -> Self {
        Self {
            has_joins_or_unions: false,
            has_cache: false,
        }
    }

    /// Projection will be done at this node, but we continue optimization
    fn no_pushdown_restart_opt(
        &mut self,
        lp: ALogicalPlan,
        acc_projections: Vec<Node>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<ALogicalPlan> {
        let inputs = lp.get_inputs();
        let exprs = lp.get_exprs();

        let new_inputs = inputs
            .iter()
            .map(|&node| {
                let alp = lp_arena.take(node);
                let alp = self.push_down(
                    alp,
                    Default::default(),
                    Default::default(),
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                lp_arena.replace(node, alp);
                Ok(node)
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let lp = lp.with_exprs_and_input(exprs, new_inputs);

        let builder = ALogicalPlanBuilder::from_lp(lp, expr_arena, lp_arena);
        Ok(self.finish_node(acc_projections, builder))
    }

    fn finish_node(
        &mut self,
        local_projections: Vec<Node>,
        builder: ALogicalPlanBuilder,
    ) -> ALogicalPlan {
        if !local_projections.is_empty() {
            builder.project(local_projections).build()
        } else {
            builder.build()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn join_push_down(
        &mut self,
        schema_left: &Schema,
        schema_right: &Schema,
        proj: Node,
        pushdown_left: &mut Vec<Node>,
        pushdown_right: &mut Vec<Node>,
        names_left: &mut PlHashSet<Arc<str>>,
        names_right: &mut PlHashSet<Arc<str>>,
        expr_arena: &mut Arena<AExpr>,
    ) -> bool {
        let mut pushed_at_least_one = false;
        let names = aexpr_to_leaf_names(proj, expr_arena);
        let root_projections = aexpr_to_leaf_nodes(proj, expr_arena);

        for (name, root_projection) in names.into_iter().zip(root_projections) {
            if check_input_node(root_projection, schema_left, expr_arena)
                && names_left.insert(name.clone())
            {
                pushdown_left.push(proj);
                pushed_at_least_one = true;
            }
            if check_input_node(root_projection, schema_right, expr_arena)
                && names_right.insert(name)
            {
                pushdown_right.push(proj);
                pushed_at_least_one = true;
            }
        }

        pushed_at_least_one
    }

    /// This pushes down current node and assigns the result to this node.
    fn pushdown_and_assign(
        &mut self,
        input: Node,
        acc_projections: Vec<Node>,
        names: PlHashSet<Arc<str>>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<()> {
        let alp = lp_arena.take(input);
        let lp = self.push_down(
            alp,
            acc_projections,
            names,
            projections_seen,
            lp_arena,
            expr_arena,
        )?;
        lp_arena.replace(input, lp);
        Ok(())
    }

    /// This pushes down the projection that are validated
    /// that they can be done successful at the schema above
    /// The result is assigned to this node.
    ///
    /// The local projections are return and still have to be applied
    fn pushdown_and_assign_check_schema(
        &mut self,
        input: Node,
        acc_projections: Vec<Node>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        // an unnest changes/expands the schema
        expands_schema: bool,
    ) -> PolarsResult<Vec<Node>> {
        let alp = lp_arena.take(input);
        let down_schema = alp.schema(lp_arena);

        let (acc_projections, local_projections, names) =
            split_acc_projections(acc_projections, &down_schema, expr_arena, expands_schema);

        let lp = self.push_down(
            alp,
            acc_projections,
            names,
            projections_seen,
            lp_arena,
            expr_arena,
        )?;
        lp_arena.replace(input, lp);
        Ok(local_projections)
    }

    /// Projection pushdown optimizer
    ///
    /// # Arguments
    ///
    /// * `AlogicalPlan` - Arena based logical plan tree representing the query.
    /// * `acc_projections` - The projections we accumulate during tree traversal.
    /// * `names` - We keep track of the names to ensure we don't do duplicate projections.
    /// * `projections_seen` - Count the number of projection operations during tree traversal.
    /// * `lp_arena` - The local memory arena for the logical plan.
    /// * `expr_arena` - The local memory arena for the expressions.
    ///
    fn push_down(
        &mut self,
        logical_plan: ALogicalPlan,
        mut acc_projections: Vec<Node>,
        mut projected_names: PlHashSet<Arc<str>>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
            Projection { expr, input, .. } => process_projection(
                self,
                input,
                expr,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            ),
            LocalProjection { expr, input, .. } => {
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                let lp = lp_arena.get(input);
                let schema = lp.schema(lp_arena);

                // projection from a wildcard may be dropped if the schema changes due to the optimization
                let proj = expr
                    .into_iter()
                    .filter(|e| check_input_node(*e, &schema, expr_arena))
                    .collect();
                Ok(ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                    .project_local(proj)
                    .build())
            }
            AnonymousScan {
                function,
                file_info,
                predicate,
                mut options,
                output_schema,
            } => {
                if function.allows_projection_pushdown() {
                    options.with_columns = get_scan_columns(&mut acc_projections, expr_arena);

                    let output_schema = if options.with_columns.is_none() {
                        None
                    } else {
                        Some(Arc::new(update_scan_schema(
                            &acc_projections,
                            expr_arena,
                            &file_info.schema,
                            true,
                        )?))
                    };
                    options.output_schema = output_schema.clone();

                    let lp = AnonymousScan {
                        function,
                        file_info,
                        output_schema,
                        options,
                        predicate,
                    };
                    Ok(lp)
                } else {
                    let lp = AnonymousScan {
                        function,
                        file_info,
                        predicate,
                        options,
                        output_schema,
                    };
                    Ok(lp)
                }
            }
            DataFrameScan {
                df,
                schema,
                mut output_schema,
                selection,
                ..
            } => {
                let mut projection = None;
                if !acc_projections.is_empty() {
                    output_schema = Some(Arc::new(update_scan_schema(
                        &acc_projections,
                        expr_arena,
                        &schema,
                        false,
                    )?));
                    projection = get_scan_columns(&mut acc_projections, expr_arena);
                }
                let lp = DataFrameScan {
                    df,
                    schema,
                    output_schema,
                    projection,
                    selection,
                };
                Ok(lp)
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                file_info,
                predicate,
                mut options,
                ..
            } => {
                let with_columns = get_scan_columns(&mut acc_projections, expr_arena);
                let output_schema = if with_columns.is_none() {
                    None
                } else {
                    Some(Arc::new(update_scan_schema(
                        &acc_projections,
                        expr_arena,
                        &file_info.schema,
                        false,
                    )?))
                };
                options.with_columns = with_columns;

                let lp = IpcScan {
                    path,
                    file_info,
                    output_schema,
                    predicate,
                    options,
                };
                Ok(lp)
            }

            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                file_info,
                predicate,
                mut options,
                cloud_options,
                ..
            } => {
                let with_columns = get_scan_columns(&mut acc_projections, expr_arena);
                let output_schema = if with_columns.is_none() {
                    None
                } else {
                    Some(Arc::new(update_scan_schema(
                        &acc_projections,
                        expr_arena,
                        &file_info.schema,
                        false,
                    )?))
                };
                options.with_columns = with_columns;

                let lp = ParquetScan {
                    path,
                    file_info,
                    output_schema,
                    predicate,
                    options,
                    cloud_options,
                };
                Ok(lp)
            }
            #[cfg(feature = "python")]
            PythonScan {
                mut options,
                predicate,
            } => {
                options.with_columns = get_scan_columns(&mut acc_projections, expr_arena);

                options.output_schema = if options.with_columns.is_none() {
                    None
                } else {
                    Some(Arc::new(update_scan_schema(
                        &acc_projections,
                        expr_arena,
                        &options.schema,
                        true,
                    )?))
                };
                Ok(PythonScan { options, predicate })
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                file_info,
                mut options,
                predicate,
                ..
            } => {
                options.with_columns = get_scan_columns(&mut acc_projections, expr_arena);

                let output_schema = if options.with_columns.is_none() {
                    None
                } else {
                    Some(Arc::new(update_scan_schema(
                        &acc_projections,
                        expr_arena,
                        &file_info.schema,
                        true,
                    )?))
                };

                let lp = CsvScan {
                    path,
                    file_info,
                    output_schema,
                    options,
                    predicate,
                };
                Ok(lp)
            }
            Sort {
                input,
                by_column,
                args,
            } => {
                if !acc_projections.is_empty() {
                    // Make sure that the column(s) used for the sort is projected
                    by_column.iter().for_each(|node| {
                        aexpr_to_leaf_nodes(*node, expr_arena)
                            .iter()
                            .for_each(|root| {
                                add_expr_to_accumulated(
                                    *root,
                                    &mut acc_projections,
                                    &mut projected_names,
                                    expr_arena,
                                );
                            })
                    });
                }

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Sort {
                    input,
                    by_column,
                    args,
                })
            }
            Explode {
                input,
                columns,
                schema,
            } => {
                columns.iter().for_each(|name| {
                    add_str_to_accumulated(
                        name,
                        &mut acc_projections,
                        &mut projected_names,
                        expr_arena,
                    )
                });
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Explode {
                    input,
                    columns,
                    schema,
                })
            }
            Distinct { input, options } => {
                // make sure that the set of unique columns is projected
                if let Some(subset) = options.subset.as_ref() {
                    subset.iter().for_each(|name| {
                        add_str_to_accumulated(
                            name,
                            &mut acc_projections,
                            &mut projected_names,
                            expr_arena,
                        )
                    })
                }

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Distinct { input, options })
            }
            Selection { predicate, input } => {
                if !acc_projections.is_empty() {
                    // make sure that the filter column is projected
                    add_expr_to_accumulated(
                        predicate,
                        &mut acc_projections,
                        &mut projected_names,
                        expr_arena,
                    );
                };
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Selection { predicate, input })
            }
            Melt {
                input,
                args,
                schema,
            } => process_melt(
                self,
                input,
                args,
                schema,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            ),
            Aggregate {
                input,
                keys,
                aggs,
                apply,
                schema,
                maintain_order,
                options,
            } => process_groupby(
                self,
                input,
                keys,
                aggs,
                apply,
                schema,
                maintain_order,
                options,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            ),
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => match options.how {
                #[cfg(feature = "semi_anti_join")]
                JoinType::Semi | JoinType::Anti => process_semi_anti_join(
                    self,
                    input_left,
                    input_right,
                    left_on,
                    right_on,
                    options,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                ),
                _ => process_join(
                    self,
                    input_left,
                    input_right,
                    left_on,
                    right_on,
                    options,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                ),
            },
            HStack { input, exprs, .. } => process_hstack(
                self,
                input,
                exprs,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            ),
            ExtContext {
                input, contexts, ..
            } => {
                // local projections are ignored. These are just root nodes
                // complex expression will still be done later
                let _local_projections = self.pushdown_and_assign_check_schema(
                    input,
                    acc_projections,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                    false,
                )?;

                let mut new_schema = lp_arena
                    .get(input)
                    .schema(lp_arena)
                    .as_ref()
                    .as_ref()
                    .clone();

                for node in &contexts {
                    let other_schema = lp_arena.get(*node).schema(lp_arena);
                    for fld in other_schema.iter_fields() {
                        if new_schema.get(fld.name()).is_none() {
                            new_schema.with_column(fld.name, fld.dtype);
                        }
                    }
                }

                Ok(ExtContext {
                    input,
                    contexts,
                    schema: Arc::new(new_schema),
                })
            }
            MapFunction {
                input,
                ref function,
            } => {
                let lp = MapFunction {
                    input,
                    function: function.clone(),
                };

                if let FunctionNode::Rename {
                    existing,
                    new,
                    swapping,
                } = function
                {
                    process_rename(
                        &mut acc_projections,
                        &mut projected_names,
                        expr_arena,
                        existing,
                        new,
                        *swapping,
                    )?;
                    self.pushdown_and_assign(
                        input,
                        acc_projections,
                        projected_names,
                        projections_seen,
                        lp_arena,
                        expr_arena,
                    )?;
                    return Ok(lp);
                }

                let MapFunction {ref function, ..} = lp else { unreachable!() };

                if function.allow_projection_pd() && !acc_projections.is_empty() {
                    let original_acc_projection_len = acc_projections.len();

                    // add columns needed for the function.
                    for name in function.additional_projection_pd_columns() {
                        let node = expr_arena.add(AExpr::Column(name.clone()));
                        add_expr_to_accumulated(
                            node,
                            &mut acc_projections,
                            &mut projected_names,
                            expr_arena,
                        )
                    }
                    let expands_schema = matches!(function, FunctionNode::Unnest { .. });

                    let local_projections = self.pushdown_and_assign_check_schema(
                        input,
                        acc_projections,
                        projections_seen,
                        lp_arena,
                        expr_arena,
                        expands_schema,
                    )?;
                    if local_projections.is_empty() {
                        Ok(lp)
                    } else {
                        // if we would project, we would remove pushed down predicates
                        if local_projections.len() < original_acc_projection_len {
                            Ok(ALogicalPlanBuilder::from_lp(lp, expr_arena, lp_arena)
                                .with_columns(local_projections)
                                .build())
                            // all projections are local
                        } else {
                            Ok(ALogicalPlanBuilder::from_lp(lp, expr_arena, lp_arena)
                                .project(local_projections)
                                .build())
                        }
                    }
                } else {
                    // restart projection pushdown
                    self.no_pushdown_restart_opt(
                        lp,
                        acc_projections,
                        projections_seen,
                        lp_arena,
                        expr_arena,
                    )
                }
            }
            lp @ Union { .. } => {
                self.has_joins_or_unions = true;
                process_generic(
                    self,
                    lp,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )
            }
            // These nodes only have inputs and exprs, so we can use same logic.
            lp @ Slice { .. } | lp @ FileSink { .. } => process_generic(
                self,
                lp,
                acc_projections,
                projected_names,
                projections_seen,
                lp_arena,
                expr_arena,
            ),
            Cache { .. } => {
                self.has_cache = true;
                // projections above this cache will be accumulated and pushed down
                // later
                // the redundant projection will be cleaned in the fast projection optimization
                // phase.
                if acc_projections.is_empty() {
                    Ok(logical_plan)
                } else {
                    Ok(
                        ALogicalPlanBuilder::from_lp(logical_plan, expr_arena, lp_arena)
                            .project(acc_projections)
                            .build(),
                    )
                }
            }
        }
    }

    pub fn optimize(
        &mut self,
        logical_plan: ALogicalPlan,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<ALogicalPlan> {
        let acc_projections = init_vec();
        let names = init_set();
        self.push_down(
            logical_plan,
            acc_projections,
            names,
            0,
            lp_arena,
            expr_arena,
        )
    }
}
