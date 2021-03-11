use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{aexpr_to_root_names, aexpr_to_root_nodes, check_down_node, has_aexpr};
use ahash::RandomState;
use polars_core::prelude::*;
use std::collections::HashSet;

fn init_set() -> HashSet<Arc<String>, RandomState> {
    HashSet::with_capacity_and_hasher(128, RandomState::default())
}

/// utility function to get names of the columns needed in projection at scan level
fn get_scan_columns(
    acc_projections: &mut Vec<Node>,
    expr_arena: &Arena<AExpr>,
) -> Option<Vec<String>> {
    let mut with_columns = None;
    if !acc_projections.is_empty() {
        let mut columns = Vec::with_capacity(acc_projections.len());
        for expr in acc_projections {
            for name in aexpr_to_root_names(*expr, expr_arena) {
                columns.push((*name).clone())
            }
        }
        with_columns = Some(columns);
    }
    with_columns
}

/// split in a projection vec that can be pushed down and a projection vec that should be used
/// in this node
///
/// # Returns
/// accumulated_projections, local_projections, accumulated_names
fn split_acc_projections(
    acc_projections: Vec<Node>,
    down_schema: &Schema,
    expr_arena: &mut Arena<AExpr>,
) -> (Vec<Node>, Vec<Node>, HashSet<Arc<String>, RandomState>) {
    // If node above has as many columns as the projection there is nothing to pushdown.
    if down_schema.fields().len() == acc_projections.len() {
        let local_projections = acc_projections;
        (
            vec![],
            local_projections,
            HashSet::with_hasher(RandomState::default()),
        )
    } else {
        let (acc_projections, local_projections): (Vec<Node>, Vec<Node>) = acc_projections
            .into_iter()
            .partition(|expr| check_down_node(*expr, down_schema, expr_arena));
        let mut names = init_set();
        for proj in &acc_projections {
            for name in aexpr_to_root_names(*proj, expr_arena) {
                names.insert(name);
            }
        }
        (acc_projections, local_projections, names)
    }
}

/// utility function such that we can recurse all binary expressions in the expression tree
fn add_to_accumulated(
    expr: Node,
    acc_projections: &mut Vec<Node>,
    projected_names: &mut HashSet<Arc<String>, RandomState>,
    expr_arena: &mut Arena<AExpr>,
) {
    for root_node in aexpr_to_root_nodes(expr, expr_arena) {
        for name in aexpr_to_root_names(root_node, expr_arena) {
            if projected_names.insert(name) {
                acc_projections.push(root_node)
            }
        }
    }
}

pub struct ProjectionPushDown {}

impl ProjectionPushDown {
    fn finish_node(
        &self,
        local_projections: Vec<Node>,
        builder: ALogicalPlanBuilder,
    ) -> ALogicalPlan {
        if !local_projections.is_empty() {
            builder.project(local_projections).build()
        } else {
            builder.build()
        }
    }

    /// Helper method. This pushes down current node and assigns the result to this node.
    fn pushdown_and_assign(
        &self,
        input: Node,
        mut acc_projections: Vec<Node>,
        mut names: HashSet<Arc<String>, RandomState>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<()> {
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
        &self,
        logical_plan: ALogicalPlan,
        mut acc_projections: Vec<Node>,
        mut names: HashSet<Arc<String>, RandomState>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
            Slice { input, offset, len } => {
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Slice { input, offset, len })
            }

            Projection { expr, input, .. } => {
                // A projection can consist of a chain of expressions followed by an alias.
                // We want to do the chain locally because it can have complicated side effects.
                // The only thing we push down is the root name of the projection.
                // So we:
                //      - add the root of the projections to accumulation,
                //      - also do the complete projection locally to keep the schema (column order) and the alias.
                for e in &expr {
                    // in this branch we check a double projection case
                    // df
                    //   .select(col("foo").alias("bar"))
                    //   .select(col("bar")
                    //
                    // In this query, bar cannot pass this projection, as it would not exist in DF.
                    if !acc_projections.is_empty() {
                        if let AExpr::Alias(_, name) = expr_arena.get(*e) {
                            if names.remove(name) {
                                acc_projections = acc_projections
                                    .into_iter()
                                    .filter(|expr| {
                                        !aexpr_to_root_names(*expr, expr_arena).contains(name)
                                    })
                                    .collect();
                            }
                        }
                    }

                    add_to_accumulated(*e, &mut acc_projections, &mut names, expr_arena);
                }

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                let lp = lp_arena.get(input);

                let mut local_projection = Vec::with_capacity(expr.len());

                // the projections should all be done at the latest projection node to keep the same schema order
                if projections_seen == 0 {
                    for expr in expr {
                        // TODO! maybe we can remove this check?
                        // We check if we still can the projection here.
                        if expr_arena
                            .get(expr)
                            .to_field(lp.schema(lp_arena), Context::Other, expr_arena)
                            .is_ok()
                        {
                            local_projection.push(expr);
                        }
                    }
                    // only aliases should be projected locally
                } else {
                    for expr in expr {
                        if has_aexpr(
                            expr,
                            expr_arena,
                            &AExpr::Alias(Default::default(), Arc::new("".into())),
                            true,
                        ) {
                            local_projection.push(expr)
                        }
                    }
                }

                let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena);
                Ok(self.finish_node(local_projection, builder))
            }
            LocalProjection { expr, input, .. } => {
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                let lp = lp_arena.get(input);
                let schema = lp.schema(lp_arena);

                // projection from a wildcard may be dropped if the schema changes due to the optimization
                let proj = expr
                    .into_iter()
                    .filter(|e| check_down_node(*e, schema, expr_arena))
                    .collect();
                Ok(ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                    .project_local(proj)
                    .build())
            }
            DataFrameScan {
                df,
                schema,
                selection,
                ..
            } => {
                let mut projection = None;
                if !acc_projections.is_empty() {
                    projection = Some(acc_projections)
                }
                let lp = DataFrameScan {
                    df,
                    schema,
                    projection,
                    selection,
                };
                Ok(lp)
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                predicate,
                aggregate,
                stop_after_n_rows,
                cache,
                ..
            } => {
                let with_columns = get_scan_columns(&mut acc_projections, expr_arena);
                let lp = ParquetScan {
                    path,
                    schema,
                    with_columns,
                    predicate,
                    aggregate,
                    stop_after_n_rows,
                    cache,
                };
                Ok(lp)
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
                ignore_errors,
                skip_rows,
                stop_after_n_rows,
                predicate,
                aggregate,
                cache,
                ..
            } => {
                let with_columns = get_scan_columns(&mut acc_projections, expr_arena);
                let lp = CsvScan {
                    path,
                    schema,
                    has_header,
                    delimiter,
                    ignore_errors,
                    with_columns,
                    skip_rows,
                    stop_after_n_rows,
                    predicate,
                    aggregate,
                    cache,
                };
                Ok(lp)
            }
            Sort {
                input,
                by_column,
                reverse,
            } => {
                if !acc_projections.is_empty() {
                    // Make sure that the column used for the sort is projected
                    let node = expr_arena.add(AExpr::Column(Arc::new(by_column.clone())));
                    add_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
                }

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Sort {
                    input,
                    by_column,
                    reverse,
                })
            }
            Explode { input, columns } => {
                if !acc_projections.is_empty() {
                    // Make sure that the exploded columns are projected.
                    for column in &columns {
                        let node = expr_arena.add(AExpr::Column(Arc::new(column.clone())));
                        add_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
                    }
                }
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Explode { input, columns })
            }
            Cache { input } => {
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Cache { input })
            }
            Distinct {
                input,
                maintain_order,
                subset,
            } => {
                // make sure that the set of unique columns is projected
                if let Some(subset) = subset.as_ref() {
                    if !acc_projections.is_empty() {
                        for name in subset {
                            let node = expr_arena.add(AExpr::Column(Arc::new(name.clone())));
                            add_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
                        }
                    }
                };
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Distinct {
                    input,
                    maintain_order,
                    subset,
                })
            }
            Selection { predicate, input } => {
                if !acc_projections.is_empty() {
                    // make sure that the filter column is projected
                    add_to_accumulated(predicate, &mut acc_projections, &mut names, expr_arena);
                };
                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                Ok(Selection { predicate, input })
            }
            Melt {
                input,
                id_vars,
                value_vars,
                ..
            } => {
                let (mut acc_projections, mut local_projections, names) = split_acc_projections(
                    acc_projections,
                    lp_arena.get(input).schema(lp_arena),
                    expr_arena,
                );

                if !local_projections.is_empty() {
                    local_projections.extend_from_slice(&acc_projections);
                }

                // make sure that the requested columns are projected
                if !acc_projections.is_empty() {
                    for name in id_vars.iter() {
                        let node = expr_arena.add(AExpr::Column(Arc::new(name.clone())));
                        acc_projections.push(node);
                    }
                    for name in value_vars.iter() {
                        let node = expr_arena.add(AExpr::Column(Arc::new(name.clone())));
                        acc_projections.push(node);
                    }
                }

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;

                let mut builder =
                    ALogicalPlanBuilder::new(input, expr_arena, lp_arena).melt(id_vars, value_vars);
                Ok(self.finish_node(local_projections, builder))
            }
            Aggregate {
                input,
                keys,
                aggs,
                apply,
                schema,
            } => {
                // the custom function may need all columns so we do the projections here.
                if let Some(f) = apply {
                    let lp = Aggregate {
                        input,
                        keys,
                        aggs,
                        schema,
                        apply: Some(f),
                    };
                    let input = lp_arena.add(lp);

                    let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena);
                    Ok(self.finish_node(acc_projections, builder))
                } else {
                    // todo! remove unnecessary vec alloc.
                    let (mut acc_projections, _local_projections, mut names) =
                        split_acc_projections(
                            acc_projections,
                            lp_arena.get(input).schema(lp_arena),
                            expr_arena,
                        );

                    // add the columns used in the aggregations to the projection
                    for agg in &aggs {
                        add_to_accumulated(*agg, &mut acc_projections, &mut names, expr_arena);
                    }

                    // make sure the keys are projected
                    for key in &*keys {
                        add_to_accumulated(*key, &mut acc_projections, &mut names, expr_arena);
                    }

                    self.pushdown_and_assign(
                        input,
                        acc_projections,
                        names,
                        projections_seen,
                        lp_arena,
                        expr_arena,
                    )?;
                    let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                        .groupby(keys, aggs, apply);
                    Ok(builder.build())
                }
            }

            lp => Ok(lp),
        }
    }
}
