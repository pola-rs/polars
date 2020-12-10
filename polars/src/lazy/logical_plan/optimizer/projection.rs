use crate::lazy::logical_plan::optimizer::check_down_node;
use crate::lazy::logical_plan::Context;
use crate::lazy::prelude::*;
use crate::lazy::utils::{
    expr_to_root_column_expr, expr_to_root_column_exprs, expr_to_root_column_name, has_expr,
};
use crate::prelude::*;
use ahash::RandomState;
use arrow::datatypes::Schema;
use std::collections::HashSet;
use std::sync::Arc;

fn init_vec() -> Vec<Expr> {
    Vec::with_capacity(100)
}
fn init_set() -> HashSet<Arc<String>, RandomState> {
    HashSet::with_capacity_and_hasher(128, RandomState::default())
}

// utility function such that we can recurse all binary expressions in the expression tree
fn add_to_accumulated(
    expression: &Expr,
    acc_projections: &mut Vec<Expr>,
    names: &mut HashSet<Arc<String>, RandomState>,
) -> Result<()> {
    for e in expr_to_root_column_exprs(expression) {
        let name = expr_to_root_column_name(&e)?;
        if names.insert(name) {
            acc_projections.push(e)
        }
    }
    Ok(())
}

fn get_scan_columns(acc_projections: &mut Vec<Expr>) -> Option<Vec<String>> {
    let mut with_columns = None;
    if !acc_projections.is_empty() {
        let mut columns = Vec::with_capacity(acc_projections.len());
        for expr in acc_projections {
            if let Ok(name) = expr_to_root_column_name(expr) {
                columns.push((*name).clone())
            }
        }
        with_columns = Some(columns);
    }
    with_columns
}

pub struct ProjectionPushDown {}

impl ProjectionPushDown {
    /// split in a projection vec that can be pushed down and a projection vec that should be used
    /// in this node
    ///
    /// # Returns
    /// accumulated_projections, local_projections, accumulated_names
    fn split_acc_projections(
        &self,
        acc_projections: Vec<Expr>,
        down_schema: &Schema,
    ) -> (Vec<Expr>, Vec<Expr>, HashSet<Arc<String>, RandomState>) {
        // If node above has as many columns as the projection there is nothing to pushdown.
        if down_schema.fields().len() == acc_projections.len() {
            let local_projections = acc_projections;

            (
                vec![],
                local_projections,
                HashSet::with_hasher(RandomState::default()),
            )
        } else {
            let (acc_projections, local_projections) = acc_projections
                .into_iter()
                .partition(|expr| check_down_node(expr, down_schema));
            let mut names = init_set();
            for proj in &acc_projections {
                let name = expr_to_root_column_name(proj).unwrap();
                names.insert(name);
            }
            (acc_projections, local_projections, names)
        }
    }
    fn finish_node(
        &self,
        local_projections: Vec<Expr>,
        builder: LogicalPlanBuilder,
    ) -> LogicalPlan {
        if !local_projections.is_empty() {
            builder.project(local_projections).build()
        } else {
            builder.build()
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn join_push_down(
        &self,
        schema_left: &Schema,
        schema_right: &Schema,
        proj: &Expr,
        pushdown_left: &mut Vec<Expr>,
        pushdown_right: &mut Vec<Expr>,
        names_left: &mut HashSet<Arc<String>, RandomState>,
        names_right: &mut HashSet<Arc<String>, RandomState>,
    ) -> Result<bool> {
        let mut pushed_at_least_one = false;
        let name = expr_to_root_column_name(&proj)?;
        let root_projection = expr_to_root_column_expr(proj)?;

        if check_down_node(&root_projection, schema_left) && names_left.insert(name.clone()) {
            pushdown_left.push(proj.clone());
            pushed_at_least_one = true;
        }
        if check_down_node(&root_projection, schema_right) && names_right.insert(name) {
            pushdown_right.push(proj.clone());
            pushed_at_least_one = true;
        }
        Ok(pushed_at_least_one)
    }

    // We recurrently traverse the logical plan and every projection we encounter we add to the accumulated
    // projections.
    // Every non projection operation we recurse and rebuild that operation on the output of the recursion.
    // The recursion stops at the nodes of the logical plan. These nodes IO or existing DataFrames. On top of
    // these nodes we apply the projection.
    fn push_down(
        &self,
        logical_plan: LogicalPlan,
        mut acc_projections: Vec<Expr>,
        mut names: HashSet<Arc<String>, RandomState>,
        projections_seen: usize,
    ) -> Result<LogicalPlan> {
        use LogicalPlan::*;
        match logical_plan {
            Slice { input, offset, len } => {
                let input =
                    Box::new(self.push_down(*input, acc_projections, names, projections_seen)?);
                Ok(Slice { input, offset, len })
            }
            Projection { expr, input, .. } => {
                dbg!(&expr, &acc_projections, projections_seen);
                // add the root of the projections to accumulation,
                // but also do them locally to keep the schema and the alias.
                for e in &expr {
                    // in this branch we check a double projection case
                    // df
                    //   .select(col("foo").alias("bar"))
                    //   .select(col("bar")
                    //
                    // In this query, bar cannot pass this projection, as it would not exist in DF.
                    if !acc_projections.is_empty() {
                        if let Expr::Alias(_, name) = e {
                            if names.remove(name) {
                                acc_projections = acc_projections
                                    .into_iter()
                                    .filter(|expr| &expr_to_root_column_name(expr).unwrap() != name)
                                    .collect();
                            }
                        }
                    }

                    add_to_accumulated(e, &mut acc_projections, &mut names)?;
                }
                let lp = self.push_down(*input, acc_projections, names, projections_seen + 1)?;

                let mut local_projection = Vec::with_capacity(expr.len());

                // the projections should all be done locally to keep the same schema order
                if projections_seen == 0 {
                    for expr in expr {
                        // why do we check this?
                        if expr.to_field(lp.schema(), Context::Other).is_ok() {
                            local_projection.push(expr);
                        }
                    }
                // only aliases should be projected locally
                } else {
                    let dummy = Expr::Alias(Box::new(Expr::Wildcard), Arc::new("".to_string()));
                    for expr in expr {
                        if has_expr(&expr, &dummy) {
                            local_projection.push(expr)
                        }
                    }
                }

                let builder = LogicalPlanBuilder::from(lp);
                Ok(self.finish_node(local_projection, builder))
            }
            LocalProjection { expr, input, .. } => {
                let lp = self.push_down(*input, acc_projections, names, projections_seen)?;
                let schema = lp.schema();
                // projection from a wildcard may be dropped if the schema changes due to the optimization
                let proj = expr
                    .into_iter()
                    .filter(|e| check_down_node(e, schema))
                    .collect();
                Ok(LogicalPlanBuilder::from(lp).project_local(proj).build())
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
                stop_after_n_rows,
                cache,
                ..
            } => {
                let with_columns = get_scan_columns(&mut acc_projections);
                let lp = ParquetScan {
                    path,
                    schema,
                    with_columns,
                    predicate,
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
                cache,
                ..
            } => {
                let with_columns = get_scan_columns(&mut acc_projections);
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
                    cache,
                };
                Ok(lp)
            }
            Sort {
                input,
                by_column,
                reverse,
            } => {
                let input =
                    Box::new(self.push_down(*input, acc_projections, names, projections_seen)?);
                Ok(Sort {
                    input,
                    by_column,
                    reverse,
                })
            }
            Explode { input, column } => {
                let input =
                    Box::new(self.push_down(*input, acc_projections, names, projections_seen)?);
                Ok(Explode { input, column })
            }
            Cache { input } => {
                let input =
                    Box::new(self.push_down(*input, acc_projections, names, projections_seen)?);
                Ok(Cache { input })
            }
            Distinct {
                input,
                maintain_order,
                subset,
            } => {
                if let Some(subset) = subset.as_ref() {
                    if !acc_projections.is_empty() {
                        for name in subset {
                            add_to_accumulated(&col(name), &mut acc_projections, &mut names)
                                .unwrap();
                        }
                    }
                };

                let input = self.push_down(*input, acc_projections, names, projections_seen)?;
                Ok(Distinct {
                    input: Box::new(input),
                    maintain_order,
                    subset,
                })
            }
            Selection { predicate, input } => {
                if !acc_projections.is_empty() {
                    add_to_accumulated(&predicate, &mut acc_projections, &mut names)?;
                };
                let input =
                    Box::new(self.push_down(*input, acc_projections, names, projections_seen)?);
                Ok(Selection { predicate, input })
            }
            Aggregate {
                input, keys, aggs, ..
            } => {
                // todo! remove unnecessary vec alloc.
                let (mut acc_projections, _local_projections, mut names) =
                    self.split_acc_projections(acc_projections, input.schema());

                // add the columns used in the aggregations to the projection
                for agg in &aggs {
                    add_to_accumulated(agg, &mut acc_projections, &mut names)?;
                }

                // make sure the keys are projected
                for key in &*keys {
                    add_to_accumulated(&col(key), &mut acc_projections, &mut names)?;
                }

                let lp = self.push_down(*input, acc_projections, names, projections_seen)?;
                let builder = LogicalPlanBuilder::from(lp).groupby(keys, aggs);
                Ok(builder.build())
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                how,
                allow_par,
                force_par,
                ..
            } => {
                let mut pushdown_left = init_vec();
                let mut pushdown_right = init_vec();
                let mut names_left = init_set();
                let mut names_right = init_set();
                let mut local_projection = init_vec();

                // if there are no projections we don't have to do anything
                if !acc_projections.is_empty() {
                    let schema_left = input_left.schema();
                    let schema_right = input_right.schema();

                    // We need the join columns so we push the projection downwards
                    pushdown_left.push(left_on.clone());
                    pushdown_right.push(right_on.clone());

                    for mut proj in acc_projections {
                        let mut add_local = true;

                        // if it is an alias we want to project the root column name downwards
                        // but we don't want to project it a this level, otherwise we project both
                        // the root and the alias, hence add_local = false.
                        if let Expr::Alias(expr, name) = proj {
                            let root_name = expr_to_root_column_name(&expr).unwrap();

                            proj = Expr::Column(root_name);
                            local_projection.push(Expr::Alias(Box::new(proj.clone()), name));

                            // now we don
                            add_local = false;
                        }

                        // Path for renamed columns due to the join. The column name of the left table
                        // stays as is, the column of the right will have the "_right" suffix.
                        // Thus joining two tables with both a foo column leads to ["foo", "foo_right"]
                        if !self.join_push_down(
                            schema_left,
                            schema_right,
                            &proj,
                            &mut pushdown_left,
                            &mut pushdown_right,
                            &mut names_left,
                            &mut names_right,
                        )? {
                            // Column name of the projection without any alias.
                            let root_column_name = expr_to_root_column_name(&proj).unwrap();

                            // If _right suffix exists we need to push a projection down without this
                            // suffix.
                            if root_column_name.ends_with("_right") {
                                // downwards name is the name without the _right i.e. "foo".
                                let (downwards_name, _) = root_column_name
                                    .split_at(root_column_name.len() - "_right".len());

                                // project downwards and locally immediately alias to prevent wrong projections
                                if names_right.insert(Arc::new(downwards_name.to_string())) {
                                    let projection = col(downwards_name);
                                    pushdown_right.push(projection);
                                }
                                // locally we project and alias
                                let projection =
                                    col(downwards_name).alias(&format!("{}_right", downwards_name));
                                local_projection.push(projection);
                            }
                        } else if add_local {
                            // always also do the projection locally, because the join columns may not be
                            // included in the projection.
                            // for instance:
                            //
                            // SELECT [COLUMN temp]
                            // FROM
                            // JOIN (["days", "temp"]) WITH (["days", "rain"]) ON (left: days right: days)
                            //
                            // should drop the days column after the join.
                            local_projection.push(proj)
                        }
                    }
                }
                let lp_left =
                    self.push_down(*input_left, pushdown_left, names_left, projections_seen)?;
                let lp_right =
                    self.push_down(*input_right, pushdown_right, names_right, projections_seen)?;
                let builder = LogicalPlanBuilder::from(lp_left)
                    .join(lp_right, how, left_on, right_on, allow_par, force_par);
                Ok(self.finish_node(local_projection, builder))
            }
            HStack { input, exprs, .. } => {
                // Make sure that columns selected with_columns are available
                // only if not empty. If empty we already select everything.
                if !acc_projections.is_empty() {
                    for expression in &exprs {
                        add_to_accumulated(expression, &mut acc_projections, &mut names)?;
                    }
                }

                let (acc_projections, _, names) =
                    self.split_acc_projections(acc_projections, input.schema());

                let lp = LogicalPlanBuilder::from(self.push_down(
                    *input,
                    acc_projections,
                    names,
                    projections_seen,
                )?)
                .with_columns(exprs)
                .build();
                Ok(lp)
            }
        }
    }
}

impl Optimize for ProjectionPushDown {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        self.push_down(logical_plan, init_vec(), init_set(), 0)
    }
}
