use crate::lazy::logical_plan::optimizer::check_down_node;
use crate::lazy::prelude::*;
use crate::lazy::utils::{
    expr_to_root_column, expr_to_root_column_expr, expressions_to_root_column_exprs,
    projected_names, unpack_binary_exprs,
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
    predicate: &Expr,
    acc_projections: &mut Vec<Expr>,
    names: &mut HashSet<Arc<String>, RandomState>,
) -> Result<()> {
    if let Expr::Literal(_) = predicate {
        return Ok(());
    }
    match unpack_binary_exprs(predicate) {
        Ok((left, right)) => {
            add_to_accumulated(left, acc_projections, names)?;
            add_to_accumulated(right, acc_projections, names)?;
        }
        Err(_) => {
            let name = expr_to_root_column(predicate)?;
            if names.insert(name) {
                acc_projections.push(expr_to_root_column_expr(predicate)?.clone());
            }
        }
    }
    Ok(())
}

pub struct ProjectionPushDown {}

impl ProjectionPushDown {
    fn finish_at_leaf(&self, lp: LogicalPlan, acc_projections: Vec<Expr>) -> Result<LogicalPlan> {
        match acc_projections.len() {
            // There was no Projection in the logical plan
            0 => Ok(lp),
            _ => Ok(LogicalPlanBuilder::from(lp)
                .project(acc_projections)
                .build()),
        }
    }

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
                let name = expr_to_root_column(proj).unwrap();
                names.insert(name);
            }
            (acc_projections, local_projections, names)
        }
    }

    fn finish_node(
        &self,
        local_projections: Vec<Expr>,
        builder: LogicalPlanBuilder,
    ) -> Result<LogicalPlan> {
        if !local_projections.is_empty() {
            Ok(builder.project(local_projections).build())
        } else {
            Ok(builder.build())
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
        let name = expr_to_root_column(&proj)?;
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
    ) -> Result<LogicalPlan> {
        use LogicalPlan::*;
        match logical_plan {
            Projection { expr, input, .. } => {
                // add the root of the projections to accumulation,
                // but also do them locally to keep the schema and the alias.
                for e in &expr {
                    let expr = expr_to_root_column_expr(e)?;
                    let name = expr_to_root_column(expr)?;
                    if names.insert(name) {
                        acc_projections.push(expr.clone());
                    }
                }

                let (acc_projections, _local_projections, names) =
                    self.split_acc_projections(acc_projections, input.schema());
                let lp = self.push_down(*input, acc_projections, names)?;

                let mut local_projection = Vec::with_capacity(expr.len());
                for expr in expr {
                    if expr.to_field(lp.schema()).is_ok() {
                        local_projection.push(expr);
                    }
                }

                let builder = LogicalPlanBuilder::from(lp);
                self.finish_node(local_projection, builder)
            }
            LocalProjection { expr, input, .. } => {
                let lp = self.push_down(*input, acc_projections, names)?;
                let schema = lp.schema();
                // projection from a wildcard may be dropped if the schema changes due to the optimization
                let proj = expr
                    .into_iter()
                    .filter(|e| check_down_node(e, schema))
                    .collect();
                Ok(LogicalPlanBuilder::from(lp).project_local(proj).build())
            }
            DataFrameScan { df, schema } => {
                let lp = DataFrameScan { df, schema };
                self.finish_at_leaf(lp, acc_projections)
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
            } => {
                let lp = CsvScan {
                    path,
                    schema,
                    has_header,
                    delimiter,
                };
                self.finish_at_leaf(lp, acc_projections)
            }
            DataFrameOp { input, operation } => {
                let input = self.push_down(*input, acc_projections, names)?;
                Ok(DataFrameOp {
                    input: Box::new(input),
                    operation,
                })
            }
            Selection { predicate, input } => {
                let local_projections = if !acc_projections.is_empty() {
                    let local_projections = projected_names(&acc_projections)?;
                    add_to_accumulated(&predicate, &mut acc_projections, &mut names)?;

                    local_projections
                } else {
                    vec![]
                };

                let builder =
                    LogicalPlanBuilder::from(self.push_down(*input, acc_projections, names)?)
                        .filter(predicate);
                self.finish_node(local_projections, builder)
            }
            Aggregate {
                input, keys, aggs, ..
            } => {
                let (acc_projections, local_projections) = if !acc_projections.is_empty() {
                    // todo! remove unnecessary vec alloc.
                    let (mut acc_projections, _local_projections, mut names) =
                        self.split_acc_projections(acc_projections, input.schema());

                    // add the columns used in the aggregations to the projection
                    // todo! remove aggregations that aren't selected?
                    let root_projections = expressions_to_root_column_exprs(&aggs)?;

                    for proj in root_projections {
                        let name = expr_to_root_column(&proj)?;
                        if names.insert(name) {
                            acc_projections.push(proj)
                        }
                    }

                    // todo! maybe we need this later if an uptree udf needs a column?
                    // create local projections. This is the key plus the aggregations
                    let mut local_projections = Vec::with_capacity(aggs.len() + keys.len());

                    // make sure the keys are projected
                    for key in &*keys {
                        local_projections.push(col(key));

                        acc_projections.push(col(key))
                    }
                    for agg in &aggs {
                        local_projections.push(col(agg.to_field(input.schema())?.name()))
                    }

                    (acc_projections, local_projections)
                } else {
                    (vec![], vec![])
                };

                let lp = self.push_down(*input, acc_projections, names)?;
                let builder = LogicalPlanBuilder::from(lp).groupby(keys, aggs);
                self.finish_node(local_projections, builder)
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                how,
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
                            let root_name = expr_to_root_column(&expr).unwrap();

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
                            let root_column_name = expr_to_root_column(&proj).unwrap();

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
                let lp_left = self.push_down(*input_left, pushdown_left, names_left)?;
                let lp_right = self.push_down(*input_right, pushdown_right, names_right)?;
                let builder =
                    LogicalPlanBuilder::from(lp_left).join(lp_right, how, left_on, right_on);
                self.finish_node(local_projection, builder)
            }
            HStack { input, exprs, .. } => {
                // just the original projections at this level that may be renamed
                let local_renamed_projections = projected_names(&acc_projections)?;

                // Make sure that columns selected with_columns are available
                // only if not empty. If empty we already select everything.
                if !acc_projections.is_empty() {
                    for expression in &exprs {
                        // todo! maybe we should loop or recurse to find all binary expressions?
                        match expr_to_root_column_expr(expression) {
                            Ok(e) => acc_projections.push(e.clone()),
                            Err(_) => {
                                // could be:
                                //   * alias(lit)
                                //   * lit
                                //   * binary expr
                                // may fail, for literal cases
                                if let Ok((left, right)) = unpack_binary_exprs(expression) {
                                    expr_to_root_column_expr(left)
                                        .map(|p| acc_projections.push(p.clone()))
                                        .ok();
                                    expr_to_root_column_expr(right)
                                        .map(|p| acc_projections.push(p.clone()))
                                        .ok();
                                }
                            }
                        }
                    }
                }

                let (acc_projections, _, names) =
                    self.split_acc_projections(acc_projections, input.schema());

                let builder =
                    LogicalPlanBuilder::from(self.push_down(*input, acc_projections, names)?)
                        .with_columns(exprs);
                // locally re-project all columns plus the stacked columns to keep the order of the schema equal
                self.finish_node(local_renamed_projections, builder)
            }
        }
    }
}

impl Optimize for ProjectionPushDown {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        self.push_down(logical_plan, init_vec(), init_set())
    }
}
