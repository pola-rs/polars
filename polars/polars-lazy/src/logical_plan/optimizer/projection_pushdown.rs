use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{aexpr_to_root_names, aexpr_to_root_nodes, check_input_node, has_aexpr};
use polars_core::{datatypes::PlHashSet, prelude::*};

fn init_vec() -> Vec<Node> {
    Vec::with_capacity(100)
}
fn init_set() -> PlHashSet<Arc<str>> {
    PlHashSet::with_capacity(32)
}

/// utility function to get names of the columns needed in projection at scan level
#[cfg(any(feature = "parquet", feature = "csv-file"))]
fn get_scan_columns(
    acc_projections: &mut Vec<Node>,
    expr_arena: &Arena<AExpr>,
) -> Option<Vec<String>> {
    let mut with_columns = None;
    if !acc_projections.is_empty() {
        let mut columns = Vec::with_capacity(acc_projections.len());
        for expr in acc_projections {
            for name in aexpr_to_root_names(*expr, expr_arena) {
                columns.push((*name).to_owned())
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
) -> (Vec<Node>, Vec<Node>, PlHashSet<Arc<str>>) {
    // If node above has as many columns as the projection there is nothing to pushdown.
    if down_schema.fields().len() == acc_projections.len() {
        let local_projections = acc_projections;
        (vec![], local_projections, PlHashSet::new())
    } else {
        let (acc_projections, local_projections): (Vec<Node>, Vec<Node>) = acc_projections
            .into_iter()
            .partition(|expr| check_input_node(*expr, down_schema, expr_arena));
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
fn add_expr_to_accumulated(
    expr: Node,
    acc_projections: &mut Vec<Node>,
    projected_names: &mut PlHashSet<Arc<str>>,
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
) -> Result<Schema> {
    let mut new_fields = Vec::with_capacity(acc_projections.len());
    for node in acc_projections.iter() {
        for name in aexpr_to_root_names(*node, expr_arena) {
            let field = schema.field_with_name(&*name)?;
            new_fields.push(field.clone())
        }
    }
    Ok(Schema::new(new_fields))
}

pub(crate) struct ProjectionPushDown {}

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

    #[allow(clippy::too_many_arguments)]
    fn join_push_down(
        &self,
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
        let names = aexpr_to_root_names(proj, expr_arena);
        let root_projections = aexpr_to_root_nodes(proj, expr_arena);

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

    /// Helper method. This pushes down current node and assigns the result to this node.
    fn pushdown_and_assign(
        &self,
        input: Node,
        acc_projections: Vec<Node>,
        names: PlHashSet<Arc<str>>,
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
        mut projected_names: PlHashSet<Arc<str>>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
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
                            if projected_names.remove(name) {
                                acc_projections = acc_projections
                                    .into_iter()
                                    .filter(|expr| {
                                        !aexpr_to_root_names(*expr, expr_arena).contains(name)
                                    })
                                    .collect();
                            }
                        }
                    }

                    add_expr_to_accumulated(
                        *e,
                        &mut acc_projections,
                        &mut projected_names,
                        expr_arena,
                    );
                }

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    projected_names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                let lp = lp_arena.get(input);

                let mut local_projection = Vec::with_capacity(expr.len());

                // the projections should all be done at the latest projection node to keep the same schema order
                if projections_seen == 0 {
                    let schema = lp.schema(lp_arena);
                    for expr in expr {
                        // Due to the pushdown, a lot of projections cannot be done anymore at the final
                        // node and should be skipped
                        if expr_arena
                            .get(expr)
                            .to_field(schema, Context::Default, expr_arena)
                            .is_ok()
                        {
                            local_projection.push(expr);
                        }
                    }
                    // only aliases should be projected locally in the rest of the projections.
                } else {
                    for expr in expr {
                        if has_aexpr(expr, expr_arena, |e| matches!(e, AExpr::Alias(_, _))) {
                            local_projection.push(expr)
                        }
                    }
                }

                let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena);
                let lp = self.finish_node(local_projection, builder);
                Ok(lp)
            }
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
                    .filter(|e| check_input_node(*e, schema, expr_arena))
                    .collect();
                Ok(ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                    .project_local(proj)
                    .build())
            }
            DataFrameScan {
                df,
                mut schema,
                selection,
                ..
            } => {
                let mut projection = None;
                if !acc_projections.is_empty() {
                    schema = Arc::new(update_scan_schema(&acc_projections, expr_arena, &*schema)?);
                    projection = Some(acc_projections);
                }
                let lp = DataFrameScan {
                    df,
                    schema,
                    projection,
                    selection,
                };
                Ok(lp)
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                schema,
                predicate,
                aggregate,
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
                        &*schema,
                    )?))
                };
                options.with_columns = with_columns;

                let lp = IpcScan {
                    path,
                    schema,
                    output_schema,
                    predicate,
                    aggregate,
                    options,
                };
                Ok(lp)
            }

            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                predicate,
                aggregate,
                n_rows,
                cache,
                ..
            } => {
                let with_columns = get_scan_columns(&mut acc_projections, expr_arena);
                let output_schema = if with_columns.is_none() {
                    None
                } else {
                    Some(Arc::new(update_scan_schema(
                        &acc_projections,
                        expr_arena,
                        &*schema,
                    )?))
                };

                let lp = ParquetScan {
                    path,
                    schema,
                    output_schema,
                    with_columns,
                    predicate,
                    aggregate,
                    n_rows,
                    cache,
                };
                Ok(lp)
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                schema,
                mut options,
                predicate,
                aggregate,
                ..
            } => {
                options.with_columns = get_scan_columns(&mut acc_projections, expr_arena);

                let output_schema = if options.with_columns.is_none() {
                    None
                } else {
                    Some(Arc::new(update_scan_schema(
                        &acc_projections,
                        expr_arena,
                        &*schema,
                    )?))
                };

                let lp = CsvScan {
                    path,
                    schema,
                    output_schema,
                    options,
                    predicate,
                    aggregate,
                };
                Ok(lp)
            }
            Sort {
                input,
                by_column,
                reverse,
            } => {
                if !acc_projections.is_empty() {
                    // Make sure that the column(s) used for the sort is projected
                    by_column.iter().for_each(|node| {
                        aexpr_to_root_nodes(*node, expr_arena)
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
                    reverse,
                })
            }
            Explode { input, columns } => {
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
                Ok(Explode { input, columns })
            }
            Distinct {
                input,
                maintain_order,
                subset,
            } => {
                // make sure that the set of unique columns is projected
                if let Some(subset) = (&*subset).as_ref() {
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
                Ok(Distinct {
                    input,
                    maintain_order,
                    subset,
                })
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
                id_vars.iter().for_each(|name| {
                    add_str_to_accumulated(
                        name,
                        &mut acc_projections,
                        &mut projected_names,
                        expr_arena,
                    )
                });
                value_vars.iter().for_each(|name| {
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
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;

                let builder =
                    ALogicalPlanBuilder::new(input, expr_arena, lp_arena).melt(id_vars, value_vars);
                Ok(self.finish_node(local_projections, builder))
            }
            Aggregate {
                input,
                keys,
                aggs,
                apply,
                schema,
                maintain_order,
                dynamic_options,
            } => {
                // the custom function may need all columns so we do the projections here.
                if let Some(f) = apply {
                    let lp = Aggregate {
                        input,
                        keys,
                        aggs,
                        schema,
                        apply: Some(f),
                        maintain_order,
                        dynamic_options,
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
                        add_expr_to_accumulated(*agg, &mut acc_projections, &mut names, expr_arena);
                    }

                    // make sure the keys are projected
                    for key in &*keys {
                        add_expr_to_accumulated(*key, &mut acc_projections, &mut names, expr_arena);
                    }

                    // make sure that the dynamic key is projected
                    if let Some(options) = &dynamic_options {
                        let node =
                            expr_arena.add(AExpr::Column(Arc::from(options.time_column.as_str())));
                        add_expr_to_accumulated(node, &mut acc_projections, &mut names, expr_arena);
                    }

                    self.pushdown_and_assign(
                        input,
                        acc_projections,
                        names,
                        projections_seen,
                        lp_arena,
                        expr_arena,
                    )?;

                    let builder = ALogicalPlanBuilder::new(input, expr_arena, lp_arena).groupby(
                        keys,
                        aggs,
                        apply,
                        maintain_order,
                        dynamic_options,
                    );
                    Ok(builder.build())
                }
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                let mut pushdown_left = init_vec();
                let mut pushdown_right = init_vec();
                let mut names_left = init_set();
                let mut names_right = init_set();
                let mut local_projection = init_vec();

                // if there are no projections we don't have to do anything (all columns are projected)
                // otherwise we build local projections to sort out proper column names due to the
                // join operation
                //
                // Joins on columns with different names, for example
                // left_on = "a", right_on = "b
                // will remove the name "b" (it is "a" now). That columns should therefore not
                // be added to a local projection.
                if !acc_projections.is_empty() {
                    let schema_left = lp_arena.get(input_left).schema(lp_arena);
                    let schema_right = lp_arena.get(input_right).schema(lp_arena);

                    // We need the join columns so we push the projection downwards
                    for e in &left_on {
                        add_expr_to_accumulated(
                            *e,
                            &mut pushdown_left,
                            &mut names_left,
                            expr_arena,
                        );
                        if !local_projection.contains(e) {
                            local_projection.push(*e)
                        }
                    }
                    for e in &right_on {
                        add_expr_to_accumulated(
                            *e,
                            &mut pushdown_right,
                            &mut names_right,
                            expr_arena,
                        );
                        // we don't add right column names to local_projection as they are removed
                    }

                    for proj in acc_projections {
                        let mut add_local = true;

                        // if it is an alias we want to project the root column name downwards
                        // but we don't want to project it a this level, otherwise we project both
                        // the root and the alias, hence add_local = false.
                        if let AExpr::Alias(expr, name) = expr_arena.get(proj).clone() {
                            for root_name in aexpr_to_root_names(expr, expr_arena) {
                                let node = expr_arena.add(AExpr::Column(root_name));
                                let proj = expr_arena.add(AExpr::Alias(node, name.clone()));
                                local_projection.push(proj)
                            }
                            // now we don
                            add_local = false;
                        }

                        // Path for renamed columns due to the join. The column name of the left table
                        // stays as is, the column of the right will have the "_right" suffix.
                        // Thus joining two tables with both a foo column leads to ["foo", "foo_right"]
                        if !self.join_push_down(
                            schema_left,
                            schema_right,
                            proj,
                            &mut pushdown_left,
                            &mut pushdown_right,
                            &mut names_left,
                            &mut names_right,
                            expr_arena,
                        ) {
                            // Column name of the projection without any alias.
                            let root_column_name =
                                aexpr_to_root_names(proj, expr_arena).pop().unwrap();

                            let suffix = options.suffix.as_deref().unwrap_or("_right");
                            // If _right suffix exists we need to push a projection down without this
                            // suffix.
                            if root_column_name.ends_with(suffix) {
                                // downwards name is the name without the _right i.e. "foo".
                                let (downwards_name, _) = root_column_name
                                    .split_at(root_column_name.len() - suffix.len());

                                let downwards_name_column =
                                    expr_arena.add(AExpr::Column(Arc::from(downwards_name)));
                                // project downwards and locally immediately alias to prevent wrong projections
                                if names_right.insert(Arc::from(downwards_name)) {
                                    pushdown_right.push(downwards_name_column);
                                }

                                // locally we project and alias
                                let projection = expr_arena.add(AExpr::Alias(
                                    downwards_name_column,
                                    Arc::from(format!("{}{}", downwards_name, suffix)),
                                ));
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
                            local_projection.push(proj);
                        }
                    }
                }

                self.pushdown_and_assign(
                    input_left,
                    pushdown_left,
                    names_left,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                self.pushdown_and_assign(
                    input_right,
                    pushdown_right,
                    names_right,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;

                let builder = ALogicalPlanBuilder::new(input_left, expr_arena, lp_arena).join(
                    input_right,
                    left_on,
                    right_on,
                    options,
                );
                Ok(self.finish_node(local_projection, builder))
            }
            HStack { input, exprs, .. } => {
                // Make sure that columns selected with_columns are available
                // only if not empty. If empty we already select everything.
                if !acc_projections.is_empty() {
                    for expression in &exprs {
                        add_expr_to_accumulated(
                            *expression,
                            &mut acc_projections,
                            &mut projected_names,
                            expr_arena,
                        );
                    }
                }

                let (acc_projections, _, names) = split_acc_projections(
                    acc_projections,
                    lp_arena.get(input).schema(lp_arena),
                    expr_arena,
                );

                self.pushdown_and_assign(
                    input,
                    acc_projections,
                    names,
                    projections_seen,
                    lp_arena,
                    expr_arena,
                )?;
                let lp = ALogicalPlanBuilder::new(input, expr_arena, lp_arena)
                    .with_columns(exprs)
                    .build();
                Ok(lp)
            }
            Udf {
                input,
                function,
                predicate_pd,
                projection_pd,
                schema,
            } => {
                if projection_pd {
                    self.pushdown_and_assign(
                        input,
                        acc_projections,
                        projected_names,
                        projections_seen,
                        lp_arena,
                        expr_arena,
                    )?;
                }
                Ok(Udf {
                    input,
                    function,
                    predicate_pd,
                    projection_pd,
                    schema,
                })
            }
            // Slice and Cache have only inputs and exprs, so we can use same logic.
            lp @ Slice { .. } | lp @ Cache { .. } | lp @ Union { .. } => {
                let inputs = lp.get_inputs();
                let exprs = lp.get_exprs();

                let new_inputs = if inputs.len() == 1 {
                    let node = inputs[0];
                    let alp = lp_arena.take(node);
                    let alp = self.push_down(
                        alp,
                        acc_projections,
                        projected_names,
                        projections_seen,
                        lp_arena,
                        expr_arena,
                    )?;
                    lp_arena.replace(node, alp);
                    vec![node]
                } else {
                    inputs
                        .iter()
                        .map(|&node| {
                            let alp = lp_arena.take(node);
                            let alp = self.push_down(
                                alp,
                                acc_projections.clone(),
                                projected_names.clone(),
                                projections_seen,
                                lp_arena,
                                expr_arena,
                            )?;
                            lp_arena.replace(node, alp);
                            Ok(node)
                        })
                        .collect::<Result<Vec<_>>>()?
                };

                Ok(lp.from_exprs_and_input(exprs, new_inputs))
            }
        }
    }

    pub fn optimize(
        &self,
        logical_plan: ALogicalPlan,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        let acc_predicates = init_vec();
        let names = init_set();
        self.push_down(logical_plan, acc_predicates, names, 0, lp_arena, expr_arena)
    }
}
