use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{aexpr_to_root_nodes, has_aexpr};
use polars_core::prelude::*;

pub(crate) struct AggregatePushdown {
    state: Vec<Node>,
    processed_state: bool,
}

impl AggregatePushdown {
    pub(crate) fn new() -> Self {
        AggregatePushdown {
            state: vec![],
            processed_state: false,
        }
    }
    fn drain_nodes(&mut self) -> impl Iterator<Item = Node> {
        self.processed_state = true;
        let state = std::mem::take(&mut self.state);
        state.into_iter()
    }

    fn pushdown_projection(
        &mut self,
        node: Node,
        expr: Vec<Node>,
        input: Node,
        schema: SchemaRef,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Option<ALogicalPlan> {
        let dummy_node = usize::max_value();
        let dummy_min = AExpr::Agg(AAggExpr::Min(Node(dummy_node)));
        let dummy_max = AExpr::Agg(AAggExpr::Max(Node(dummy_node)));
        let dummy_first = AExpr::Agg(AAggExpr::First(Node(dummy_node)));
        let dummy_last = AExpr::Agg(AAggExpr::First(Node(dummy_node)));
        let dummy_sum = AExpr::Agg(AAggExpr::Sum(Node(dummy_node)));

        // only do aggregation pushdown if all projections are aggregations
        #[allow(clippy::blocks_in_if_conditions)]
        if !self.processed_state
            && expr.iter().all(|node| {
                (has_aexpr(*node, expr_arena, &dummy_min)
                    || has_aexpr(*node, expr_arena, &dummy_max)
                    || has_aexpr(*node, expr_arena, &dummy_first)
                    || has_aexpr(*node, expr_arena, &dummy_sum)
                    || has_aexpr(*node, expr_arena, &dummy_last))
                    && {
                        let roots = aexpr_to_root_nodes(*node, expr_arena);
                        roots.len() == 1
                    }
            })
        {
            // add to state
            self.state.extend_from_slice(&expr);
            // swap projection with the input node
            let lp = lp_arena.take(input);
            Some(lp)
        } else {
            // restore lp node
            lp_arena.replace(
                node,
                ALogicalPlan::Projection {
                    expr,
                    input,
                    schema,
                },
            );
            None
        }
    }
}

impl OptimizationRule for AggregatePushdown {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        // here we take the lp, it is replaced by a default. We must restore the lp if we don't modify
        // it
        let lp = lp_arena.take(node);
        use ALogicalPlan::*;
        match lp {
            Projection {
                expr,
                input,
                schema,
            } => self.pushdown_projection(node, expr, input, schema, lp_arena, expr_arena),
            LocalProjection {
                expr,
                input,
                schema,
            } => self.pushdown_projection(node, expr, input, schema, lp_arena, expr_arena),
            // todo! hstack should pushown not dependent columns
            Join { .. } | Aggregate { .. } | HStack { .. } | DataFrameScan { .. } => {
                if self.state.is_empty() {
                    lp_arena.replace(node, lp);
                    None
                } else {
                    // we cannot pass a join or GroupBy so we do the projection here
                    let new_node = lp_arena.add(lp.clone());
                    let input_schema = lp_arena.get(new_node).schema(lp_arena);

                    let nodes: Vec<_> = self.drain_nodes().collect();
                    let fields = nodes
                        .iter()
                        .map(|n| {
                            expr_arena
                                .get(*n)
                                .to_field(input_schema, Context::Other, expr_arena)
                                .unwrap()
                        })
                        .collect();

                    Some(ALogicalPlan::Projection {
                        expr: nodes,
                        input: new_node,
                        schema: Arc::new(Schema::new(fields)),
                    })
                }
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
                ignore_errors,
                skip_rows,
                stop_after_n_rows,
                with_columns,
                predicate,
                aggregate,
                cache,
            } => match self.state.is_empty() {
                true => {
                    lp_arena.replace(
                        node,
                        CsvScan {
                            path,
                            schema,
                            has_header,
                            delimiter,
                            ignore_errors,
                            skip_rows,
                            stop_after_n_rows,
                            with_columns,
                            predicate,
                            aggregate,
                            cache,
                        },
                    );
                    None
                }
                false => {
                    let aggregate: Vec<_> = self.drain_nodes().collect();
                    Some(ALogicalPlan::CsvScan {
                        path,
                        schema,
                        has_header,
                        delimiter,
                        ignore_errors,
                        skip_rows,
                        stop_after_n_rows,
                        with_columns,
                        predicate,
                        aggregate,
                        cache,
                    })
                }
            },
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                with_columns,
                predicate,
                aggregate,
                stop_after_n_rows,
                cache,
            } => match self.state.is_empty() {
                true => {
                    lp_arena.replace(
                        node,
                        ParquetScan {
                            path,
                            schema,
                            with_columns,
                            predicate,
                            aggregate,
                            stop_after_n_rows,
                            cache,
                        },
                    );
                    None
                }
                false => {
                    let aggregate = self.drain_nodes().collect();
                    Some(ALogicalPlan::ParquetScan {
                        path,
                        schema,
                        with_columns,
                        predicate,
                        aggregate,
                        stop_after_n_rows,
                        cache,
                    })
                }
            },
            _ => {
                // restore lp
                lp_arena.replace(node, lp);
                None
            }
        }
    }
}
