use polars_core::prelude::*;

use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{aexpr_to_root_nodes, has_aexpr};

pub(crate) struct AggregatePushdown {
    accumulated_projections: Vec<Node>,
    processed_state: bool,
}

impl AggregatePushdown {
    pub(crate) fn new() -> Self {
        AggregatePushdown {
            accumulated_projections: vec![],
            processed_state: false,
        }
    }
    fn process_nodes(&mut self) -> Vec<Node> {
        self.processed_state = true;
        std::mem::take(&mut self.accumulated_projections)
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
        // only do aggregation pushdown if all projections are aggregations
        #[allow(clippy::blocks_in_if_conditions)]
        if !self.processed_state
            && expr.iter().all(|node| {
                has_aexpr(*node, expr_arena, |e| matches!(e, AExpr::Agg(_))) && {
                    let roots = aexpr_to_root_nodes(*node, expr_arena);
                    roots.len() == 1
                }
            })
        {
            // add to state
            self.accumulated_projections.extend_from_slice(&expr);
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
                if self.accumulated_projections.is_empty() {
                    lp_arena.replace(node, lp);
                    None
                } else {
                    // we cannot pass a join or GroupBy so we do the projection here
                    let new_node = lp_arena.add(lp.clone());
                    let input_schema = lp_arena.get(new_node).schema(lp_arena);

                    let nodes: Vec<_> = self.process_nodes();
                    let fields = self
                        .accumulated_projections
                        .iter()
                        .map(|n| {
                            expr_arena
                                .get(*n)
                                .to_field(input_schema, Context::Default, expr_arena)
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
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                schema,
                output_schema,
                options,
                predicate,
                aggregate,
            } => match self.accumulated_projections.is_empty() {
                true => {
                    lp_arena.replace(
                        node,
                        CsvScan {
                            path,
                            schema,
                            output_schema,
                            options,
                            predicate,
                            aggregate,
                        },
                    );
                    None
                }
                false => {
                    let aggregate: Vec<_> = self.process_nodes();
                    Some(ALogicalPlan::CsvScan {
                        path,
                        schema,
                        output_schema,
                        options,
                        predicate,
                        aggregate,
                    })
                }
            },
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                output_schema,
                with_columns,
                predicate,
                aggregate,
                n_rows,
                cache,
            } => match self.accumulated_projections.is_empty() {
                true => {
                    lp_arena.replace(
                        node,
                        ParquetScan {
                            path,
                            schema,
                            output_schema,
                            with_columns,
                            predicate,
                            aggregate,
                            n_rows,
                            cache,
                        },
                    );
                    None
                }
                false => {
                    let aggregate = self.process_nodes();
                    Some(ALogicalPlan::ParquetScan {
                        path,
                        schema,
                        output_schema,
                        with_columns,
                        predicate,
                        aggregate,
                        n_rows,
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
