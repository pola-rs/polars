use std::sync::Arc;

use polars_core::error::PolarsError;
use polars_core::prelude::{DataType, Int32Type, Int64Type, PolarsResult};
use polars_plan::prelude::*;

use crate::executors::sinks::groupby::aggregates::convert_to_hash_agg;
use crate::executors::sinks::{groupby, OrderedSink};
use crate::executors::{operators, sources};
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{Operator, Sink, Source};
use crate::pipeline::Pipeline;

fn exprs_to_physical<F>(
    exprs: &[Node],
    expr_arena: &mut Arena<AExpr>,
    to_physical: &F,
) -> PolarsResult<Vec<Arc<dyn PhysicalPipedExpr>>>
where
    F: Fn(Node, &Arena<AExpr>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    exprs
        .iter()
        .map(|node| to_physical(*node, expr_arena))
        .collect()
}

pub fn create_pipeline<F>(
    sources: &[Node],
    operators: &[Node],
    sink: Option<Node>,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    to_physical: F,
) -> PolarsResult<Pipeline>
where
    F: Fn(Node, &Arena<AExpr>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;

    let sink = match sink {
        None => Box::new(OrderedSink::new()) as Box<dyn Sink>,
        Some(node) => match lp_arena.get(node) {
            Aggregate {
                input,
                keys,
                aggs,
                schema: output_schema,
                ..
            } => {
                assert_eq!(keys.len(), 1);
                if let AExpr::Column(key) = expr_arena.get(keys[0]) {
                    let mut aggregation_columns = Vec::with_capacity(aggs.len());
                    let mut agg_fns = Vec::with_capacity(aggs.len());

                    let input_schema = lp_arena.get(*input).schema(lp_arena);

                    for node in aggs {
                        let (index, agg_fn) =
                            convert_to_hash_agg(*node, expr_arena, &input_schema, &to_physical);
                        aggregation_columns.push(index);
                        agg_fns.push(agg_fn)
                    }
                    let aggregation_columns = Arc::new(aggregation_columns);
                    match input_schema
                        .get(key)
                        .ok_or_else(|| PolarsError::NotFound(format!("{}", key.as_ref()).into()))?
                    {
                        DataType::Int64 => {
                            Box::new(groupby::PrimitiveGroupbySink::<Int64Type>::new(
                                key.clone(),
                                aggregation_columns,
                                agg_fns,
                                output_schema.clone(),
                            )) as Box<dyn Sink>
                        }
                        DataType::Int32 => {
                            Box::new(groupby::PrimitiveGroupbySink::<Int32Type>::new(
                                key.clone(),
                                aggregation_columns,
                                agg_fns,
                                output_schema.clone(),
                            )) as Box<dyn Sink>
                        }
                        dt => panic!("dtype: '{}' not yet implemented in streaming", dt),
                    }
                } else {
                    unreachable!()
                }
            }
            _ => {
                todo!()
            }
        },
    };

    let mut source_objects = Vec::with_capacity(sources.len());

    for node in sources {
        match lp_arena.take(*node) {
            ALogicalPlan::CsvScan {
                path,
                schema,
                options,
                predicate,
                aggregate,
                ..
            } => {
                // todo! remove aggregate pushdown
                assert!(aggregate.is_empty());
                let src = sources::CsvSource::new(
                    path,
                    schema,
                    options,
                    predicate.map(|node| to_physical(node, expr_arena).unwrap()),
                )?;
                source_objects.push(Arc::new(src) as Arc<dyn Source>)
            }
            lp => {
                panic!("source {:?} not (yet) supported", lp)
            }
        }
    }

    let operators = operators
        .iter()
        .map(|node| match lp_arena.take(*node) {
            Projection { expr, .. } => {
                let op = operators::ProjectionOperator {
                    exprs: exprs_to_physical(&expr, expr_arena, &to_physical)?,
                };
                Ok(Arc::new(op) as Arc<dyn Operator>)
            }
            Selection { predicate, .. } => {
                let predicate = to_physical(predicate, expr_arena)?;
                let op = operators::FilterOperator { predicate };
                Ok(Arc::new(op) as Arc<dyn Operator>)
            }
            MapFunction {
                function: FunctionNode::FastProjection { columns },
                ..
            } => {
                let op = operators::FastProjectionOperator { columns };
                Ok(Arc::new(op) as Arc<dyn Operator>)
            }

            lp => {
                panic!("operator {:?} not (yet) supported", lp)
            }
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(Pipeline::new(source_objects, operators, sink))
}
