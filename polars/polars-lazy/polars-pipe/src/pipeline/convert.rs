use std::sync::Arc;

use polars_core::prelude::PolarsResult;
use polars_core::with_match_physical_integer_polars_type;
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

fn get_source<F>(
    source: ALogicalPlan,
    operator_objects: &mut Vec<Arc<dyn Operator>>,
    expr_arena: &Arena<AExpr>,
    to_physical: &F,
    push_predicate: bool,
) -> PolarsResult<Box<dyn Source>>
where
    F: Fn(Node, &Arena<AExpr>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
    match source {
        #[cfg(feature = "csv-file")]
        CsvScan {
            path,
            schema,
            options,
            predicate,
            ..
        } => {
            // add predicate to operators
            if let (true, Some(predicate)) = (push_predicate, predicate) {
                let predicate = to_physical(predicate, expr_arena)?;
                let op = operators::FilterOperator { predicate };
                let op = Arc::new(op) as Arc<dyn Operator>;
                operator_objects.push(op)
            }
            let src = sources::CsvSource::new(path, schema, options)?;
            Ok(Box::new(src) as Box<dyn Source>)
        }
        #[cfg(feature = "parquet")]
        ParquetScan {
            path,
            schema,
            options,
            predicate,
            ..
        } => {
            // add predicate to operators
            if let (true, Some(predicate)) = (push_predicate, predicate) {
                let predicate = to_physical(predicate, expr_arena)?;
                let op = operators::FilterOperator { predicate };
                let op = Arc::new(op) as Arc<dyn Operator>;
                operator_objects.push(op)
            }
            let src = sources::ParquetSource::new(path, options, &schema)?;
            Ok(Box::new(src) as Box<dyn Source>)
        }
        _ => todo!(),
    }
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
                options,
                ..
            } => {
                let key_columns = Arc::new(
                    keys.iter()
                        .map(|node| to_physical(*node, expr_arena).unwrap())
                        .collect::<Vec<_>>(),
                );

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

                match (
                    output_schema.get_index(0).unwrap().1.to_physical(),
                    keys.len(),
                ) {
                    (dt, 1) if dt.is_integer() => {
                        with_match_physical_integer_polars_type!(dt, |$T| {
                            Box::new(groupby::PrimitiveGroupbySink::<$T>::new(
                                key_columns[0].clone(),
                                aggregation_columns,
                                agg_fns,
                                output_schema.clone(),
                                options.slice
                            )) as Box<dyn Sink>
                        })
                    }
                    _ => Box::new(groupby::GenericGroupbySink::new(
                        key_columns,
                        aggregation_columns,
                        agg_fns,
                        output_schema.clone(),
                        options.slice,
                    )) as Box<dyn Sink>,
                }
            }
            _ => {
                todo!()
            }
        },
    };

    let mut source_objects = Vec::with_capacity(sources.len());
    let mut operator_objects = Vec::with_capacity(operators.len() + 1);

    for node in sources {
        let src = match lp_arena.take(*node) {
            #[cfg(feature = "csv-file")]
            lp @ CsvScan { .. } => {
                get_source(lp, &mut operator_objects, expr_arena, &to_physical, true)?
            }
            #[cfg(feature = "parquet")]
            lp @ ParquetScan { .. } => {
                get_source(lp, &mut operator_objects, expr_arena, &to_physical, true)?
            }
            Union { inputs, .. } => {
                let sources = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let lp = lp_arena.take(*node);
                        // only push predicate of first source
                        get_source(lp, &mut operator_objects, expr_arena, &to_physical, i == 0)
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                Box::new(sources::UnionSource::new(sources)) as Box<dyn Source>
            }
            lp => {
                panic!("source {:?} not (yet) supported", lp)
            }
        };
        source_objects.push(src)
    }

    for node in operators.iter() {
        let op = match lp_arena.take(*node) {
            Projection { expr, .. } => {
                let op = operators::ProjectionOperator {
                    exprs: exprs_to_physical(&expr, expr_arena, &to_physical)?,
                };
                Arc::new(op) as Arc<dyn Operator>
            }
            Selection { predicate, .. } => {
                let predicate = to_physical(predicate, expr_arena)?;
                let op = operators::FilterOperator { predicate };
                Arc::new(op) as Arc<dyn Operator>
            }
            MapFunction {
                function: FunctionNode::FastProjection { columns },
                ..
            } => {
                let op = operators::FastProjectionOperator { columns };
                Arc::new(op) as Arc<dyn Operator>
            }

            lp => {
                panic!("operator {:?} not (yet) supported", lp)
            }
        };
        operator_objects.push(op)
    }

    Ok(Pipeline::new(source_objects, operator_objects, sink))
}
