use std::sync::Arc;

use polars_core::prelude::PolarsResult;
use polars_plan::prelude::*;

use crate::executors::sinks::OrderedSink;
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
    F: Fn(Node, &mut Arena<AExpr>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
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
    F: Fn(Node, &mut Arena<AExpr>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
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
            ALogicalPlan::Projection { expr, .. } => {
                let op = operators::ProjectionOperator {
                    exprs: exprs_to_physical(&expr, expr_arena, &to_physical)?,
                };
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

    let sink = match sink {
        None => Box::new(OrderedSink::new()) as Box<dyn Sink>,
        Some(node) => {
            todo!()
        }
    };

    Ok(Pipeline::new(source_objects, operators, sink))
}
