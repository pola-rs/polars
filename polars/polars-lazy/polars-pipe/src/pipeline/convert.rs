use std::sync::Arc;

use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_plan::prelude::*;

use crate::executors::sinks::groupby::aggregates::convert_to_hash_agg;
use crate::executors::sinks::*;
use crate::executors::{operators, sources};
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{Operator, Sink, Source};
use crate::pipeline::PipeLine;

fn exprs_to_physical<F>(
    exprs: &[Node],
    expr_arena: &mut Arena<AExpr>,
    to_physical: &F,
    schema: Option<&SchemaRef>,
) -> PolarsResult<Vec<Arc<dyn PhysicalPipedExpr>>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    exprs
        .iter()
        .map(|node| to_physical(*node, expr_arena, schema))
        .collect()
}

fn get_source<F>(
    source: ALogicalPlan,
    operator_objects: &mut Vec<Box<dyn Operator>>,
    expr_arena: &Arena<AExpr>,
    to_physical: &F,
    push_predicate: bool,
) -> PolarsResult<Box<dyn Source>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
    match source {
        DataFrameScan {
            df,
            projection,
            selection,
            output_schema,
            ..
        } => {
            let mut df = (*df).clone();
            if push_predicate {
                if let Some(predicate) = selection {
                    let predicate = to_physical(predicate, expr_arena, output_schema.as_ref())?;
                    let op = operators::FilterOperator { predicate };
                    let op = Box::new(op) as Box<dyn Operator>;
                    operator_objects.push(op)
                }
                // projection is free
                if let Some(projection) = projection {
                    df = df.select(projection.as_slice())?;
                }
            }
            Ok(Box::new(sources::DataFrameSource::from_df(df)) as Box<dyn Source>)
        }
        #[cfg(feature = "csv-file")]
        CsvScan {
            path,
            file_info,
            options,
            predicate,
            output_schema,
            ..
        } => {
            // add predicate to operators
            if let (true, Some(predicate)) = (push_predicate, predicate) {
                let predicate = to_physical(predicate, expr_arena, output_schema.as_ref())?;
                let op = operators::FilterOperator { predicate };
                let op = Box::new(op) as Box<dyn Operator>;
                operator_objects.push(op)
            }
            let src = sources::CsvSource::new(path, file_info.schema, options)?;
            Ok(Box::new(src) as Box<dyn Source>)
        }
        #[cfg(feature = "parquet")]
        ParquetScan {
            path,
            file_info,
            options,
            cloud_options,
            predicate,
            output_schema,
            ..
        } => {
            // add predicate to operators
            if let (true, Some(predicate)) = (push_predicate, predicate) {
                let predicate = to_physical(predicate, expr_arena, output_schema.as_ref())?;
                let op = operators::FilterOperator { predicate };
                let op = Box::new(op) as Box<dyn Operator>;
                operator_objects.push(op)
            }
            let src = sources::ParquetSource::new(path, options, cloud_options, &file_info.schema)?;
            Ok(Box::new(src) as Box<dyn Source>)
        }
        _ => todo!(),
    }
}

pub fn get_sink<F>(
    node: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    to_physical: &F,
) -> PolarsResult<Box<dyn Sink>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
    let out = match lp_arena.get(node) {
        #[cfg(any(feature = "parquet", feature = "ipc"))]
        FileSink { input, payload } => {
            let path = payload.path.as_ref().as_path();
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            match &payload.file_type {
                #[cfg(feature = "parquet")]
                FileType::Parquet(options) => {
                    Box::new(ParquetSink::new(path, *options, input_schema.as_ref())?)
                        as Box<dyn Sink>
                }
                #[cfg(feature = "ipc")]
                FileType::Ipc(options) => {
                    Box::new(IpcSink::new(path, *options, input_schema.as_ref())?) as Box<dyn Sink>
                }
            }
        }
        Join {
            input_left,
            input_right,
            options,
            left_on,
            right_on,
            ..
        } => {
            // slice pushdown optimization should not set this one in a streaming query.
            assert!(options.slice.is_none());

            match &options.how {
                #[cfg(feature = "cross_join")]
                JoinType::Cross => {
                    Box::new(CrossJoin::new(options.suffix.clone())) as Box<dyn Sink>
                }
                join_type @ JoinType::Inner | join_type @ JoinType::Left => {
                    let input_schema_left = lp_arena.get(*input_left).schema(lp_arena);
                    let join_columns_left = Arc::new(exprs_to_physical(
                        left_on,
                        expr_arena,
                        to_physical,
                        Some(input_schema_left.as_ref()),
                    )?);
                    let input_schema_right = lp_arena.get(*input_right).schema(lp_arena);
                    let join_columns_right = Arc::new(exprs_to_physical(
                        right_on,
                        expr_arena,
                        to_physical,
                        Some(input_schema_right.as_ref()),
                    )?);

                    let swapped = swap_join_order(options);

                    let (join_columns_left, join_columns_right) = if swapped {
                        (join_columns_right, join_columns_left)
                    } else {
                        (join_columns_left, join_columns_right)
                    };

                    Box::new(GenericBuild::new(
                        Arc::from(options.suffix.as_ref()),
                        join_type.clone(),
                        swapped,
                        join_columns_left,
                        join_columns_right,
                    ))
                }
                _ => unimplemented!(),
            }
        }
        Slice { offset, len, .. } => {
            let slice = SliceSink::new(*offset as u64, *len as usize);
            Box::new(slice) as Box<dyn Sink>
        }
        Sort {
            input,
            by_column,
            args,
        } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            assert_eq!(by_column.len(), 1);
            let by_column = aexpr_to_leaf_names_iter(by_column[0], expr_arena)
                .next()
                .unwrap();
            let index = input_schema.try_index_of(by_column.as_ref())?;

            let sort_sink = SortSink::new(
                index,
                args.reverse[0],
                input_schema.into_owned(),
                args.slice,
            );
            Box::new(sort_sink) as Box<dyn Sink>
        }
        Aggregate {
            input,
            keys,
            aggs,
            schema: output_schema,
            options,
            ..
        } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena).as_ref().clone();
            let key_columns = Arc::new(exprs_to_physical(
                keys,
                expr_arena,
                to_physical,
                Some(&input_schema),
            )?);

            let mut aggregation_columns = Vec::with_capacity(aggs.len());
            let mut agg_fns = Vec::with_capacity(aggs.len());

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
                            input_schema,
                            output_schema.clone(),
                            options.slice
                        )) as Box<dyn Sink>
                    })
                }
                (DataType::Utf8, 1) => Box::new(groupby::Utf8GroupbySink::new(
                    key_columns[0].clone(),
                    aggregation_columns,
                    agg_fns,
                    input_schema,
                    output_schema.clone(),
                    options.slice,
                )) as Box<dyn Sink>,
                _ => Box::new(groupby::GenericGroupbySink::new(
                    key_columns,
                    aggregation_columns,
                    agg_fns,
                    input_schema,
                    output_schema.clone(),
                    options.slice,
                )) as Box<dyn Sink>,
            }
        }
        lp => {
            panic!("{lp:?} not implemented")
        }
    };
    Ok(out)
}

pub fn get_dummy_operator() -> Box<dyn Operator> {
    Box::new(operators::PlaceHolder {})
}

pub fn get_operator<F>(
    node: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    to_physical: &F,
) -> PolarsResult<Box<dyn Operator>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
    let op = match lp_arena.get(node) {
        Projection { expr, input, .. } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            let op = operators::ProjectionOperator {
                exprs: exprs_to_physical(expr, expr_arena, &to_physical, Some(&input_schema))?,
            };
            Box::new(op) as Box<dyn Operator>
        }
        HStack { exprs, input, .. } => {
            let input_schema = (*lp_arena.get(*input).schema(lp_arena)).clone();
            let op = operators::HstackOperator {
                exprs: exprs_to_physical(exprs, expr_arena, &to_physical, Some(&input_schema))?,
                input_schema,
            };
            Box::new(op) as Box<dyn Operator>
        }
        Selection { predicate, input } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            let predicate = to_physical(*predicate, expr_arena, Some(input_schema.as_ref()))?;
            let op = operators::FilterOperator { predicate };
            Box::new(op) as Box<dyn Operator>
        }
        MapFunction {
            function: FunctionNode::FastProjection { columns },
            ..
        } => {
            // TODO! pass schema to FastProjection so that
            // projection can be based on already known schema.
            let op = operators::FastProjectionOperator {
                columns: columns.clone(),
            };
            Box::new(op) as Box<dyn Operator>
        }
        MapFunction { function, .. } => {
            let op = operators::FunctionOperator {
                function: function.clone(),
            };
            Box::new(op) as Box<dyn Operator>
        }

        lp => {
            panic!("operator {lp:?} not (yet) supported")
        }
    };
    Ok(op)
}

#[allow(clippy::too_many_arguments)]
pub fn create_pipeline<F>(
    sources: &[Node],
    operators: Vec<Box<dyn Operator>>,
    operator_nodes: Vec<Node>,
    sink_nodes: Vec<(usize, Node)>,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    to_physical: F,
    verbose: bool,
) -> PolarsResult<PipeLine>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;

    let mut source_objects = Vec::with_capacity(sources.len());
    let mut operator_objects = Vec::with_capacity(operators.len() + 1);

    for node in sources {
        let src = match lp_arena.get(*node) {
            lp @ DataFrameScan { .. } => get_source(
                lp.clone(),
                &mut operator_objects,
                expr_arena,
                &to_physical,
                true,
            )?,
            #[cfg(feature = "csv-file")]
            lp @ CsvScan { .. } => get_source(
                lp.clone(),
                &mut operator_objects,
                expr_arena,
                &to_physical,
                true,
            )?,
            #[cfg(feature = "parquet")]
            lp @ ParquetScan { .. } => get_source(
                lp.clone(),
                &mut operator_objects,
                expr_arena,
                &to_physical,
                true,
            )?,
            Union { inputs, .. } => {
                let sources = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let lp = lp_arena.get(*node);
                        // only push predicate of first source
                        get_source(
                            lp.clone(),
                            &mut operator_objects,
                            expr_arena,
                            &to_physical,
                            i == 0,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                Box::new(sources::UnionSource::new(sources)) as Box<dyn Source>
            }
            lp => {
                panic!("source {lp:?} not (yet) supported")
            }
        };
        source_objects.push(src)
    }

    // this offset is because the source might have inserted operators
    let operator_offset = operator_objects.len();
    operator_objects.extend(operators);

    let mut sink_nodes = sink_nodes
        .into_iter()
        .map(|(offset, node)| {
            Ok((
                offset + operator_offset,
                node,
                get_sink(node, lp_arena, expr_arena, &to_physical)?,
            ))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    if sink_nodes.is_empty() ||
        // if this evaluates true
        // then there are still operators after the last sink
        // so we add a final sink to make sure the latest operators run
        sink_nodes[sink_nodes.len() - 1].0 < operator_nodes.len()
    {
        sink_nodes.push((
            operator_objects.len(),
            Node::default(),
            Box::new(OrderedSink::new()),
        ));
    }

    Ok(PipeLine::new(
        source_objects,
        operator_objects,
        operator_nodes,
        sink_nodes,
        operator_offset,
        verbose,
    ))
}

pub fn swap_join_order(options: &JoinOptions) -> bool {
    matches!(options.how, JoinType::Left)
        || match (options.rows_left, options.rows_right) {
            ((Some(left), _), (Some(right), _)) => left > right,
            ((_, left), (_, right)) => left > right,
        }
}
