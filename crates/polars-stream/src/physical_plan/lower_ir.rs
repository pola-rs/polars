use std::sync::Arc;

use polars_core::prelude::PlHashMap;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::reduce::can_convert_into_reduction;
use polars_plan::plans::{AExpr, Context, IR};
use polars_plan::prelude::SinkType;
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey, PhysNodeKind};

fn is_streamable(node: Node, arena: &Arena<AExpr>) -> bool {
    polars_plan::plans::is_streamable(node, arena, Context::Default)
}

#[recursive::recursive]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
) -> PolarsResult<PhysNodeKey> {
    let ir_node = ir_arena.get(node);
    let node_kind = match ir_node {
        IR::SimpleProjection { input, columns } => {
            let columns = columns.iter_names().map(|s| s.to_string()).collect();
            let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;
            PhysNodeKind::SimpleProjection {
                input: phys_input,
                columns,
            }
        },

        // TODO: split partially streamable selections to avoid fallback as much as possible.
        IR::Select { input, expr, .. }
            if expr.iter().all(|e| is_streamable(e.node(), expr_arena)) =>
        {
            let selectors = expr.clone();
            let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;
            PhysNodeKind::Select {
                input: phys_input,
                selectors,
                extend_original: false,
            }
        },

        // TODO: split partially streamable selections to avoid fallback as much as possible.
        IR::HStack { input, exprs, .. }
            if exprs.iter().all(|e| is_streamable(e.node(), expr_arena)) =>
        {
            let selectors = exprs.clone();
            let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;
            PhysNodeKind::Select {
                input: phys_input,
                selectors,
                extend_original: true,
            }
        },

        // TODO: split reductions and streamable selections. E.g. sum(a) + sum(b) should be split
        // into Select(a + b) -> Reduce(sum(a), sum(b)
        IR::Select { input, expr, .. }
            if expr
                .iter()
                .all(|e| can_convert_into_reduction(e.node(), expr_arena)) =>
        {
            let exprs = expr.clone();
            let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;
            PhysNodeKind::Reduce {
                input: phys_input,
                exprs,
            }
        },

        IR::Slice { input, offset, len } => {
            if *offset >= 0 {
                let offset = *offset as usize;
                let length = *len as usize;
                let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;
                PhysNodeKind::StreamingSlice {
                    input: phys_input,
                    offset,
                    length,
                }
            } else {
                todo!()
            }
        },

        IR::Filter { input, predicate } if is_streamable(predicate.node(), expr_arena) => {
            let predicate = predicate.clone();
            let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;
            PhysNodeKind::Filter {
                input: phys_input,
                predicate,
            }
        },

        IR::DataFrameScan {
            df,
            output_schema: projection,
            filter,
            schema,
            ..
        } => {
            let mut schema = schema.clone(); // This is initially the schema of df, but can change with the projection.
            let mut node_kind = PhysNodeKind::InMemorySource { df: df.clone() };

            if let Some(projection_schema) = projection {
                let phys_input = phys_sm.insert(PhysNode::new(schema, node_kind));
                node_kind = PhysNodeKind::SimpleProjection {
                    input: phys_input,
                    columns: projection_schema
                        .iter_names()
                        .map(|s| s.to_string())
                        .collect(),
                };
                schema = projection_schema.clone();
            }

            if let Some(predicate) = filter.clone() {
                if !is_streamable(predicate.node(), expr_arena) {
                    todo!()
                }

                let phys_input = phys_sm.insert(PhysNode::new(schema, node_kind));
                node_kind = PhysNodeKind::Filter {
                    input: phys_input,
                    predicate,
                };
            }

            node_kind
        },

        IR::Sink { input, payload } => {
            if *payload == SinkType::Memory {
                let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;
                PhysNodeKind::InMemorySink { input: phys_input }
            } else {
                todo!()
            }
        },

        IR::MapFunction { input, function } => {
            let function = function.clone();
            let phys_input = lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?;

            if function.is_streamable() {
                let map = Arc::new(move |df| function.evaluate(df));
                PhysNodeKind::Map {
                    input: phys_input,
                    map,
                }
            } else {
                let map = Arc::new(move |df| function.evaluate(df));
                PhysNodeKind::InMemoryMap {
                    input: phys_input,
                    map,
                }
            }
        },

        IR::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => PhysNodeKind::Sort {
            by_column: by_column.clone(),
            slice: *slice,
            sort_options: sort_options.clone(),
            input: lower_ir(*input, ir_arena, expr_arena, phys_sm, schema_cache)?,
        },

        IR::Union { inputs, options } => {
            if options.slice.is_some() {
                todo!()
            }

            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir(input, ir_arena, expr_arena, phys_sm, schema_cache))
                .collect::<Result<_, _>>()?;
            PhysNodeKind::OrderedUnion { inputs }
        },

        IR::HConcat {
            inputs,
            schema: _,
            options: _,
        } => {
            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir(input, ir_arena, expr_arena, phys_sm, schema_cache))
                .collect::<Result<_, _>>()?;
            PhysNodeKind::Zip {
                inputs,
                null_extend: true,
            }
        },

        _ => todo!(),
    };

    let output_schema = IR::schema_with_cache(node, ir_arena, schema_cache);
    Ok(phys_sm.insert(PhysNode::new(output_schema, node_kind)))
}
