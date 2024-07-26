use std::sync::Arc;

use polars_error::PolarsResult;
use polars_expr::reduce::can_convert_into_reduction;
use polars_plan::plans::{AExpr, Context, IR};
use polars_plan::prelude::SinkType;
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey};

fn is_streamable(node: Node, arena: &Arena<AExpr>) -> bool {
    polars_plan::plans::is_streamable(node, arena, Context::Default)
}

#[recursive::recursive]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PolarsResult<PhysNodeKey> {
    let ir_node = ir_arena.get(node);
    match ir_node {
        IR::SimpleProjection { input, columns } => {
            let input_ir_node = ir_arena.get(*input);
            let input_schema = input_ir_node.schema(ir_arena).into_owned();
            let columns = columns.iter_names().map(|s| s.to_string()).collect();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::SimpleProjection {
                input,
                columns,
                input_schema,
            }))
        },

        // TODO: split partially streamable selections to avoid fallback as much as possible.
        IR::Select {
            input,
            expr,
            schema,
            ..
        } if expr.iter().all(|e| is_streamable(e.node(), expr_arena)) => {
            let selectors = expr.clone();
            let output_schema = schema.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::Select {
                input,
                selectors,
                output_schema,
                extend_original: false,
            }))
        },
        // TODO: split reductions and streamable selections. E.g. sum(a) + sum(b) should be split
        // into Select(a + b) -> Reduce(sum(a), sum(b)
        IR::Select {
            input,
            expr,
            schema: output_schema,
            ..
        } if expr
            .iter()
            .all(|e| can_convert_into_reduction(e.node(), expr_arena)) =>
        {
            let exprs = expr.clone();
            let input_ir_node = ir_arena.get(*input);
            let input_schema = input_ir_node.schema(ir_arena).into_owned();
            let output_schema = output_schema.clone();
            let input_node = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::Reduce {
                input: input_node,
                exprs,
                input_schema,
                output_schema,
            }))
        },

        // TODO: split partially streamable selections to avoid fallback as much as possible.
        IR::HStack {
            input,
            exprs,
            schema,
            ..
        } if exprs.iter().all(|e| is_streamable(e.node(), expr_arena)) => {
            let selectors = exprs.clone();
            let output_schema = schema.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::Select {
                input,
                selectors,
                output_schema,
                extend_original: true,
            }))
        },

        IR::Slice { input, offset, len } => {
            if *offset >= 0 {
                let offset = *offset as usize;
                let length = *len as usize;
                let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
                Ok(phys_sm.insert(PhysNode::StreamingSlice {
                    input,
                    offset,
                    length,
                }))
            } else {
                todo!()
            }
        },

        IR::Filter { input, predicate } if is_streamable(predicate.node(), expr_arena) => {
            let predicate = predicate.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::Filter { input, predicate }))
        },

        IR::DataFrameScan {
            df,
            output_schema,
            filter,
            schema: input_schema,
            ..
        } => {
            if let Some(filter) = filter {
                if !is_streamable(filter.node(), expr_arena) {
                    todo!()
                }
            }

            let mut phys_node = phys_sm.insert(PhysNode::InMemorySource { df: df.clone() });

            if let Some(schema) = output_schema {
                phys_node = phys_sm.insert(PhysNode::SimpleProjection {
                    input: phys_node,
                    input_schema: input_schema.clone(),
                    columns: schema.iter_names().map(|s| s.to_string()).collect(),
                })
            }

            if let Some(predicate) = filter.clone() {
                phys_node = phys_sm.insert(PhysNode::Filter {
                    input: phys_node,
                    predicate,
                })
            }

            Ok(phys_node)
        },

        IR::Sink { input, payload } => {
            if *payload == SinkType::Memory {
                let schema = ir_node.schema(ir_arena).into_owned();
                let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
                return Ok(phys_sm.insert(PhysNode::InMemorySink { input, schema }));
            }

            todo!()
        },

        IR::MapFunction { input, function } => {
            let input_schema = ir_arena.get(*input).schema(ir_arena).into_owned();
            let function = function.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;

            let phys_node = if function.is_streamable() {
                let map = Arc::new(move |df| function.evaluate(df));
                PhysNode::Map { input, map }
            } else {
                let map = Arc::new(move |df| function.evaluate(df));
                PhysNode::InMemoryMap {
                    input,
                    input_schema,
                    map,
                }
            };

            Ok(phys_sm.insert(phys_node))
        },

        IR::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input_schema = ir_arena.get(*input).schema(ir_arena).into_owned();
            let phys_node = PhysNode::Sort {
                input_schema,
                by_column: by_column.clone(),
                slice: *slice,
                sort_options: sort_options.clone(),
                input: lower_ir(*input, ir_arena, expr_arena, phys_sm)?,
            };
            Ok(phys_sm.insert(phys_node))
        },

        IR::Union { inputs, options } => {
            if options.slice.is_some() {
                todo!()
            }

            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir(input, ir_arena, expr_arena, phys_sm))
                .collect::<Result<_, _>>()?;
            Ok(phys_sm.insert(PhysNode::OrderedUnion { inputs }))
        },

        IR::HConcat {
            inputs,
            schema: _,
            options: _,
        } => {
            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir(input, ir_arena, expr_arena, phys_sm))
                .collect::<Result<_, _>>()?;
            Ok(phys_sm.insert(PhysNode::Zip { inputs }))
        },

        _ => todo!(),
    }
}
