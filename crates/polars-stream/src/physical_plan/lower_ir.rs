use std::sync::Arc;

use polars_core::prelude::{InitHashMaps, PlHashMap, PlIndexMap};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{AExpr, ColumnName, IR};
use polars_plan::prelude::SinkType;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey, PhysNodeKind};
use crate::physical_plan::lower_expr::{is_elementwise, ExprCache};

#[recursive::recursive]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
    expr_cache: &mut ExprCache,
) -> PolarsResult<PhysNodeKey> {
    let ir_node = ir_arena.get(node);
    let output_schema = IR::schema_with_cache(node, ir_arena, schema_cache);
    let node_kind = match ir_node {
        IR::SimpleProjection { input, columns } => {
            let columns = columns.iter_names().map(|s| s.to_string()).collect();
            let phys_input = lower_ir(
                *input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
            )?;
            PhysNodeKind::SimpleProjection {
                input: phys_input,
                columns,
            }
        },

        IR::Select { input, expr, .. } => {
            let selectors = expr.clone();
            let phys_input = lower_ir(
                *input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
            )?;
            return super::lower_expr::build_select_node(
                phys_input, &selectors, expr_arena, phys_sm, expr_cache,
            );
        },

        IR::HStack { input, exprs, .. }
            if exprs
                .iter()
                .all(|e| is_elementwise(e.node(), expr_arena, expr_cache)) =>
        {
            // FIXME: constant literal columns should be broadcasted with hstack.
            let selectors = exprs.clone();
            let phys_input = lower_ir(
                *input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
            )?;
            PhysNodeKind::Select {
                input: phys_input,
                selectors,
                extend_original: true,
            }
        },

        IR::HStack { input, exprs, .. } => {
            // We already handled the all-streamable case above, so things get more complicated.
            // For simplicity we just do a normal select with all the original columns prepended.
            //
            // FIXME: constant literal columns should be broadcasted with hstack.
            let exprs = exprs.clone();
            let phys_input = lower_ir(
                *input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
            )?;
            let input_schema = &phys_sm[phys_input].output_schema;
            let mut selectors = PlIndexMap::with_capacity(input_schema.len() + exprs.len());
            for name in input_schema.iter_names() {
                let col_name: Arc<str> = name.as_str().into();
                let col_expr = expr_arena.add(AExpr::Column(col_name.clone()));
                selectors.insert(
                    name.clone(),
                    ExprIR::new(col_expr, OutputName::ColumnLhs(col_name)),
                );
            }
            for expr in exprs {
                selectors.insert(expr.output_name().into(), expr);
            }
            let selectors = selectors.into_values().collect_vec();
            return super::lower_expr::build_select_node(
                phys_input, &selectors, expr_arena, phys_sm, expr_cache,
            );
        },

        IR::Slice { input, offset, len } => {
            if *offset >= 0 {
                let offset = *offset as usize;
                let length = *len as usize;
                let phys_input = lower_ir(
                    *input,
                    ir_arena,
                    expr_arena,
                    phys_sm,
                    schema_cache,
                    expr_cache,
                )?;
                PhysNodeKind::StreamingSlice {
                    input: phys_input,
                    offset,
                    length,
                }
            } else {
                todo!()
            }
        },

        IR::Filter { input, predicate } => {
            let predicate = predicate.clone();
            let phys_input = lower_ir(
                *input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
            )?;
            let cols_and_predicate = output_schema
                .iter_names()
                .map(|name| {
                    let name: ColumnName = name.as_str().into();
                    ExprIR::new(
                        expr_arena.add(AExpr::Column(name.clone())),
                        OutputName::ColumnLhs(name),
                    )
                })
                .chain([predicate])
                .collect_vec();
            let (trans_input, mut trans_cols_and_predicate) = super::lower_expr::lower_exprs(
                phys_input,
                &cols_and_predicate,
                expr_arena,
                phys_sm,
                expr_cache,
            )?;

            let filter_schema = phys_sm[trans_input].output_schema.clone();
            let filter = PhysNodeKind::Filter {
                input: trans_input,
                predicate: trans_cols_and_predicate.last().unwrap().clone(),
            };

            let post_filter = phys_sm.insert(PhysNode::new(filter_schema, filter));
            trans_cols_and_predicate.pop(); // Remove predicate.
            return super::lower_expr::build_select_node(
                post_filter,
                &trans_cols_and_predicate,
                expr_arena,
                phys_sm,
                expr_cache,
            );
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

            // Do we need to apply a projection?
            if let Some(projection_schema) = projection {
                if projection_schema.len() != schema.len()
                    || projection_schema
                        .iter_names()
                        .zip(schema.iter_names())
                        .any(|(l, r)| l != r)
                {
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
            }

            if let Some(predicate) = filter.clone() {
                if !is_elementwise(predicate.node(), expr_arena, expr_cache) {
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
                let phys_input = lower_ir(
                    *input,
                    ir_arena,
                    expr_arena,
                    phys_sm,
                    schema_cache,
                    expr_cache,
                )?;
                PhysNodeKind::InMemorySink { input: phys_input }
            } else {
                todo!()
            }
        },

        IR::MapFunction { input, function } => {
            let function = function.clone();
            let phys_input = lower_ir(
                *input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
            )?;

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
            input: lower_ir(
                *input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
            )?,
        },

        IR::Union { inputs, options } => {
            if options.slice.is_some() {
                todo!()
            }

            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| {
                    lower_ir(
                        input,
                        ir_arena,
                        expr_arena,
                        phys_sm,
                        schema_cache,
                        expr_cache,
                    )
                })
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
                .map(|input| {
                    lower_ir(
                        input,
                        ir_arena,
                        expr_arena,
                        phys_sm,
                        schema_cache,
                        expr_cache,
                    )
                })
                .collect::<Result<_, _>>()?;
            PhysNodeKind::Zip {
                inputs,
                null_extend: true,
            }
        },

        v @ IR::Scan { .. } => {
            let IR::Scan {
                paths,
                file_info,
                hive_parts,
                output_schema,
                scan_type,
                predicate,
                file_options,
            } = v.clone()
            else {
                unreachable!();
            };

            PhysNodeKind::FileScan {
                paths,
                file_info,
                hive_parts,
                output_schema,
                scan_type,
                predicate,
                file_options,
            }
        },

        _ => todo!(),
    };

    Ok(phys_sm.insert(PhysNode::new(output_schema, node_kind)))
}
