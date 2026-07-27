use std::collections::VecDeque;

use polars_core::prelude::SortMultipleOptions;
use polars_descriptions::{
    IrNodeDescription, IrPropsDescription, PredicateFileSkipDescription,
    PythonPredicateDescription, SinkDestDescription, SortColumnDescription,
};
use polars_ops::frame::JoinType;
use polars_time::{DynamicGroupOptions, RollingGroupOptions};
use polars_utils::aliases::{InitHashMaps, PlIndexSet};
use polars_utils::arena::{Arena, Node};
use polars_utils::index::idxsize_to_u64;

use crate::dsl::{
    GroupbyOptions, HConcatOptions, JoinTypeOptionsIR, SinkTypeIR, UnifiedScanArgs, UnionOptions,
};
use crate::plans::{AExpr, ArrowPredicate, ExprIR, IR, PythonOptions, PythonPredicate};
use crate::prelude::{DistinctOptionsIR, ProjectionOptions};

pub fn ir_plan_to_description(
    roots: &[Node],
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Vec<IrNodeDescription> {
    let mut nodes = Vec::new();
    let mut queue: VecDeque<Node> = VecDeque::new();
    let mut visited: PlIndexSet<Node> = PlIndexSet::new();

    for root in roots.iter().copied() {
        if visited.insert(root) {
            queue.push_back(root);
        }
    }

    while let Some(key) = queue.pop_front() {
        let ir = ir_arena.get(key);
        let inputs: Vec<Node> = ir.inputs().collect();
        let properties = ir_props(ir, expr_arena);
        let node = IrNodeDescription {
            id: key.0,
            input_ids: inputs.iter().map(|n| n.0).collect(),
            properties,
        };

        for input in inputs {
            if visited.insert(input) {
                queue.push_back(input);
            }
        }

        nodes.push(node);
    }

    nodes
}

pub fn ir_props(ir: &IR, expr_arena: &Arena<AExpr>) -> IrPropsDescription {
    match ir {
        IR::Cache { id, .. } => IrPropsDescription::Cache { id: id.to_string() },
        IR::DataFrameScan { df, schema, .. } => IrPropsDescription::DataFrameScan {
            n_rows: df.height(),
            schema_names: schema.iter_names().map(ToString::to_string).collect(),
        },
        IR::Distinct {
            options:
                DistinctOptionsIR {
                    subset,
                    maintain_order,
                    keep_strategy,
                    slice,
                    ..
                },
            ..
        } => IrPropsDescription::Distinct {
            subset: subset
                .as_deref()
                .map(|x| x.iter().map(ToString::to_string).collect()),
            maintain_order: *maintain_order,
            keep_strategy: format!("{:?}", keep_strategy),
            slice: *slice,
        },
        IR::ExtContext {
            contexts, schema, ..
        } => IrPropsDescription::ExtContext {
            num_contexts: contexts.len(),
            schema_names: schema.iter_names().map(ToString::to_string).collect(),
        },
        IR::Filter { predicate, .. } => IrPropsDescription::Filter {
            predicate: fmt_predicate(predicate, expr_arena),
        },
        IR::GroupBy {
            keys,
            aggs,
            maintain_order,
            options,
            ..
        } => {
            let GroupbyOptions {
                dynamic,
                rolling,
                slice,
            } = options.as_ref();

            let keys = fmt_exprs(keys, expr_arena);
            let aggs = fmt_exprs(aggs, expr_arena);

            if let Some(DynamicGroupOptions {
                index_column,
                every,
                period,
                offset,
                label,
                include_boundaries,
                closed_window,
                start_by,
            }) = dynamic
            {
                IrPropsDescription::DynamicGroupBy {
                    index_column: index_column.to_string(),
                    aggs,
                    every: every.to_string(),
                    period: period.to_string(),
                    offset: offset.to_string(),
                    label: format!("{:?}", label),
                    include_boundaries: *include_boundaries,
                    closed_window: format!("{:?}", closed_window),
                    group_by: keys,
                    start_by: format!("{:?}", start_by),
                }
            } else if let Some(RollingGroupOptions {
                index_column,
                period,
                offset,
                closed_window,
            }) = rolling
            {
                IrPropsDescription::RollingGroupBy {
                    keys,
                    aggs,
                    index_column: index_column.to_string(),
                    period: period.to_string(),
                    offset: offset.to_string(),
                    closed_window: format!("{:?}", closed_window),
                    slice: *slice,
                }
            } else {
                IrPropsDescription::GroupBy {
                    keys,
                    aggs,
                    maintain_order: *maintain_order,
                    slice: *slice,
                }
            }
        },
        IR::HConcat {
            inputs,
            schema,
            options: HConcatOptions { strict, .. },
            ..
        } => IrPropsDescription::HConcat {
            num_inputs: inputs.len(),
            schema_names: schema.iter_names().map(ToString::to_string).collect(),
            strict: *strict,
        },
        IR::HStack {
            exprs,
            options: ProjectionOptions {
                should_broadcast, ..
            },
            ..
        } => IrPropsDescription::HStack {
            exprs: fmt_exprs(exprs, expr_arena),
            should_broadcast: *should_broadcast,
        },
        IR::Invalid => IrPropsDescription::Invalid,
        IR::Join {
            left_on,
            right_on,
            options,
            ..
        } => {
            let o = options.as_ref();
            let args = &o.args;

            let generic_join = || IrPropsDescription::Join {
                how: args.how.to_string(),
                left_on: fmt_exprs(left_on, expr_arena),
                right_on: fmt_exprs(right_on, expr_arena),
                nulls_equal: args.nulls_equal,
                coalesce: fmt_from_static_str(args.coalesce),
                maintain_order: fmt_from_static_str(args.maintain_order),
                validation: fmt_from_static_str(args.validation),
                suffix: args.suffix.as_ref().map(ToString::to_string),
                slice: args.slice,
            };

            match &args.how {
                JoinType::Cross => IrPropsDescription::CrossJoin {
                    maintain_order: fmt_from_static_str(args.maintain_order),
                    slice: args.slice,
                    predicate: o.options.as_ref().and_then(|x| match x {
                        JoinTypeOptionsIR::CrossAndFilter { predicate } => {
                            Some(fmt_predicate(predicate, expr_arena))
                        },
                        _ => None,
                    }),
                    suffix: args.suffix.as_ref().map(ToString::to_string),
                },
                #[cfg(feature = "asof_join")]
                JoinType::AsOf(asof_options) => {
                    use polars_ops::prelude::AsOfOptions;

                    let AsOfOptions {
                        strategy,
                        tolerance,
                        left_by,
                        right_by,
                        allow_eq,
                        check_sortedness,
                        ..
                    } = asof_options.as_ref();

                    IrPropsDescription::AsOfJoin {
                        left_on: fmt_exprs(left_on, expr_arena),
                        right_on: fmt_exprs(right_on, expr_arena),
                        left_by: left_by
                            .as_ref()
                            .map(|v| v.iter().map(ToString::to_string).collect()),
                        right_by: right_by
                            .as_ref()
                            .map(|v| v.iter().map(ToString::to_string).collect()),
                        strategy: fmt_from_static_str(strategy),
                        tolerance: tolerance.as_ref().map(|scalar| {
                            [
                                scalar.value().to_string(),
                                fmt_from_static_str(scalar.dtype()),
                            ]
                        }),
                        suffix: args.suffix.as_ref().map(ToString::to_string),
                        slice: args.slice,
                        coalesce: fmt_from_static_str(args.coalesce),
                        allow_eq: *allow_eq,
                        check_sortedness: *check_sortedness,
                    }
                },
                #[cfg(feature = "iejoin")]
                JoinType::IEJoin => match &o.options {
                    Some(JoinTypeOptionsIR::IEJoin(polars_ops::frame::IEJoinOptions {
                        operator1,
                        operator2,
                    })) => IrPropsDescription::IEJoin {
                        left_on: fmt_exprs(left_on, expr_arena),
                        right_on: fmt_exprs(right_on, expr_arena),
                        inequality_operators: if let Some(operator2) = operator2 {
                            vec![
                                fmt_from_static_str(operator1),
                                fmt_from_static_str(operator2),
                            ]
                        } else {
                            vec![fmt_from_static_str(operator1)]
                        },
                        suffix: args.suffix.as_ref().map(ToString::to_string),
                        slice: args.slice,
                    },
                    _ => generic_join(),
                },
                _ => generic_join(),
            }
        },
        IR::MapFunction { function, .. } => IrPropsDescription::MapFunction {
            function: function.to_string(),
        },
        IR::Scan {
            sources,
            file_info,
            predicate,
            predicate_file_skip_applied,
            scan_type,
            unified_scan_args,
            hive_parts,
            ..
        } => {
            let UnifiedScanArgs {
                projection,
                column_mapping,
                row_index,
                pre_slice,
                include_file_paths,
                table_statistics,
                ..
            } = unified_scan_args.as_ref();

            let file_columns: Option<Vec<String>> =
                file_info.iter_reader_schema_names().map(|iter| {
                    iter.filter(|&name| {
                        !(row_index.as_ref().is_some_and(|ri| name == &ri.name)
                            || include_file_paths.as_ref().is_some_and(|x| name == x))
                    })
                    .map(ToString::to_string)
                    .collect()
                });

            IrPropsDescription::Scan {
                scan_type: <&str>::from(scan_type.as_ref()).to_string(),
                num_sources: sources.len(),
                first_source: sources
                    .first()
                    .map(|x| x.to_include_path_name().to_string()),
                file_columns,
                projection: projection
                    .as_deref()
                    .map(|cols| cols.iter().map(|x| x.to_string()).collect()),
                row_index_name: row_index.as_ref().map(|ri| ri.name.to_string()),
                row_index_offset: row_index.as_ref().map(|ri| idxsize_to_u64(ri.offset)),
                pre_slice: pre_slice
                    .as_ref()
                    .map(|x| x.to_signed_offset_len())
                    .map(|x| (x.0, idxsize_to_u64(x.1))),
                predicate: predicate.as_ref().map(|e| fmt_predicate(e, expr_arena)),
                predicate_file_skip_applied: predicate_file_skip_applied.map(|pfs| {
                    PredicateFileSkipDescription {
                        no_residual_predicate: pfs.no_residual_predicate,
                        original_len: pfs.original_len,
                    }
                }),
                has_table_statistics: table_statistics.is_some(),
                include_file_paths: include_file_paths.as_ref().map(ToString::to_string),
                column_mapping_type: column_mapping.as_ref().map(fmt_from_static_str),
                hive_columns: hive_parts.as_ref().map(|x| {
                    x.df()
                        .schema()
                        .iter_names()
                        .map(ToString::to_string)
                        .collect()
                }),
            }
        },
        IR::Select { expr, .. } => IrPropsDescription::Select {
            exprs: fmt_exprs(expr, expr_arena),
        },
        IR::SimpleProjection { columns, .. } => IrPropsDescription::SimpleProjection {
            columns: columns.iter_names().map(ToString::to_string).collect(),
        },
        IR::Sink { payload, .. } => IrPropsDescription::Sink {
            dest: match payload {
                SinkTypeIR::Memory => SinkDestDescription::Memory,
                SinkTypeIR::Callback(_) => SinkDestDescription::Callback,
                SinkTypeIR::File(f) => SinkDestDescription::File {
                    file_format: fmt_from_static_str(&f.file_format),
                    target: f.target.to_display_string(),
                },
                SinkTypeIR::Partitioned(p) => SinkDestDescription::Partitioned {
                    file_format: fmt_from_static_str(&p.file_format),
                    base_path: p.base_path.as_str().to_string(),
                },
            },
        },
        IR::SinkMultiple { inputs, .. } => IrPropsDescription::SinkMultiple {
            num_inputs: inputs.len(),
        },
        IR::Slice { offset, len, .. } => IrPropsDescription::Slice {
            offset: *offset,
            len: idxsize_to_u64(*len),
        },
        IR::Sort {
            by_column,
            slice,
            sort_options:
                SortMultipleOptions {
                    descending,
                    nulls_last,
                    maintain_order,
                    limit,
                    ..
                },
            ..
        } => IrPropsDescription::Sort {
            sort_columns: by_column
                .iter()
                .zip(descending.iter())
                .zip(nulls_last.iter())
                .map(|((expr, &descending), &nulls_last)| SortColumnDescription {
                    expr: expr.display(expr_arena).to_string(),
                    descending,
                    nulls_last,
                })
                .collect(),
            slice: slice.as_ref().map(|(offset, len, dyn_pred)| {
                (*offset, *len, dyn_pred.as_ref().map(|dp| format!("{dp:?}")))
            }),
            maintain_order: *maintain_order,
            limit: limit.map(idxsize_to_u64),
        },
        IR::Union {
            inputs,
            options:
                UnionOptions {
                    slice,
                    maintain_order,
                    ..
                },
            ..
        } => IrPropsDescription::Union {
            num_inputs: inputs.len(),
            maintain_order: *maintain_order,
            slice: *slice,
        },
        IR::PythonScan {
            options:
                PythonOptions {
                    schema,
                    with_columns,
                    python_source,
                    n_rows,
                    predicate,
                    validate_schema,
                    is_pure,
                    ..
                },
            ..
        } => IrPropsDescription::PythonScan {
            scan_source_type: format!("{:?}", python_source),
            n_rows: *n_rows,
            projection: with_columns
                .as_deref()
                .map(|cols| cols.iter().map(|x| x.to_string()).collect()),
            predicate: match predicate {
                PythonPredicate::None => PythonPredicateDescription::None,
                PythonPredicate::PyArrow(ArrowPredicate {
                    predicate,
                    has_residual,
                    ..
                }) => PythonPredicateDescription::PyArrow {
                    predicate: format!("{:?}", predicate),
                    has_residual: *has_residual,
                },
                PythonPredicate::Polars(p) => PythonPredicateDescription::Polars {
                    predicate: p.display(expr_arena).to_string(),
                },
            },
            schema_names: schema.iter_names().map(ToString::to_string).collect(),
            is_pure: *is_pure,
            validate_schema: *validate_schema,
        },
        IR::UnoptimizedDispatch {
            inputs, operation, ..
        } => IrPropsDescription::UnoptimizedDispatch {
            num_inputs: inputs.len(),
            operation: operation.to_string(),
        },
        IR::Gather { null_on_oob, .. } => IrPropsDescription::Gather {
            null_on_oob: *null_on_oob,
        },
        #[cfg(feature = "merge_sorted")]
        IR::MergeSorted {
            key,
            maintain_order,
            ..
        } => IrPropsDescription::MergeSorted {
            keys: key.iter().map(|k| k.to_string()).collect(),
            maintain_order: *maintain_order,
        },
        #[allow(unreachable_patterns)]
        _ => IrPropsDescription::Other,
    }
}

fn fmt_predicate(predicate: &ExprIR, expr_arena: &Arena<AExpr>) -> Vec<String> {
    use crate::plans::{ExprIRDisplay, MintermIter};

    MintermIter::new(predicate.node(), expr_arena)
        .map(|node| ExprIRDisplay::display_node(node, expr_arena).to_string())
        .collect()
}

fn fmt_exprs(exprs: &[ExprIR], expr_arena: &Arena<AExpr>) -> Vec<String> {
    exprs
        .iter()
        .map(|e| e.display(expr_arena).to_string())
        .collect()
}

/// Helper to one-line [`strum_macros::IntoStaticStr`]
fn fmt_from_static_str(x: impl Into<&'static str>) -> String {
    x.into().to_string()
}
