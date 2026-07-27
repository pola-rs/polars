use std::collections::VecDeque;

use polars_core::prelude::SortMultipleOptions;
#[cfg(feature = "iejoin")]
use polars_descriptions::InequalityOperatorDescription;
use polars_descriptions::{
    FileProviderDescription, PhysicalNodeDescription, PhysicalPropsDescription,
    PredicateFileSkipDescription, PythonPredicateDescription, SortColumnDescription,
};
use polars_ops::frame::JoinType;
#[cfg(feature = "iejoin")]
use polars_plan::dsl::JoinTypeOptionsIR;
use polars_plan::dsl::{
    FileSinkOptions, PartitionStrategyIR, PartitionedSinkOptionsIR, UnifiedSinkArgs,
};
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, ArrowPredicate, PythonOptions, PythonPredicate};
use polars_time::DynamicGroupOptions;
use polars_utils::aliases::{InitHashMaps, PlIndexSet};
use polars_utils::arena::Arena;
use polars_utils::index::idxsize_to_u64;
use slotmap::{Key, SlotMap};

use crate::{PhysNode, PhysNodeKey, PhysNodeKind};

pub fn physical_plan_to_description(
    roots: &[PhysNodeKey],
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
) -> Vec<PhysicalNodeDescription> {
    let mut nodes = Vec::new();
    let mut queue: VecDeque<PhysNodeKey> = VecDeque::new();
    let mut visited: PlIndexSet<PhysNodeKey> = PlIndexSet::new();

    for root in roots.iter().copied() {
        if visited.insert(root) {
            queue.push_back(root);
        }
    }

    while let Some(key) = queue.pop_front() {
        let node = &phys_sm[key];
        let kind = node.kind();
        let (properties, inputs) = phys_props(kind, expr_arena);
        let node = PhysicalNodeDescription {
            id: key.data().as_ffi(),
            input_ids: inputs.iter().map(|k| k.data().as_ffi()).collect(),
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

pub fn phys_props(
    kind: &PhysNodeKind,
    expr_arena: &Arena<AExpr>,
) -> (PhysicalPropsDescription, Vec<PhysNodeKey>) {
    match kind {
        PhysNodeKind::InMemorySource { df, .. } => (
            PhysicalPropsDescription::InMemorySource {
                n_rows: df.height(),
                schema_names: df.schema().iter_names().map(ToString::to_string).collect(),
            },
            vec![],
        ),
        PhysNodeKind::Select {
            input,
            selectors,
            extend_original,
            ..
        } => (
            PhysicalPropsDescription::Select {
                selectors: fmt_exprs(selectors, expr_arena),
                extend_original: *extend_original,
            },
            vec![input.node],
        ),
        PhysNodeKind::InputIndependentSelect { selectors, .. } => (
            PhysicalPropsDescription::InputIndependentSelect {
                selectors: fmt_exprs(selectors, expr_arena),
            },
            vec![],
        ),
        PhysNodeKind::WithRowIndex {
            input,
            name,
            offset,
            ..
        } => (
            PhysicalPropsDescription::WithRowIndex {
                name: name.to_string(),
                offset: offset.map(idxsize_to_u64),
            },
            vec![input.node],
        ),
        PhysNodeKind::Reduce { input, exprs, .. } => (
            PhysicalPropsDescription::Reduce {
                exprs: fmt_exprs(exprs, expr_arena),
            },
            vec![input.node],
        ),
        PhysNodeKind::StreamingSlice {
            input,
            offset,
            length,
            ..
        } => (
            PhysicalPropsDescription::Slice {
                offset: *offset as i64,
                length: *length,
            },
            vec![input.node],
        ),
        PhysNodeKind::NegativeSlice {
            input,
            offset,
            length,
            ..
        } => (
            PhysicalPropsDescription::NegativeSlice {
                offset: *offset,
                length: *length,
            },
            vec![input.node],
        ),
        PhysNodeKind::DynamicSlice {
            input,
            offset,
            length,
            ..
        } => (
            PhysicalPropsDescription::DynamicSlice,
            vec![input.node, offset.node, length.node],
        ),
        PhysNodeKind::Shift {
            input,
            offset,
            fill,
            ..
        } => {
            let mut inputs = vec![input.node, offset.node];
            if let Some(fill) = fill {
                inputs.push(fill.node);
            }
            (
                PhysicalPropsDescription::Shift {
                    has_fill: fill.is_some(),
                },
                inputs,
            )
        },
        PhysNodeKind::Filter {
            input, predicate, ..
        } => (
            PhysicalPropsDescription::Filter {
                predicate: predicate.display(expr_arena).to_string(),
            },
            vec![input.node],
        ),
        PhysNodeKind::SimpleProjection { input, columns, .. } => (
            PhysicalPropsDescription::SimpleProjection {
                columns: columns.values().map(ToString::to_string).collect(),
            },
            vec![input.node],
        ),
        PhysNodeKind::InMemorySink { input, .. } => {
            (PhysicalPropsDescription::InMemorySink, vec![input.node])
        },
        PhysNodeKind::CallbackSink {
            input,
            maintain_order,
            chunk_size,
            ..
        } => (
            PhysicalPropsDescription::CallbackSink {
                maintain_order: *maintain_order,
                chunk_size: chunk_size.map(|c| c.get()),
            },
            vec![input.node],
        ),
        PhysNodeKind::FileSink {
            input,
            options:
                FileSinkOptions {
                    target,
                    file_format,
                    unified_sink_args: UnifiedSinkArgs { maintain_order, .. },
                    ..
                },
            ..
        } => (
            PhysicalPropsDescription::FileSink {
                target: target.to_display_string(),
                file_format: fmt_from_static_str(file_format),
                maintain_order: *maintain_order,
            },
            vec![input.node],
        ),
        PhysNodeKind::PartitionedSink {
            input,
            options:
                PartitionedSinkOptionsIR {
                    base_path,
                    file_path_provider,
                    partition_strategy,
                    file_format,
                    unified_sink_args: UnifiedSinkArgs { maintain_order, .. },
                    max_rows_per_file,
                    approximate_bytes_per_file,
                    ..
                },
            ..
        } => {
            use polars_plan::dsl::file_provider::{FileProviderType, HivePathProvider};

            let (partition_key_exprs, include_keys) = match partition_strategy {
                PartitionStrategyIR::Keyed {
                    keys, include_keys, ..
                } => (Some(fmt_exprs(keys, expr_arena)), Some(*include_keys)),
                _ => (None, None),
            };

            (
                PhysicalPropsDescription::PartitionSink {
                    base_path: base_path.to_string(),
                    file_path_provider: match file_path_provider {
                        FileProviderType::Hive(HivePathProvider { extension }) => {
                            FileProviderDescription::Hive {
                                extension: extension.to_string(),
                            }
                        },
                        FileProviderType::Function(_) => FileProviderDescription::Function,
                        FileProviderType::Iceberg(_) => FileProviderDescription::Iceberg,
                    },
                    file_format: fmt_from_static_str(file_format),
                    partition_strategy: fmt_from_static_str(partition_strategy),
                    partition_key_exprs,
                    include_keys,
                    maintain_order: *maintain_order,
                    max_rows_per_file: idxsize_to_u64(*max_rows_per_file),
                    approximate_bytes_per_file: *approximate_bytes_per_file,
                },
                vec![input.node],
            )
        },
        PhysNodeKind::SinkMultiple { sinks, .. } => (
            PhysicalPropsDescription::SinkMultiple {
                num_sinks: sinks.len(),
            },
            sinks.clone(),
        ),
        PhysNodeKind::InMemoryMap {
            input, format_str, ..
        } => (
            PhysicalPropsDescription::InMemoryMap {
                format_str: format_str
                    .as_deref()
                    .unwrap_or("error: prepare_visualization was not set during conversion")
                    .to_string(),
            },
            vec![input.node],
        ),
        PhysNodeKind::Map { input, .. } => (PhysicalPropsDescription::Map, vec![input.node]),
        PhysNodeKind::ColumnarFunction {
            inputs, format_str, ..
        } => (
            PhysicalPropsDescription::ColumnarFunction {
                num_inputs: inputs.len(),
                name: format_str.clone(),
            },
            inputs.iter().map(|s| s.node).collect(),
        ),
        PhysNodeKind::SortedGroupBy {
            input,
            key,
            aggs,
            slice,
            ..
        } => (
            PhysicalPropsDescription::SortedGroupBy {
                key: key.to_string(),
                aggs: fmt_exprs(aggs, expr_arena),
                slice: slice.map(|(o, l)| (idxsize_to_u64(o), idxsize_to_u64(l))),
            },
            vec![input.node],
        ),
        PhysNodeKind::Sort {
            input,
            by_column,
            slice,
            sort_options:
                SortMultipleOptions {
                    descending,
                    nulls_last,
                    multithreaded,
                    maintain_order,
                    limit,
                    ..
                },
            ..
        } => (
            PhysicalPropsDescription::Sort {
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
                slice: *slice,
                multithreaded: *multithreaded,
                maintain_order: *maintain_order,
                limit: limit.map(idxsize_to_u64),
            },
            vec![input.node],
        ),
        PhysNodeKind::TopK {
            input,
            k,
            by_column,
            reverse,
            nulls_last,
            dyn_pred,
            ..
        } => (
            PhysicalPropsDescription::TopK {
                by_exprs: fmt_exprs(by_column, expr_arena),
                reverse: reverse.clone(),
                nulls_last: nulls_last.clone(),
                dyn_pred: dyn_pred.as_ref().map(|dp| format!("{dp:?}")),
            },
            vec![input.node, k.node],
        ),
        PhysNodeKind::Repeat { value, repeats, .. } => (
            PhysicalPropsDescription::Repeat,
            vec![value.node, repeats.node],
        ),
        PhysNodeKind::GatherEvery {
            input, n, offset, ..
        } => (
            PhysicalPropsDescription::GatherEvery {
                n: *n,
                offset: *offset,
            },
            vec![input.node],
        ),
        PhysNodeKind::ForwardFill { input, limit, .. } => (
            PhysicalPropsDescription::ForwardFill {
                limit: limit.map(idxsize_to_u64),
            },
            vec![input.node],
        ),
        PhysNodeKind::BackwardFill { input, limit, .. } => (
            PhysicalPropsDescription::BackwardFill {
                limit: limit.map(idxsize_to_u64),
            },
            vec![input.node],
        ),
        PhysNodeKind::Rle(input, ..) => (PhysicalPropsDescription::Rle, vec![input.node]),
        PhysNodeKind::RleId(input, ..) => (PhysicalPropsDescription::RleId, vec![input.node]),
        PhysNodeKind::SortedUnique { input, keys, .. } => (
            PhysicalPropsDescription::SortedUnique {
                keys: keys.iter().map(ToString::to_string).collect(),
            },
            vec![input.node],
        ),
        PhysNodeKind::PeakMinMax {
            input, is_peak_max, ..
        } => (
            if *is_peak_max {
                PhysicalPropsDescription::PeakMax
            } else {
                PhysicalPropsDescription::PeakMin
            },
            vec![input.node],
        ),
        PhysNodeKind::IsSorted {
            input,
            descending,
            nulls_last,
            output_name,
            ..
        } => (
            PhysicalPropsDescription::IsSorted {
                descending: *descending,
                nulls_last: *nulls_last,
                output_name: output_name.to_string(),
            },
            vec![input.node],
        ),
        PhysNodeKind::OrderedUnion { inputs, .. } => (
            PhysicalPropsDescription::OrderedUnion {
                num_inputs: inputs.len(),
            },
            inputs.iter().map(|s| s.node).collect(),
        ),
        PhysNodeKind::UnorderedUnion { inputs, .. } => (
            PhysicalPropsDescription::UnorderedUnion {
                num_inputs: inputs.len(),
            },
            inputs.iter().map(|s| s.node).collect(),
        ),
        PhysNodeKind::Zip {
            inputs,
            zip_behavior,
            ..
        } => {
            use crate::ZipBehavior;
            (
                PhysicalPropsDescription::Zip {
                    num_inputs: inputs.len(),
                    zip_behavior: match zip_behavior {
                        ZipBehavior::NullExtend => "NullExtend",
                        ZipBehavior::Broadcast => "Broadcast",
                        ZipBehavior::Strict => "Strict",
                    }
                    .to_string(),
                },
                inputs.iter().map(|s| s.node).collect(),
            )
        },
        PhysNodeKind::Multiplexer { input, .. } => {
            (PhysicalPropsDescription::Multiplexer, vec![input.node])
        },
        PhysNodeKind::MultiScan {
            scan_sources,
            file_reader_builder,
            file_projection_builder,
            row_index,
            pre_slice,
            predicate,
            predicate_file_skip_applied,
            hive_parts,
            include_file_paths,
            deletion_files,
            table_statistics,
            ..
        } => {
            let pre_slice = pre_slice.as_ref().map(|x| x.to_signed_offset_len());
            (
                PhysicalPropsDescription::MultiScan {
                    scan_type: file_reader_builder.reader_name().to_string(),
                    num_sources: scan_sources.len(),
                    first_source: scan_sources
                        .first()
                        .map(|x| x.to_include_path_name().to_string()),
                    projected_file_columns: file_projection_builder
                        .projected_names()
                        .map(ToString::to_string)
                        .collect(),
                    row_index_name: row_index.as_ref().map(|ri| ri.name.to_string()),
                    row_index_offset: row_index.as_ref().map(|ri| idxsize_to_u64(ri.offset)),
                    pre_slice: pre_slice.map(|(o, l)| (o, idxsize_to_u64(l))),
                    predicate: predicate
                        .as_ref()
                        .map(|e| e.display(expr_arena).to_string()),
                    predicate_file_skip_applied: predicate_file_skip_applied.map(|pfs| {
                        PredicateFileSkipDescription {
                            no_residual_predicate: pfs.no_residual_predicate,
                            original_len: pfs.original_len,
                        }
                    }),
                    has_table_statistics: table_statistics.is_some(),
                    include_file_paths: include_file_paths.as_ref().map(ToString::to_string),
                    deletion_files_type: deletion_files.as_ref().map(fmt_from_static_str),
                    hive_columns: hive_parts.as_ref().map(|x| {
                        x.df()
                            .schema()
                            .iter_names()
                            .map(ToString::to_string)
                            .collect()
                    }),
                },
                vec![],
            )
        },
        PhysNodeKind::GroupBy {
            inputs,
            key_per_input,
            aggs_per_input,
            ..
        } => (
            PhysicalPropsDescription::GroupBy {
                num_inputs: inputs.len(),
                key_per_input: key_per_input
                    .iter()
                    .map(|k| fmt_exprs(k, expr_arena))
                    .collect(),
                aggs_per_input: aggs_per_input
                    .iter()
                    .map(|a| fmt_exprs(a, expr_arena))
                    .collect(),
            },
            inputs.iter().map(|s| s.node).collect(),
        ),
        PhysNodeKind::EquiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            ..
        } => (
            PhysicalPropsDescription::EquiJoin {
                how: format!("{}", args.how),
                left_on: fmt_exprs(left_on, expr_arena),
                right_on: fmt_exprs(right_on, expr_arena),
                nulls_equal: args.nulls_equal,
                coalesce: fmt_from_static_str(args.coalesce),
                maintain_order: fmt_from_static_str(args.maintain_order),
                validation: fmt_from_static_str(args.validation),
                suffix: args.suffix.as_ref().map(ToString::to_string),
            },
            vec![input_left.node, input_right.node],
        ),
        PhysNodeKind::MergeJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            descending,
            nulls_last,
            keys_row_encoded,
            args,
            ..
        } => (
            PhysicalPropsDescription::MergeJoin {
                how: args.how.to_string(),
                left_on: left_on.iter().map(ToString::to_string).collect(),
                right_on: right_on.iter().map(ToString::to_string).collect(),
                keys_row_encoded: *keys_row_encoded,
                descending: *descending,
                nulls_last: *nulls_last,
                nulls_equal: args.nulls_equal,
                coalesce: fmt_from_static_str(args.coalesce),
                maintain_order: fmt_from_static_str(args.maintain_order),
                validation: fmt_from_static_str(args.validation),
                suffix: args.suffix.as_ref().map(ToString::to_string),
            },
            vec![input_left.node, input_right.node],
        ),
        PhysNodeKind::SemiAntiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            output_bool,
            ..
        } => (
            PhysicalPropsDescription::SemiAntiJoin {
                left_on: fmt_exprs(left_on, expr_arena),
                right_on: fmt_exprs(right_on, expr_arena),
                nulls_equal: args.nulls_equal,
                output_as_bool: *output_bool,
            },
            vec![input_left.node, input_right.node],
        ),
        PhysNodeKind::CrossJoin {
            input_left,
            input_right,
            args,
            ..
        } => (
            PhysicalPropsDescription::CrossJoin {
                maintain_order: fmt_from_static_str(args.maintain_order),
                suffix: args.suffix.as_ref().map(ToString::to_string),
            },
            vec![input_left.node, input_right.node],
        ),
        #[cfg(feature = "asof_join")]
        PhysNodeKind::AsOfJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            ..
        } => {
            let props = match &args.how {
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

                    PhysicalPropsDescription::AsOfJoin {
                        left_on: left_on.to_string(),
                        right_on: right_on.to_string(),
                        left_by: left_by
                            .as_ref()
                            .map(|by| by.iter().map(ToString::to_string).collect()),
                        right_by: right_by
                            .as_ref()
                            .map(|by| by.iter().map(ToString::to_string).collect()),
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
                _ => PhysicalPropsDescription::Other,
            };
            (props, vec![input_left.node, input_right.node])
        },
        #[cfg(feature = "iejoin")]
        PhysNodeKind::RangeJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            descending,
            args,
            ..
        } => (
            PhysicalPropsDescription::RangeJoin {
                left_on: left_on.iter().map(ToString::to_string).collect(),
                right_on: right_on.iter().map(ToString::to_string).collect(),
                suffix: args.suffix.as_ref().map(ToString::to_string),
                slice: args.slice,
                coalesce: fmt_from_static_str(args.coalesce),
                descending: *descending,
            },
            vec![input_left.node, input_right.node],
        ),
        PhysNodeKind::InMemoryJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            #[cfg(feature = "iejoin")]
            options,
            ..
        } => {
            let generic_join = || PhysicalPropsDescription::InMemoryJoin {
                how: format!("{}", args.how),
                left_on: fmt_exprs(left_on, expr_arena),
                right_on: fmt_exprs(right_on, expr_arena),
                nulls_equal: args.nulls_equal,
                coalesce: fmt_from_static_str(args.coalesce),
                maintain_order: fmt_from_static_str(args.maintain_order),
                validation: fmt_from_static_str(args.validation),
                suffix: args.suffix.as_ref().map(ToString::to_string),
                slice: args.slice,
            };

            let props = match &args.how {
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

                    PhysicalPropsDescription::InMemoryAsOfJoin {
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
                JoinType::IEJoin => match options {
                    Some(JoinTypeOptionsIR::IEJoin(polars_ops::frame::IEJoinOptions {
                        operator1,
                        operator2,
                    })) => {
                        use polars_ops::prelude::InequalityOperator;

                        let to_description = |o: &InequalityOperator| match o {
                            InequalityOperator::Lt => InequalityOperatorDescription::Lt,
                            InequalityOperator::LtEq => InequalityOperatorDescription::LtEq,
                            InequalityOperator::Gt => InequalityOperatorDescription::Gt,
                            InequalityOperator::GtEq => InequalityOperatorDescription::GtEq,
                        };

                        PhysicalPropsDescription::InMemoryIEJoin {
                            left_on: fmt_exprs(left_on, expr_arena),
                            right_on: fmt_exprs(right_on, expr_arena),
                            inequality_operators: if let Some(operator2) = operator2 {
                                vec![to_description(operator1), to_description(operator2)]
                            } else {
                                vec![to_description(operator1)]
                            },
                            suffix: args.suffix.as_ref().map(ToString::to_string),
                            slice: args.slice,
                        }
                    },
                    _ => generic_join(),
                },
                _ => generic_join(),
            };
            (props, vec![input_left.node, input_right.node])
        },
        PhysNodeKind::Gather {
            input,
            idxs,
            null_on_oob,
            ..
        } => (
            PhysicalPropsDescription::Gather {
                null_on_oob: *null_on_oob,
            },
            vec![input.node, idxs.node],
        ),
        #[cfg(feature = "dynamic_group_by")]
        PhysNodeKind::DynamicGroupBy {
            input,
            options,
            aggs,
            slice,
            ..
        } => {
            let DynamicGroupOptions {
                index_column,
                every,
                period,
                offset,
                label,
                include_boundaries,
                closed_window,
                start_by,
            } = options;
            (
                PhysicalPropsDescription::DynamicGroupBy {
                    index_column: index_column.to_string(),
                    period: period.to_string(),
                    every: every.to_string(),
                    offset: offset.to_string(),
                    start_by: fmt_from_static_str(start_by),
                    label: fmt_from_static_str(label),
                    include_boundaries: *include_boundaries,
                    closed_window: fmt_from_static_str(closed_window),
                    aggs: fmt_exprs(aggs, expr_arena),
                    slice: slice.map(|(o, l)| (idxsize_to_u64(o), idxsize_to_u64(l))),
                },
                vec![input.node],
            )
        },
        #[cfg(feature = "dynamic_group_by")]
        PhysNodeKind::RollingGroupBy {
            input,
            index_column,
            period,
            offset,
            closed,
            slice,
            aggs,
            ..
        } => (
            PhysicalPropsDescription::RollingGroupBy {
                index_column: index_column.to_string(),
                period: period.to_string(),
                offset: offset.to_string(),
                closed_window: fmt_from_static_str(closed),
                slice: slice.map(|(o, l)| (idxsize_to_u64(o), idxsize_to_u64(l))),
                aggs: fmt_exprs(aggs, expr_arena),
            },
            vec![input.node],
        ),
        #[cfg(feature = "is_first_distinct")]
        PhysNodeKind::IsFirstDistinct { input, columns, .. } => (
            PhysicalPropsDescription::IsFirstDistinct {
                keys: columns.iter().map(ToString::to_string).collect(),
            },
            vec![input.node],
        ),
        #[cfg(feature = "cum_agg")]
        PhysNodeKind::CumAgg { input, kind, .. } => (
            PhysicalPropsDescription::CumAgg {
                kind: fmt_from_static_str(kind),
            },
            vec![input.node],
        ),
        #[cfg(feature = "interpolate")]
        PhysNodeKind::Interpolate { input, method, .. } => (
            PhysicalPropsDescription::Interpolate {
                method: fmt_from_static_str(method),
            },
            vec![input.node],
        ),
        #[cfg(any(
            feature = "dtype-date",
            feature = "dtype-datetime",
            feature = "dtype-time"
        ))]
        PhysNodeKind::StrptimeInfer { input, options, .. } => (
            PhysicalPropsDescription::StrptimeInfer {
                format: options.format.as_ref().map(ToString::to_string),
                strict: options.strict,
                exact: options.exact,
            },
            vec![input.node],
        ),
        #[cfg(feature = "merge_sorted")]
        PhysNodeKind::MergeSorted {
            input_left,
            input_right,
            maintain_order,
            ..
        } => (
            PhysicalPropsDescription::MergeSorted {
                maintain_order: *maintain_order,
            },
            vec![input_left.node, input_right.node],
        ),
        #[cfg(feature = "python")]
        PhysNodeKind::PythonScan {
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
        } => (
            PhysicalPropsDescription::PythonScan {
                scan_source_type: fmt_from_static_str(python_source),
                n_rows: *n_rows,
                projection: with_columns
                    .as_deref()
                    .map(|cols| cols.iter().map(ToString::to_string).collect()),
                predicate: match predicate {
                    PythonPredicate::None => PythonPredicateDescription::None,
                    PythonPredicate::PyArrow(ArrowPredicate {
                        predicate,
                        has_residual,
                        ..
                    }) => PythonPredicateDescription::PyArrow {
                        predicate: predicate.display(expr_arena).to_string(),
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
            vec![],
        ),
        #[cfg(feature = "ewma")]
        PhysNodeKind::EwmMean { input, options, .. } => {
            (ewm_props(options, "EwmMean"), vec![input.node])
        },
        #[cfg(feature = "ewma")]
        PhysNodeKind::EwmVar { input, options, .. } => {
            (ewm_props(options, "EwmVar"), vec![input.node])
        },
        #[cfg(feature = "ewma")]
        PhysNodeKind::EwmStd { input, options, .. } => {
            (ewm_props(options, "EwmStd"), vec![input.node])
        },
        #[allow(unreachable_patterns)]
        _ => (PhysicalPropsDescription::Other, vec![]),
    }
}

#[cfg(feature = "ewma")]
fn ewm_props(options: &polars_ops::series::EWMOptions, variant: &str) -> PhysicalPropsDescription {
    PhysicalPropsDescription::Ewm {
        variant: variant.to_string(),
        alpha: options.alpha,
        adjust: options.adjust,
        bias: options.bias,
        min_periods: options.min_periods,
        ignore_nulls: options.ignore_nulls,
    }
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
