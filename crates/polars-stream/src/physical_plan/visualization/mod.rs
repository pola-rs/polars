use std::collections::VecDeque;

use polars_core::prelude::SortMultipleOptions;
use polars_ops::frame::{JoinArgs, JoinType};
use polars_plan::dsl::{JoinTypeOptionsIR, PartitionVariantIR, SinkOptions, SinkTarget};
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::prelude::AExpr;
use polars_utils::arena::Arena;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

pub mod models;
pub use models::{PhysNodeInfo, PhysicalPlanVisualizationData};
use slotmap::SlotMap;

use crate::physical_plan::visualization::models::{Edge, PhysNodeProperties};
use crate::physical_plan::{PhysNode, PhysNodeKey, PhysNodeKind};

pub fn generate_visualization_data(
    title: PlSmallStr,
    roots: &[PhysNodeKey],
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
) -> PhysicalPlanVisualizationData {
    let (nodes_list, edges) = PhysicalPlanVisualizationDataGenerator {
        phys_sm,
        expr_arena,
        queue: VecDeque::from_iter(roots.iter().copied()),
        nodes_list: vec![],
        edges: vec![],
    }
    .generate();

    PhysicalPlanVisualizationData {
        title,
        num_roots: roots.len().try_into().unwrap(),
        nodes: nodes_list,
        edges,
    }
}

struct PhysicalPlanVisualizationDataGenerator<'a> {
    phys_sm: &'a SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &'a Arena<AExpr>,
    queue: VecDeque<PhysNodeKey>,
    nodes_list: Vec<PhysNodeInfo>,
    edges: Vec<Edge>,
}

impl PhysicalPlanVisualizationDataGenerator<'_> {
    fn generate(mut self) -> (Vec<PhysNodeInfo>, Vec<Edge>) {
        let mut node_inputs: Vec<PhysNodeKey> = vec![];

        while let Some(key) = self.queue.pop_front() {
            let node: &PhysNode = self.phys_sm.get(key).unwrap();
            let mut phys_node_info = self.get_phys_node_info(node, &mut node_inputs);
            let current_node_key: u64 = key.0.as_ffi();
            phys_node_info.id = current_node_key;

            for input_node in node_inputs.drain(..) {
                let input_node_key = input_node.0.as_ffi();

                self.queue.push_back(input_node);
                self.edges.push(Edge::new(current_node_key, input_node_key));
            }

            self.nodes_list.push(phys_node_info);
        }

        assert!(self.queue.is_empty());
        (self.nodes_list, self.edges)
    }

    fn get_phys_node_info(
        &self,
        phys_node: &PhysNode,
        phys_node_inputs: &mut Vec<PhysNodeKey>,
    ) -> PhysNodeInfo {
        match phys_node.kind() {
            PhysNodeKind::CallbackSink {
                input,
                function,
                maintain_order,
                chunk_size,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::CallbackSink {
                    callback_function: format_pl_smallstr!("{:?}", function),
                    maintain_order: *maintain_order,
                    chunk_size: *chunk_size,
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::DynamicSlice {
                input,
                offset,
                length,
            } => {
                phys_node_inputs.push(input.node);
                phys_node_inputs.push(offset.node);
                phys_node_inputs.push(length.node);

                let properties = PhysNodeProperties::DynamicSlice;

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::FileSink {
                target,
                sink_options:
                    SinkOptions {
                        sync_on_close,
                        maintain_order,
                        mkdir,
                    },
                file_type,
                input,
                cloud_options: _,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::FileSink {
                    target: match target {
                        SinkTarget::Path(p) => format_pl_smallstr!("Path({})", p.to_str()),
                        SinkTarget::Dyn(_) => PlSmallStr::from_static("DynWriteable"),
                    },
                    sync_on_close: *sync_on_close,
                    maintain_order: *maintain_order,
                    mkdir: *mkdir,
                    file_type: PlSmallStr::from_static(file_type.into()),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Filter { input, predicate } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Filter {
                    predicate: format_pl_smallstr!("{}", predicate.display(self.expr_arena)),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::GatherEvery { input, n, offset } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::GatherEvery {
                    n: (*n).try_into().unwrap(),
                    offset: (*offset).try_into().unwrap(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::GroupBy { input, key, aggs } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::GroupBy {
                    keys: expr_list(key, self.expr_arena),
                    aggs: expr_list(aggs, self.expr_arena),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "dynamic_group_by")]
            PhysNodeKind::RollingGroupBy {
                input,
                index_column,
                period,
                offset,
                closed,
                aggs,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::RollingGroupBy {
                    index_column: index_column.clone(),
                    period: format_pl_smallstr!("{period}"),
                    offset: format_pl_smallstr!("{offset}"),
                    closed_window: PlSmallStr::from_static(closed.into()),
                    aggs: expr_list(aggs, self.expr_arena),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::InMemoryMap {
                input,
                map: _, // dyn DataFrameUdf
                format_str,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::InMemoryMap {
                    format_str: format_str.as_deref().map_or(
                        PlSmallStr::from_static(
                            "error: prepare_visualization was not set during conversion",
                        ),
                        PlSmallStr::from_str,
                    ),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::InMemorySink { input } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::InMemorySink;

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::InMemorySource { df } => {
                let properties = PhysNodeProperties::InMemorySource {
                    n_rows: df.height().try_into().unwrap(),
                    schema_names: df.schema().iter_names_cloned().collect(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::InputIndependentSelect { selectors } => {
                let properties = PhysNodeProperties::InputIndependentSelect {
                    selectors: expr_list(selectors, self.expr_arena),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            // Joins
            PhysNodeKind::CrossJoin {
                input_left,
                input_right,
                args,
            } => {
                phys_node_inputs.push(input_left.node);
                phys_node_inputs.push(input_right.node);

                let JoinArgs {
                    how: _,
                    validation: _,
                    suffix,
                    slice: _,
                    nulls_equal: _,
                    coalesce: _,
                    maintain_order,
                } = args;

                let properties = PhysNodeProperties::CrossJoin {
                    maintain_order: *maintain_order,
                    suffix: suffix.clone(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::EquiJoin {
                input_left,
                input_right,
                left_on,
                right_on,
                args,
            } => {
                phys_node_inputs.push(input_left.node);
                phys_node_inputs.push(input_right.node);

                let JoinArgs {
                    how,
                    validation,
                    suffix,
                    // Lowers to a separate node
                    slice: _,
                    nulls_equal,
                    coalesce,
                    maintain_order,
                } = args;

                let properties = PhysNodeProperties::EquiJoin {
                    how: format_pl_smallstr!("{}", how),
                    left_on: expr_list(left_on, self.expr_arena),
                    right_on: expr_list(right_on, self.expr_arena),
                    nulls_equal: *nulls_equal,
                    coalesce: *coalesce,
                    maintain_order: *maintain_order,
                    validation: *validation,
                    suffix: suffix.clone(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::InMemoryJoin {
                input_left,
                input_right,
                left_on,
                right_on,
                args:
                    JoinArgs {
                        how,
                        validation,
                        suffix,
                        slice,
                        nulls_equal,
                        coalesce,
                        maintain_order,
                    },
                options,
            } => {
                phys_node_inputs.push(input_left.node);
                phys_node_inputs.push(input_right.node);

                let properties = match how {
                    JoinType::AsOf(asof_options) => {
                        use polars_ops::frame::AsOfOptions;

                        #[expect(unused_variables)]
                        let AsOfOptions {
                            strategy,
                            tolerance,
                            tolerance_str,
                            left_by,
                            right_by,
                            allow_eq,
                            check_sortedness,
                        } = asof_options.as_ref();

                        assert_eq!(left_on.len(), 1);
                        assert_eq!(right_on.len(), 1);

                        PhysNodeProperties::InMemoryAsOfJoin {
                            left_on: format_pl_smallstr!("{}", left_on[0].display(self.expr_arena)),
                            right_on: format_pl_smallstr!(
                                "{}",
                                right_on[0].display(self.expr_arena)
                            ),
                            left_by: left_by.clone(),
                            right_by: right_by.clone(),
                            strategy: *strategy,
                            tolerance: tolerance.as_ref().map(|scalar| {
                                [
                                    format_pl_smallstr!("{}", scalar.value()),
                                    format_pl_smallstr!("{:?}", scalar.dtype()),
                                ]
                            }),
                            suffix: suffix.clone(),
                            slice: convert_opt_slice(slice),
                            coalesce: *coalesce,
                            allow_eq: *allow_eq,
                            check_sortedness: *check_sortedness,
                        }
                    },
                    JoinType::IEJoin => {
                        use polars_ops::frame::IEJoinOptions;

                        let Some(JoinTypeOptionsIR::IEJoin(IEJoinOptions {
                            operator1,
                            operator2,
                        })) = options
                        else {
                            unreachable!()
                        };

                        PhysNodeProperties::InMemoryIEJoin {
                            left_on: expr_list(left_on, self.expr_arena),
                            right_on: expr_list(right_on, self.expr_arena),
                            inequality_operators: if let Some(operator2) = operator2 {
                                vec![*operator1, *operator2]
                            } else {
                                vec![*operator1]
                            },
                            suffix: suffix.clone(),
                            slice: convert_opt_slice(slice),
                        }
                    },
                    JoinType::Cross => unreachable!(),
                    _ => PhysNodeProperties::InMemoryJoin {
                        how: format_pl_smallstr!("{}", how),
                        left_on: expr_list(left_on, self.expr_arena),
                        right_on: expr_list(right_on, self.expr_arena),
                        nulls_equal: *nulls_equal,
                        coalesce: *coalesce,
                        maintain_order: *maintain_order,
                        validation: *validation,
                        suffix: suffix.clone(),
                        slice: convert_opt_slice(slice),
                    },
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Map { input, map } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Map {
                    display_str: map.display_str(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::MultiScan {
                scan_sources,
                file_reader_builder,
                cloud_options: _,
                file_projection_builder,
                output_schema: _,
                row_index,
                pre_slice,
                predicate,
                predicate_file_skip_applied: _,
                hive_parts,
                include_file_paths,
                cast_columns_policy: _,
                missing_columns_policy: _,
                forbid_extra_columns: _,
                deletion_files,
                table_statistics,
                file_schema: _,
            } => {
                let properties = PhysNodeProperties::MultiScan {
                    scan_type: file_reader_builder.reader_name().into(),
                    num_sources: scan_sources.len().try_into().unwrap(),
                    first_source: scan_sources
                        .first()
                        .map(|x| x.to_include_path_name().into()),
                    projected_file_columns: file_projection_builder
                        .projected_names()
                        .cloned()
                        .collect(),
                    file_projection_builder_type: PlSmallStr::from_static(
                        file_projection_builder.into(),
                    ),
                    row_index_name: row_index.as_ref().map(|ri| ri.name.clone()),
                    #[allow(clippy::useless_conversion)]
                    row_index_offset: row_index.as_ref().map(|ri| ri.offset.into()),
                    pre_slice: pre_slice.clone().map(|x| {
                        let (offset, len) = <(i128, i128)>::from(x);
                        [offset, len]
                    }),
                    predicate: predicate
                        .as_ref()
                        .map(|e| format_pl_smallstr!("{}", e.display(self.expr_arena))),
                    has_table_statistics: table_statistics.is_some(),
                    include_file_paths: include_file_paths.clone(),
                    deletion_files_type: deletion_files
                        .as_ref()
                        .map(|x| PlSmallStr::from_static(x.into())),
                    hive_columns: hive_parts
                        .as_ref()
                        .map(|x| x.df().schema().iter_names_cloned().collect()),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Multiplexer { input } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Multiplexer;

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::NegativeSlice {
                input,
                offset,
                length,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::NegativeSlice {
                    offset: (*offset).into(),
                    length: (*length).try_into().unwrap(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::OrderedUnion { inputs } => {
                for input in inputs {
                    phys_node_inputs.push(input.node);
                }

                let properties = PhysNodeProperties::OrderedUnion {
                    num_inputs: inputs.len().try_into().unwrap(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::PartitionSink {
                input,
                base_path,
                file_path_cb,
                sink_options:
                    SinkOptions {
                        sync_on_close,
                        maintain_order,
                        mkdir,
                    },
                variant,
                file_type,
                cloud_options: _,
                per_partition_sort_by,
                finish_callback,
            } => {
                phys_node_inputs.push(input.node);

                let (
                    partition_variant_max_size,
                    partition_variant_key_exprs,
                    partition_variant_include_key,
                ) = match variant {
                    PartitionVariantIR::ByKey {
                        key_exprs,
                        include_key,
                    }
                    | PartitionVariantIR::Parted {
                        key_exprs,
                        include_key,
                    } => (
                        None,
                        Some(expr_list(key_exprs, self.expr_arena)),
                        Some(*include_key),
                    ),
                    #[allow(clippy::useless_conversion)]
                    PartitionVariantIR::MaxSize(max_size) => (Some((*max_size).into()), None, None),
                };

                let (
                    per_partition_sort_exprs,
                    per_partition_sort_descending,
                    per_partition_sort_nulls_last,
                ) = per_partition_sort_by
                    .as_ref()
                    .map_or((None, None, None), |x| {
                        let (a, (b, c)): (Vec<_>, (Vec<_>, Vec<_>)) = x
                            .iter()
                            .map(|x| {
                                (
                                    format_pl_smallstr!("{}", x.expr.display(self.expr_arena)),
                                    (x.descending, x.nulls_last),
                                )
                            })
                            .unzip();

                        (Some(a), Some(b), Some(c))
                    });

                let properties = PhysNodeProperties::PartitionSink {
                    base_path: base_path.to_str().into(),
                    file_path_callback: file_path_cb.as_ref().map(|x| x.display_str()),
                    partition_variant: PlSmallStr::from_static(variant.into()),
                    partition_variant_max_size,
                    partition_variant_key_exprs,
                    partition_variant_include_key,
                    file_type: PlSmallStr::from_static(file_type.into()),
                    per_partition_sort_exprs,
                    per_partition_sort_descending,
                    per_partition_sort_nulls_last,
                    finish_callback: finish_callback.as_ref().map(|x| x.display_str()),
                    sync_on_close: *sync_on_close,
                    maintain_order: *maintain_order,
                    mkdir: *mkdir,
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::PeakMinMax { input, is_peak_max } => {
                phys_node_inputs.push(input.node);

                let properties = if *is_peak_max {
                    PhysNodeProperties::PeakMax
                } else {
                    PhysNodeProperties::PeakMin
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Reduce { input, exprs } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Reduce {
                    exprs: expr_list(exprs, self.expr_arena),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Repeat { value, repeats } => {
                phys_node_inputs.push(value.node);
                phys_node_inputs.push(repeats.node);

                let properties = PhysNodeProperties::Repeat;

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Rle(input) => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Rle;

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::RleId(input) => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::RleId;

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Select {
                input,
                selectors,
                extend_original,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Select {
                    selectors: expr_list(selectors, self.expr_arena),
                    extend_original: *extend_original,
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Shift {
                input,
                offset,
                fill,
            } => {
                phys_node_inputs.push(input.node);
                phys_node_inputs.push(offset.node);

                if let Some(fill) = fill {
                    phys_node_inputs.push(fill.node);
                }

                let properties = PhysNodeProperties::Shift {
                    has_fill: fill.is_some(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::SimpleProjection { input, columns } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::SimpleProjection {
                    columns: columns.clone(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::SinkMultiple { sinks } => {
                for node in sinks {
                    phys_node_inputs.push(*node);
                }

                let properties = PhysNodeProperties::SinkMultiple {
                    num_sinks: sinks.len().try_into().unwrap(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
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
                    },
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Sort {
                    by_exprs: expr_list(by_column, self.expr_arena),
                    slice: convert_opt_slice(slice),
                    descending: descending.clone(),
                    nulls_last: nulls_last.clone(),
                    multithreaded: *multithreaded,
                    maintain_order: *maintain_order,
                    #[allow(clippy::useless_conversion)]
                    limit: limit.map(|x| x.into()),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::StreamingSlice {
                input,
                offset,
                length,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::Slice {
                    offset: (*offset).try_into().unwrap(),
                    length: (*length).try_into().unwrap(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::TopK {
                input,
                k,
                by_column,
                reverse,
                nulls_last,
            } => {
                phys_node_inputs.push(input.node);
                phys_node_inputs.push(k.node);

                let properties = PhysNodeProperties::TopK {
                    by_exprs: expr_list(by_column, self.expr_arena),
                    reverse: reverse.clone(),
                    nulls_last: nulls_last.clone(),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::WithRowIndex {
                input,
                name,
                offset,
            } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::WithRowIndex {
                    name: name.clone(),
                    #[allow(clippy::useless_conversion)]
                    offset: offset.map(|x| x.into()),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            PhysNodeKind::Zip {
                inputs,
                null_extend,
            } => {
                for input in inputs {
                    phys_node_inputs.push(input.node);
                }

                let properties = PhysNodeProperties::Zip {
                    num_inputs: inputs.len().try_into().unwrap(),
                    null_extend: *null_extend,
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "cum_agg")]
            PhysNodeKind::CumAgg { input, kind } => {
                phys_node_inputs.push(input.node);

                let properties = PhysNodeProperties::CumAgg {
                    kind: format_pl_smallstr!("{:?}", kind),
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "ewma")]
            PhysNodeKind::EwmMean { input, options }
            | PhysNodeKind::EwmVar { input, options }
            | PhysNodeKind::EwmStd { input, options } => {
                phys_node_inputs.push(input.node);

                let polars_ops::series::EWMOptions {
                    alpha,
                    adjust,
                    bias,
                    min_periods,
                    ignore_nulls,
                } = options;

                let properties = PhysNodeProperties::Ewm {
                    variant: PlSmallStr::from_static(phys_node.kind().into()),
                    alpha: *alpha,
                    adjust: *adjust,
                    bias: *bias,
                    min_periods: *min_periods,
                    ignore_nulls: *ignore_nulls,
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "semi_anti_join")]
            PhysNodeKind::SemiAntiJoin {
                input_left,
                input_right,
                left_on,
                right_on,
                args,
                output_bool,
            } => {
                phys_node_inputs.push(input_left.node);
                phys_node_inputs.push(input_right.node);

                let properties = PhysNodeProperties::SemiAntiJoin {
                    left_on: expr_list(left_on, self.expr_arena),
                    right_on: expr_list(right_on, self.expr_arena),
                    nulls_equal: args.nulls_equal,
                    output_as_bool: *output_bool,
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "merge_sorted")]
            PhysNodeKind::MergeSorted {
                input_left,
                input_right,
            } => {
                phys_node_inputs.push(input_left.node);
                phys_node_inputs.push(input_right.node);

                let properties = PhysNodeProperties::MergeSorted;

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "python")]
            PhysNodeKind::PythonScan {
                options:
                    polars_plan::plans::PythonOptions {
                        scan_fn: _,
                        schema,
                        output_schema: _,
                        with_columns,
                        python_source,
                        n_rows,
                        predicate,
                        validate_schema,
                        is_pure,
                    },
            } => {
                use polars_plan::plans::PythonPredicate;

                let properties = PhysNodeProperties::PythonScan {
                    scan_source_type: python_source.clone(),
                    n_rows: n_rows.map(|x| x.try_into().unwrap()),
                    projection: with_columns.as_deref().map(list_str_cloned),
                    predicate: match predicate {
                        PythonPredicate::None => None,
                        PythonPredicate::PyArrow(s) => Some(s.into()),
                        PythonPredicate::Polars(p) => {
                            Some(format_pl_smallstr!("{}", p.display(self.expr_arena)))
                        },
                    },
                    schema_names: schema.iter_names_cloned().collect(),
                    is_pure: *is_pure,
                    validate_schema: *validate_schema,
                };

                PhysNodeInfo {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
        }
    }
}

impl PhysNodeProperties {
    fn variant_name(&self) -> PlSmallStr {
        PlSmallStr::from_static(<&'static str>::from(self))
    }
}

fn list_str_cloned<I, T>(iter: I) -> Vec<PlSmallStr>
where
    I: IntoIterator<Item = T>,
    T: AsRef<str>,
{
    iter.into_iter()
        .map(|x| PlSmallStr::from_str(x.as_ref()))
        .collect()
}

fn convert_opt_slice<T, U>(slice: &Option<(T, U)>) -> Option<[i128; 2]>
where
    T: Copy + TryInto<i128>,
    U: Copy + TryInto<i128>,
    <T as TryInto<i128>>::Error: std::fmt::Debug,
    <U as TryInto<i128>>::Error: std::fmt::Debug,
{
    slice.map(|(offset, len)| [offset.try_into().unwrap(), len.try_into().unwrap()])
}

fn expr_list(exprs: &[ExprIR], expr_arena: &Arena<AExpr>) -> Vec<PlSmallStr> {
    exprs
        .iter()
        .map(|e| format_pl_smallstr!("{}", e.display(expr_arena)))
        .collect()
}
