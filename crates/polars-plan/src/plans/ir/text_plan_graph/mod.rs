use std::collections::VecDeque;

use polars_core::prelude::{PlHashMap, SortMultipleOptions};
use polars_ops::frame::{JoinArgs, JoinType};
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;

use crate::dsl::{
    GroupbyOptions, HConcatOptions, JoinOptionsIR, JoinTypeOptionsIR, UnifiedScanArgs, UnionOptions,
};
use crate::plans::text_plan_graph::models::{Edge, IRTpgProperties};
use crate::plans::{AExpr, ExprIR, FileInfo, IR};
use crate::prelude::{DistinctOptionsIR, ProjectionOptions};

pub mod models;
use models::{TextPlanGraph, TpgNode};

pub fn generate_text_plan_graph<'a>(
    title: PlSmallStr,
    roots: &[Node],
    ir_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
) -> TextPlanGraph {
    let (nodes_list, edges) = TextPlanGraphGenerator {
        ir_arena,
        expr_arena,
        queue: VecDeque::from_iter(roots.iter().copied()),
        nodes_list: vec![],
        edges: vec![],
        cache_node_to_position: Default::default(),
    }
    .generate();

    TextPlanGraph {
        title,
        roots: roots.len(),
        nodes: nodes_list,
        edges,
        legend: None,
    }
}

struct TextPlanGraphGenerator<'a> {
    ir_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
    queue: VecDeque<Node>,
    nodes_list: Vec<TpgNode>,
    edges: Vec<Edge>,
    cache_node_to_position: PlHashMap<UniqueId, usize>,
}

impl TextPlanGraphGenerator<'_> {
    fn generate(mut self) -> (Vec<TpgNode>, Vec<Edge>) {
        // Note, intentionally a queue to traverse in insertion order, the `cache_node_to_position`
        // mapping assumes this.
        while let Some(node) = self.queue.pop_front() {
            let ir = self.ir_arena.get(node);
            let mut tpg_node = self.get_tpg_node(ir);
            let current_node_position: u64 = self.nodes_list.len().try_into().unwrap();
            tpg_node.id = current_node_position;

            for input_node in ir.inputs() {
                // +1 is for the current node we haven't inserted yet.
                let input_node_position = 1 + self.nodes_list.len() + self.queue.len();

                if let IR::Cache { id, input: _ } = self.ir_arena.get(input_node) {
                    if let Some(cache_node_position) = self.cache_node_to_position.get(id) {
                        self.edges
                            .push(Edge::new(current_node_position, *cache_node_position));
                        continue;
                    }

                    self.cache_node_to_position.insert(*id, input_node_position);
                }

                self.queue.push_back(input_node);
                self.edges
                    .push(Edge::new(current_node_position, input_node_position));
            }

            self.nodes_list.push(tpg_node);
        }

        assert!(self.queue.is_empty());
        (self.nodes_list, self.edges)
    }

    fn get_tpg_node(&self, ir: &IR) -> TpgNode {
        match ir {
            IR::Cache { input: _, id } => {
                let properties = IRTpgProperties::Cache { id: *id };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::DataFrameScan {
                df,
                schema,
                output_schema: _,
            } => {
                let properties = IRTpgProperties::DataFrameScan {
                    n_rows: df.height().try_into().unwrap(),
                    schema_names: schema.iter_names_cloned().collect(),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Distinct {
                input: _,
                options:
                    DistinctOptionsIR {
                        subset,
                        maintain_order,
                        keep_strategy,
                        slice,
                    },
            } => {
                let properties = IRTpgProperties::Distinct {
                    subset: subset.as_deref().map(|x| x.to_vec()),
                    maintain_order: *maintain_order,
                    keep_strategy: *keep_strategy,
                    slice: convert_opt_slice(slice),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::ExtContext {
                input: _,
                contexts,
                schema,
            } => {
                let properties = IRTpgProperties::ExtContext {
                    num_contexts: contexts.len().try_into().unwrap(),
                    schema_names: schema.iter_names_cloned().collect(),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Filter {
                input: _,
                predicate,
            } => {
                let properties = IRTpgProperties::Filter {
                    predicate: format_pl_smallstr!("{}", predicate.display(self.expr_arena)),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::GroupBy {
                input: _,
                keys,
                aggs,
                schema: _,
                maintain_order,
                options,
                apply,
            } => {
                let GroupbyOptions {
                    #[cfg(feature = "dynamic_group_by")]
                    dynamic,
                    #[cfg(feature = "dynamic_group_by")]
                    rolling,
                    slice,
                } = options.as_ref();

                let keys = expr_list(keys, self.expr_arena);
                let aggs = expr_list(aggs, self.expr_arena);
                let maintain_order = *maintain_order;
                let plan_callback = apply.as_ref().map(|x| format_pl_smallstr!("{:?}", x));

                let properties = match () {
                    #[cfg(feature = "dynamic_group_by")]
                    _ if dynamic.is_some() => {
                        let Some(polars_time::DynamicGroupOptions {
                            index_column,
                            every,
                            period,
                            offset,
                            label,
                            include_boundaries,
                            closed_window,
                            start_by,
                        }) = dynamic
                        else {
                            unreachable!()
                        };

                        IRTpgProperties::DynamicGroupBy {
                            index_column: index_column.clone(),
                            every: format_pl_smallstr!("{}", every),
                            period: format_pl_smallstr!("{}", period),
                            offset: format_pl_smallstr!("{}", offset),
                            label: *label,
                            include_boundaries: *include_boundaries,
                            closed_window: *closed_window,
                            group_by: keys,
                            start_by: *start_by,
                            plan_callback,
                        }
                    },
                    #[cfg(feature = "dynamic_group_by")]
                    _ if rolling.is_some() => {
                        let Some(polars_time::RollingGroupOptions {
                            index_column,
                            period,
                            offset,
                            closed_window,
                        }) = rolling
                        else {
                            unreachable!()
                        };

                        IRTpgProperties::RollingGroupBy {
                            keys,
                            aggs,
                            index_column: index_column.clone(),
                            period: format_pl_smallstr!("{}", period),
                            offset: format_pl_smallstr!("{}", offset),
                            closed_window: *closed_window,
                            slice: convert_opt_slice(slice),
                            plan_callback,
                        }
                    },
                    _ => IRTpgProperties::GroupBy {
                        keys,
                        aggs,
                        maintain_order,
                        slice: convert_opt_slice(slice),
                        plan_callback,
                    },
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::HConcat {
                inputs,
                schema,
                options: HConcatOptions { parallel },
            } => {
                let properties = IRTpgProperties::HConcat {
                    num_inputs: inputs.len().try_into().unwrap(),
                    schema_names: schema.iter_names_cloned().collect(),
                    parallel: *parallel,
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::HStack {
                input: _,
                exprs,
                schema: _,
                options:
                    ProjectionOptions {
                        run_parallel,
                        duplicate_check,
                        should_broadcast,
                    },
            } => {
                let properties = IRTpgProperties::HStack {
                    exprs: expr_list(exprs, self.expr_arena),
                    run_parallel: *run_parallel,
                    duplicate_check: *duplicate_check,
                    should_broadcast: *should_broadcast,
                };
                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Invalid => {
                let properties = IRTpgProperties::Invalid;

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
                let JoinOptionsIR {
                    allow_parallel,
                    force_parallel,
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
                    rows_left: _,
                    rows_right: _,
                } = options.as_ref();

                let properties = match how {
                    #[cfg(feature = "asof_join")]
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

                        IRTpgProperties::AsOfJoin {
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
                    JoinType::Cross => {
                        let predicate: Option<PlSmallStr> = options.as_ref().map(|x| {
                            let JoinTypeOptionsIR::CrossAndFilter { predicate } = x else {
                                panic!("{x:?}")
                            };

                            format_pl_smallstr!("{}", predicate.display(self.expr_arena))
                        });

                        IRTpgProperties::CrossJoin {
                            maintain_order: *maintain_order,
                            slice: convert_opt_slice(slice),
                            predicate,
                            suffix: suffix.clone(),
                        }
                    },
                    _ => IRTpgProperties::Join {
                        how: format_pl_smallstr!("{}", how),
                        left_on: expr_list(left_on, self.expr_arena),
                        right_on: expr_list(right_on, self.expr_arena),
                        nulls_equal: *nulls_equal,
                        coalesce: *coalesce,
                        maintain_order: *maintain_order,
                        validation: *validation,
                        suffix: suffix.clone(),
                        slice: convert_opt_slice(slice),
                        allow_parallel: *allow_parallel,
                        force_parallel: *force_parallel,
                    },
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::MapFunction { input: _, function } => {
                let properties = IRTpgProperties::MapFunction {
                    function: format_pl_smallstr!("{}", function),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Scan {
                sources,
                file_info:
                    file_info @ FileInfo {
                        schema: _,
                        reader_schema: _,
                        row_estimation: _,
                    },
                predicate,
                scan_type,
                unified_scan_args,
                hive_parts,
                output_schema: _,
            } => {
                let UnifiedScanArgs {
                    schema: _,
                    cloud_options: _,
                    hive_options: _,
                    rechunk,
                    cache: _,
                    glob: _,
                    hidden_file_prefix: _,
                    projection,
                    column_mapping,
                    default_values,
                    row_index,
                    pre_slice,
                    cast_columns_policy: _,
                    missing_columns_policy: _,
                    extra_columns_policy: _,
                    include_file_paths,
                    deletion_files,
                    table_statistics,
                } = unified_scan_args.as_ref();

                let file_columns: Option<Vec<PlSmallStr>> =
                    file_info.iter_reader_schema_names().map(|iter| {
                        iter.filter(|&name| {
                            !(row_index.as_ref().is_some_and(|ri| name == &ri.name)
                                || include_file_paths.as_ref().is_some_and(|x| name == x))
                        })
                        .cloned()
                        .collect()
                    });

                let properties = IRTpgProperties::Scan {
                    scan_type: PlSmallStr::from_static(scan_type.as_ref().into()),
                    num_sources: sources.len().try_into().unwrap(),
                    first_source: sources.first().map(|x| x.to_include_path_name().into()),
                    file_columns,
                    projection: projection.as_deref().map(list_str_cloned),
                    row_index_name: row_index.as_ref().map(|ri| ri.name.clone()),
                    row_index_offset: row_index.as_ref().map(|ri| {
                        #[cfg_attr(feature = "bigidx", expect(clippy::useless_conversion))]
                        ri.offset.into()
                    }),
                    pre_slice: pre_slice.clone().map(|x| {
                        let (offset, len) = <(i128, i128)>::from(x);
                        [offset, len]
                    }),
                    predicate: predicate
                        .as_ref()
                        .map(|e| format_pl_smallstr!("{}", e.display(self.expr_arena))),
                    has_table_statistics: table_statistics.is_some(),
                    include_file_paths: include_file_paths.clone(),
                    column_mapping_type: column_mapping
                        .as_ref()
                        .map(|x| PlSmallStr::from_static(x.into())),
                    default_values_type: default_values
                        .as_ref()
                        .map(|x| PlSmallStr::from_static(x.into())),
                    deletion_files_type: deletion_files
                        .as_ref()
                        .map(|x| PlSmallStr::from_static(x.into())),
                    rechunk: *rechunk,
                    hive_columns: hive_parts
                        .as_ref()
                        .map(|x| x.df().schema().iter_names_cloned().collect()),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Select {
                input: _,
                expr,
                schema: _,
                options:
                    ProjectionOptions {
                        run_parallel,
                        duplicate_check,
                        should_broadcast,
                    },
            } => {
                let properties = IRTpgProperties::Select {
                    exprs: expr_list(expr, self.expr_arena),
                    run_parallel: *run_parallel,
                    duplicate_check: *duplicate_check,
                    should_broadcast: *should_broadcast,
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::SimpleProjection { input: _, columns } => {
                let properties = IRTpgProperties::SimpleProjection {
                    columns: columns.iter_names_cloned().collect(),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Sink { input: _, payload } => {
                let properties = IRTpgProperties::Sink {
                    payload: format_pl_smallstr!("{:?}", payload),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::SinkMultiple { inputs } => {
                let properties = IRTpgProperties::SinkMultiple {
                    num_inputs: inputs.len().try_into().unwrap(),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Slice {
                input: _,
                offset,
                len,
            } => {
                #[cfg_attr(feature = "bigidx", expect(clippy::useless_conversion))]
                let properties = IRTpgProperties::Slice {
                    offset: (*offset).into(),
                    len: (*len).into(),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Sort {
                input: _,
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
                let properties = IRTpgProperties::Sort {
                    by_exprs: expr_list(by_column, self.expr_arena),
                    slice: convert_opt_slice(slice),
                    descending: descending.clone(),
                    nulls_last: nulls_last.clone(),
                    multithreaded: *multithreaded,
                    maintain_order: *maintain_order,
                    #[cfg_attr(feature = "bigidx", expect(clippy::useless_conversion))]
                    limit: limit.map(|x| x.into()),
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            IR::Union {
                inputs: _,
                options:
                    UnionOptions {
                        slice,
                        rows: _,
                        parallel,
                        from_partitioned_ds,
                        flattened_by_opt,
                        rechunk,
                        maintain_order,
                    },
            } => {
                let properties = IRTpgProperties::Union {
                    maintain_order: *maintain_order,
                    parallel: *parallel,
                    rechunk: *rechunk,
                    slice: convert_opt_slice(slice),
                    from_partitioned_ds: *from_partitioned_ds,
                    flattened_by_opt: *flattened_by_opt,
                };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left: _,
                input_right: _,
                key,
            } => {
                let properties = IRTpgProperties::MergeSorted { key: key.clone() };

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
            #[cfg(feature = "python")]
            IR::PythonScan {
                options:
                    crate::plans::PythonOptions {
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
                use crate::plans::PythonPredicate;

                let properties = IRTpgProperties::PythonScan {
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

                TpgNode {
                    title: properties.variant_name(),
                    properties,
                    ..Default::default()
                }
            },
        }
    }
}

impl IRTpgProperties {
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
