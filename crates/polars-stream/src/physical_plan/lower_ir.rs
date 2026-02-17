use std::sync::Arc;

use parking_lot::Mutex;
use polars_core::frame::{DataFrame, UniqueKeepStrategy};
use polars_core::prelude::{DataType, PlHashMap, PlHashSet};
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_core::{SchemaExtPl, config};
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_ops::frame::JoinType;
use polars_plan::constants::get_literal_name;
use polars_plan::dsl::default_values::DefaultFieldValues;
use polars_plan::dsl::deletion::DeletionFilesList;
use polars_plan::dsl::{CallbackSinkType, ExtraColumnsPolicy, FileScanIR, SinkTypeIR};
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::{
    AExpr, FunctionIR, IR, IRAggExpr, LiteralValue, are_keys_sorted_any, is_sorted,
    write_ir_non_recursive,
};
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "parquet")]
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::row_counter::RowCounter;
use polars_utils::slice_enum::Slice;
use polars_utils::unique_id::UniqueId;
use polars_utils::{IdxSize, format_pl_smallstr, unique_column_name};
use slotmap::SlotMap;

use super::lower_expr::build_hstack_stream;
use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream};
use crate::nodes::io_sources::multi_scan;
use crate::nodes::io_sources::multi_scan::components::forbid_extra_columns::ForbidExtraColumns;
use crate::nodes::io_sources::multi_scan::components::projection::builder::ProjectionBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::physical_plan::ZipBehavior;
use crate::physical_plan::lower_expr::{ExprCache, build_select_stream, lower_exprs};
use crate::physical_plan::lower_group_by::build_group_by_stream;
use crate::utils::late_materialized_df::LateMaterializedDataFrame;

/// Creates a new PhysStream which outputs a slice of the input stream.
pub fn build_slice_stream(
    input: PhysStream,
    offset: i64,
    length: usize,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PhysStream {
    if offset >= 0 {
        let offset = offset as usize;
        PhysStream::first(phys_sm.insert(PhysNode::new(
            phys_sm[input.node].output_schema.clone(),
            PhysNodeKind::StreamingSlice {
                input,
                offset,
                length,
            },
        )))
    } else {
        PhysStream::first(phys_sm.insert(PhysNode::new(
            phys_sm[input.node].output_schema.clone(),
            PhysNodeKind::NegativeSlice {
                input,
                offset,
                length,
            },
        )))
    }
}

/// Creates a new PhysStream which is filters the input stream.
pub(super) fn build_filter_stream(
    input: PhysStream,
    predicate: ExprIR,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<PhysStream> {
    let predicate = predicate;
    let cols_and_predicate = phys_sm[input.node]
        .output_schema
        .iter_names()
        .cloned()
        .map(|name| {
            ExprIR::new(
                expr_arena.add(AExpr::Column(name.clone())),
                OutputName::ColumnLhs(name),
            )
        })
        .chain([predicate])
        .collect_vec();
    let (trans_input, mut trans_cols_and_predicate) = lower_exprs(
        input,
        &cols_and_predicate,
        expr_arena,
        phys_sm,
        expr_cache,
        ctx,
    )?;

    let filter_schema = phys_sm[trans_input.node].output_schema.clone();
    let filter = PhysNodeKind::Filter {
        input: trans_input,
        predicate: trans_cols_and_predicate.last().unwrap().clone(),
    };

    let post_filter = phys_sm.insert(PhysNode::new(filter_schema, filter));
    trans_cols_and_predicate.pop(); // Remove predicate.
    build_select_stream(
        PhysStream::first(post_filter),
        &trans_cols_and_predicate,
        expr_arena,
        phys_sm,
        expr_cache,
        ctx,
    )
}

/// Creates a new PhysStream with row index attached with the given name.
pub fn build_row_idx_stream(
    input: PhysStream,
    name: PlSmallStr,
    offset: Option<IdxSize>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PhysStream {
    let input_schema = &phys_sm[input.node].output_schema;
    let mut output_schema = (**input_schema).clone();
    output_schema
        .insert_at_index(0, name.clone(), DataType::IDX_DTYPE)
        .unwrap();
    let kind = PhysNodeKind::WithRowIndex {
        input,
        name,
        offset,
    };
    let with_row_idx_node_key = phys_sm.insert(PhysNode::new(Arc::new(output_schema), kind));
    PhysStream::first(with_row_idx_node_key)
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingLowerIRContext {
    pub prepare_visualization: bool,
}

#[recursive::recursive]
#[allow(clippy::too_many_arguments)]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
    expr_cache: &mut ExprCache,
    cache_nodes: &mut PlHashMap<UniqueId, PhysStream>,
    ctx: StreamingLowerIRContext,
    mut disable_morsel_split: Option<bool>,
) -> PolarsResult<PhysStream> {
    // Helper macro to simplify recursive calls.
    macro_rules! lower_ir {
        ($input:expr) => {{
            // Disable for remaining execution graph if it wasn't explicitly set
            // by the current IR.
            disable_morsel_split.get_or_insert(false);

            lower_ir(
                $input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
                expr_cache,
                cache_nodes,
                ctx,
                disable_morsel_split,
            )
        }};
    }

    // Require the code below to explicitly set this to `true`
    if disable_morsel_split == Some(true) {
        disable_morsel_split.take();
    }

    let ir_node = ir_arena.get(node);
    let output_schema = IR::schema_with_cache(node, ir_arena, schema_cache);
    let node_kind = match ir_node {
        IR::SimpleProjection { input, columns } => {
            disable_morsel_split.get_or_insert(true);
            let columns = columns.iter_names_cloned().collect::<Vec<_>>();
            let phys_input = lower_ir!(*input)?;
            PhysNodeKind::SimpleProjection {
                input: phys_input,
                columns,
            }
        },

        IR::Select { input, expr, .. } => {
            let selectors = expr.clone();

            if selectors
                .iter()
                .all(|e| matches!(expr_arena.get(e.node()), AExpr::Len | AExpr::Column(_)))
            {
                disable_morsel_split.get_or_insert(true);
            }

            let phys_input = lower_ir!(*input)?;
            return build_select_stream(
                phys_input, &selectors, expr_arena, phys_sm, expr_cache, ctx,
            );
        },

        IR::HStack { input, exprs, .. } => {
            let exprs = exprs.to_vec();
            let phys_input = lower_ir!(*input)?;
            return build_hstack_stream(phys_input, &exprs, expr_arena, phys_sm, expr_cache, ctx);
        },

        IR::Slice { input, offset, len } => {
            let offset = *offset;
            let len = *len as usize;
            let phys_input = lower_ir!(*input)?;
            return Ok(build_slice_stream(phys_input, offset, len, phys_sm));
        },

        IR::Filter { input, predicate } => {
            let predicate = predicate.clone();
            let phys_input = lower_ir!(*input)?;
            return build_filter_stream(
                phys_input, predicate, expr_arena, phys_sm, expr_cache, ctx,
            );
        },

        IR::DataFrameScan {
            df,
            output_schema: projection,
            schema,
            ..
        } => {
            let schema = schema.clone(); // This is initially the schema of df, but can change with the projection.
            let mut node_kind = PhysNodeKind::InMemorySource {
                df: df.clone(),
                disable_morsel_split: disable_morsel_split.unwrap_or(true),
            };

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
                        input: PhysStream::first(phys_input),
                        columns: projection_schema.iter_names_cloned().collect::<Vec<_>>(),
                    };
                }
            }

            node_kind
        },

        IR::Sink { input, payload } => match payload {
            SinkTypeIR::Memory => {
                disable_morsel_split.get_or_insert(true);
                let phys_input = lower_ir!(*input)?;
                PhysNodeKind::InMemorySink { input: phys_input }
            },
            SinkTypeIR::Callback(CallbackSinkType {
                function,
                maintain_order,
                chunk_size,
            }) => {
                let function = function.clone();
                let maintain_order = *maintain_order;
                let chunk_size = *chunk_size;
                let phys_input = lower_ir!(*input)?;
                PhysNodeKind::CallbackSink {
                    input: phys_input,
                    function,
                    maintain_order,
                    chunk_size,
                }
            },

            SinkTypeIR::File(options) => {
                let options = options.clone();
                let input = lower_ir!(*input)?;
                PhysNodeKind::FileSink { input, options }
            },

            SinkTypeIR::Partitioned(options) => {
                let options = options.clone();
                let input = lower_ir!(*input)?;
                PhysNodeKind::PartitionedSink { input, options }
            },
        },

        IR::SinkMultiple { inputs } => {
            disable_morsel_split.get_or_insert(true);
            let mut sinks = Vec::with_capacity(inputs.len());
            for input in inputs.clone() {
                let phys_node_stream = match ir_arena.get(input) {
                    IR::Sink { .. } => lower_ir!(input)?,
                    _ => lower_ir!(ir_arena.add(IR::Sink {
                        input,
                        payload: SinkTypeIR::Memory
                    }))?,
                };
                sinks.push(phys_node_stream.node);
            }
            PhysNodeKind::SinkMultiple { sinks }
        },

        #[cfg(feature = "merge_sorted")]
        IR::MergeSorted {
            input_left,
            input_right,
            key,
        } => {
            let input_left = *input_left;
            let input_right = *input_right;
            let key = key.clone();

            let mut phys_left = lower_ir!(input_left)?;
            let mut phys_right = lower_ir!(input_right)?;

            let left_schema = &phys_sm[phys_left.node].output_schema;
            let right_schema = &phys_sm[phys_right.node].output_schema;

            left_schema.ensure_is_exact_match(right_schema).unwrap();

            let key_dtype = left_schema.try_get(key.as_str())?.clone();

            let key_name = unique_column_name();
            use polars_plan::plans::{AExprBuilder, RowEncodingVariant};

            // Add the key column as the last column for both inputs.
            for s in [&mut phys_left, &mut phys_right] {
                let key_dtype = key_dtype.clone();
                let mut expr = AExprBuilder::col(key.clone(), expr_arena);
                if key_dtype.is_nested() {
                    expr = AExprBuilder::row_encode(
                        vec![expr.expr_ir(key_name.clone())],
                        vec![key_dtype],
                        RowEncodingVariant::Ordered {
                            descending: None,
                            nulls_last: None,
                            broadcast_nulls: None,
                        },
                        expr_arena,
                    );
                }

                *s = build_hstack_stream(
                    *s,
                    &[expr.expr_ir(key_name.clone())],
                    expr_arena,
                    phys_sm,
                    expr_cache,
                    ctx,
                )?;
            }

            PhysNodeKind::MergeSorted {
                input_left: phys_left,
                input_right: phys_right,
            }
        },

        IR::MapFunction { input, function } => {
            let function = function.clone();
            let phys_input = lower_ir!(*input)?;

            match function {
                FunctionIR::RowIndex {
                    name,
                    offset,
                    schema: _,
                } => PhysNodeKind::WithRowIndex {
                    input: phys_input,
                    name,
                    offset,
                },

                function if function.is_streamable() => {
                    let map = Arc::new(move |df| function.evaluate(df));
                    let format_str = ctx.prepare_visualization.then(|| {
                        let mut buffer = String::new();
                        write_ir_non_recursive(
                            &mut buffer,
                            ir_arena.get(node),
                            expr_arena,
                            phys_sm.get(phys_input.node).unwrap().output_schema.as_ref(),
                            0,
                        )
                        .unwrap();
                        buffer
                    });
                    PhysNodeKind::Map {
                        input: phys_input,
                        map,
                        format_str,
                    }
                },

                function => {
                    let format_str = ctx.prepare_visualization.then(|| {
                        let mut buffer = String::new();
                        write_ir_non_recursive(
                            &mut buffer,
                            ir_arena.get(node),
                            expr_arena,
                            phys_sm.get(phys_input.node).unwrap().output_schema.as_ref(),
                            0,
                        )
                        .unwrap();
                        buffer
                    });
                    let map = Arc::new(move |df| function.evaluate(df));
                    PhysNodeKind::InMemoryMap {
                        input: phys_input,
                        map,
                        format_str,
                    }
                },
            }
        },

        IR::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let slice = *slice;
            let mut by_column = by_column.clone();
            let mut sort_options = sort_options.clone();
            let phys_input = lower_ir!(*input)?;

            // See if we can insert a top k.
            let mut limit = u64::MAX;
            if let Some((0, l)) = slice {
                limit = limit.min(l as u64);
            }
            #[allow(clippy::unnecessary_cast)]
            if let Some(l) = sort_options.limit {
                limit = limit.min(l as u64);
            };

            let mut stream = phys_input;
            if limit < u64::MAX {
                // If we need to maintain order augment with row index.
                if sort_options.maintain_order {
                    let row_idx_name = unique_column_name();
                    stream = build_row_idx_stream(stream, row_idx_name.clone(), None, phys_sm);

                    // Add row index to sort columns.
                    let row_idx_node = expr_arena.add(AExpr::Column(row_idx_name.clone()));
                    by_column.push(ExprIR::new(
                        row_idx_node,
                        OutputName::ColumnLhs(row_idx_name),
                    ));
                    sort_options.descending.push(false);
                    sort_options.nulls_last.push(true);

                    // No longer needed for the actual sort itself, handled by row index.
                    sort_options.maintain_order = false;
                }

                let k_node =
                    expr_arena.add(AExpr::Literal(LiteralValue::Scalar(Scalar::from(limit))));
                let k_selector = ExprIR::from_node(k_node, expr_arena);
                let k_output_schema = Schema::from_iter([(get_literal_name(), DataType::UInt64)]);
                let k_node = phys_sm.insert(PhysNode::new(
                    Arc::new(k_output_schema),
                    PhysNodeKind::InputIndependentSelect {
                        selectors: vec![k_selector],
                    },
                ));

                let mut trans_by_column;
                (stream, trans_by_column) =
                    lower_exprs(stream, &by_column, expr_arena, phys_sm, expr_cache, ctx)?;

                trans_by_column = trans_by_column
                    .into_iter()
                    .enumerate()
                    .map(|(i, expr)| expr.with_alias(format_pl_smallstr!("__POLARS_KEYCOL_{}", i)))
                    .collect_vec();

                stream = PhysStream::first(phys_sm.insert(PhysNode {
                    output_schema: phys_sm[stream.node].output_schema.clone(),
                    kind: PhysNodeKind::TopK {
                        input: stream,
                        k: PhysStream::first(k_node),
                        by_column: trans_by_column,
                        reverse: sort_options.descending.iter().map(|x| !x).collect(),
                        nulls_last: sort_options.nulls_last.clone(),
                    },
                }));
            }

            stream = PhysStream::first(phys_sm.insert(PhysNode {
                output_schema: phys_sm[stream.node].output_schema.clone(),
                kind: PhysNodeKind::Sort {
                    input: stream,
                    by_column,
                    slice,
                    sort_options,
                },
            }));

            // Remove any temporary columns we may have added.
            let exprs: Vec<_> = output_schema
                .iter_names()
                .map(|name| {
                    let node = expr_arena.add(AExpr::Column(name.clone()));
                    ExprIR::new(node, OutputName::ColumnLhs(name.clone()))
                })
                .collect();
            stream = build_select_stream(stream, &exprs, expr_arena, phys_sm, expr_cache, ctx)?;

            return Ok(stream);
        },
        IR::Union { inputs, options } => {
            let options = *options;

            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir!(input))
                .collect::<Result<_, _>>()?;

            let kind = if options.maintain_order {
                PhysNodeKind::OrderedUnion { inputs }
            } else {
                PhysNodeKind::UnorderedUnion { inputs }
            };

            let node = phys_sm.insert(PhysNode {
                output_schema,
                kind,
            });
            let mut stream = PhysStream::first(node);

            if let Some((offset, length)) = options.slice {
                stream = build_slice_stream(stream, offset, length, phys_sm);
            }

            return Ok(stream);
        },

        IR::HConcat {
            inputs,
            schema: _,
            options,
        } => {
            let zip_behavior = if options.strict {
                ZipBehavior::Strict
            } else if options.broadcast_unit_length {
                ZipBehavior::Broadcast
            } else {
                ZipBehavior::NullExtend
            };
            let inputs = inputs
                .clone() // Needed to borrow ir_arena mutably.
                .into_iter()
                .map(|input| lower_ir!(input))
                .collect::<Result<_, _>>()?;
            PhysNodeKind::Zip {
                inputs,
                zip_behavior,
            }
        },

        v @ IR::Scan { .. } => {
            let IR::Scan {
                sources: scan_sources,
                file_info,
                mut hive_parts,
                output_schema: _,
                scan_type,
                predicate,
                predicate_file_skip_applied,
                unified_scan_args,
            } = v.clone()
            else {
                unreachable!();
            };

            if (scan_sources.is_empty()
                && !matches!(scan_type.as_ref(), FileScanIR::Anonymous { .. }))
                || unified_scan_args
                    .pre_slice
                    .as_ref()
                    .is_some_and(|slice| slice.len() == 0)
            {
                if config::verbose() {
                    eprintln!("lower_ir: scan IR lowered as empty InMemorySource")
                }

                // If there are no sources, just provide an empty in-memory source with the right
                // schema.
                PhysNodeKind::InMemorySource {
                    df: Arc::new(DataFrame::empty_with_schema(output_schema.as_ref())),
                    disable_morsel_split: disable_morsel_split.unwrap_or(true),
                }
            } else if output_schema.is_empty()
                && let Some((physical_rows, deleted_rows)) = unified_scan_args.row_count
                && unified_scan_args.pre_slice.is_none()
                && predicate.is_none()
            {
                // Fast-count for scan_iceberg will hit here.
                let row_counter = RowCounter::new(physical_rows, deleted_rows);
                row_counter.num_rows_idxsize()?;
                let num_rows = row_counter.num_rows()?;

                if config::verbose() {
                    eprintln!(
                        "lower_ir: scan IR lowered as 0-width InMemorySource with height {} ({:?})",
                        num_rows, &row_counter
                    )
                }

                PhysNodeKind::InMemorySource {
                    df: Arc::new(DataFrame::empty_with_height(num_rows)),
                    disable_morsel_split: disable_morsel_split.unwrap_or(true),
                }
            } else {
                let file_reader_builder: Arc<dyn FileReaderBuilder> = match &*scan_type {
                    #[cfg(feature = "parquet")]
                    FileScanIR::Parquet {
                        options,
                        metadata: first_metadata,
                    } => Arc::new(
                        crate::nodes::io_sources::parquet::builder::ParquetReaderBuilder {
                            options: Arc::new(options.clone()),
                            first_metadata: first_metadata.clone(),
                            prefetch_limit: RelaxedCell::new_usize(0),
                            prefetch_semaphore: std::sync::OnceLock::new(),
                            shared_prefetch_wait_group_slot: Default::default(),
                            io_metrics: std::sync::OnceLock::new(),
                        },
                    ) as _,

                    #[cfg(feature = "ipc")]
                    FileScanIR::Ipc {
                        options,
                        metadata: first_metadata,
                    } => Arc::new(crate::nodes::io_sources::ipc::builder::IpcReaderBuilder {
                        options: Arc::new(options.clone()),
                        first_metadata: first_metadata.clone(),
                        prefetch_limit: RelaxedCell::new_usize(0),
                        prefetch_semaphore: std::sync::OnceLock::new(),
                        shared_prefetch_wait_group_slot: Default::default(),
                        io_metrics: std::sync::OnceLock::new(),
                    }) as _,

                    #[cfg(feature = "csv")]
                    FileScanIR::Csv { options } => Arc::new(Arc::clone(options)) as _,

                    #[cfg(feature = "json")]
                    FileScanIR::NDJson { options } => Arc::new(
                        crate::nodes::io_sources::ndjson::builder::NDJsonReaderBuilder {
                            options: Arc::new(options.clone()),
                            prefetch_limit: RelaxedCell::new_usize(0),
                            prefetch_semaphore: std::sync::OnceLock::new(),
                            shared_prefetch_wait_group_slot: Default::default(),
                            io_metrics: std::sync::OnceLock::new(),
                        },
                    ) as _,
                    // Arc::new(options.clone()) as _,
                    #[cfg(feature = "python")]
                    FileScanIR::PythonDataset {
                        dataset_object: _,
                        cached_ir,
                    } => {
                        use crate::physical_plan::io::python_dataset::python_dataset_scan_to_reader_builder;
                        let guard = cached_ir.lock().unwrap();

                        let expanded_scan = guard
                            .as_ref()
                            .expect("python dataset should be resolved")
                            .python_scan()
                            .expect("should be python scan");

                        python_dataset_scan_to_reader_builder(expanded_scan)
                    },

                    #[cfg(feature = "scan_lines")]
                    FileScanIR::Lines { name: _ } => {
                        Arc::new(crate::nodes::io_sources::lines::LineReaderBuilder {
                            prefetch_limit: RelaxedCell::new_usize(0),
                            prefetch_semaphore: std::sync::OnceLock::new(),
                            shared_prefetch_wait_group_slot: Default::default(),
                            io_metrics: std::sync::OnceLock::new(),
                        }) as _
                    },

                    FileScanIR::Anonymous { .. } => todo!("unimplemented: AnonymousScan"),
                };

                {
                    let cloud_options = unified_scan_args.cloud_options.clone().map(Arc::new);
                    let file_schema = file_info.schema;

                    let (projected_schema, file_schema) =
                        multi_scan::functions::resolve_projections::resolve_projections(
                            &output_schema,
                            &file_schema,
                            &mut hive_parts,
                            unified_scan_args
                                .row_index
                                .as_ref()
                                .map(|ri| ri.name.as_str()),
                            unified_scan_args
                                .include_file_paths
                                .as_ref()
                                .map(|x| x.as_str()),
                        );

                    let file_projection_builder = ProjectionBuilder::new(
                        projected_schema,
                        unified_scan_args.column_mapping.as_ref(),
                        unified_scan_args
                            .default_values
                            .filter(|DefaultFieldValues::Iceberg(v)| !v.is_empty())
                            .map(|DefaultFieldValues::Iceberg(v)| v),
                    );

                    // TODO: We ignore the parameter for some scan types to maintain old behavior,
                    // as they currently don't expose an API for it to be configured.
                    let extra_columns_policy = match &*scan_type {
                        #[cfg(feature = "parquet")]
                        FileScanIR::Parquet { .. } => unified_scan_args.extra_columns_policy,

                        _ => {
                            if unified_scan_args.projection.is_some() {
                                ExtraColumnsPolicy::Ignore
                            } else {
                                ExtraColumnsPolicy::Raise
                            }
                        },
                    };

                    let forbid_extra_columns = ForbidExtraColumns::opt_new(
                        &extra_columns_policy,
                        &file_schema,
                        unified_scan_args.column_mapping.as_ref(),
                    );

                    let pre_slice = unified_scan_args.pre_slice.clone();
                    let disable_morsel_split = disable_morsel_split.unwrap_or(true);

                    let mut multi_scan_node = PhysNodeKind::MultiScan {
                        scan_sources,
                        file_reader_builder,
                        cloud_options,
                        file_projection_builder,
                        output_schema: output_schema.clone(),
                        row_index: None,
                        pre_slice,
                        predicate,
                        predicate_file_skip_applied,
                        hive_parts,
                        cast_columns_policy: unified_scan_args.cast_columns_policy,
                        missing_columns_policy: unified_scan_args.missing_columns_policy,
                        forbid_extra_columns,
                        include_file_paths: unified_scan_args.include_file_paths,
                        // Set to None if empty for performance.
                        deletion_files: DeletionFilesList::filter_empty(
                            unified_scan_args.deletion_files,
                        ),
                        table_statistics: unified_scan_args.table_statistics,
                        file_schema,
                        disable_morsel_split,
                    };

                    let PhysNodeKind::MultiScan {
                        output_schema: multi_scan_output_schema,
                        row_index: row_index_to_multiscan,
                        pre_slice: pre_slice_to_multiscan,
                        predicate: predicate_to_multiscan,
                        ..
                    } = &mut multi_scan_node
                    else {
                        unreachable!()
                    };

                    let mut row_index_post = unified_scan_args.row_index;

                    // * If a predicate was pushed then we always push row index
                    if predicate_to_multiscan.is_some()
                        || matches!(pre_slice_to_multiscan, Some(Slice::Negative { .. }))
                    {
                        *row_index_to_multiscan = row_index_post.take();
                    }

                    // TODO
                    // Projection pushdown could change the row index column position. Ideally it shouldn't,
                    // and instead just put a projection on top of the scan node in the IR. But for now
                    // we do that step here.
                    let mut schema_after_row_index_post = multi_scan_output_schema.clone();
                    let mut reorder_after_row_index_post = false;

                    // Remove row index from multiscan schema if not pushed.
                    if let Some(ri) = row_index_post.as_ref() {
                        let row_index_post_position =
                            multi_scan_output_schema.index_of(&ri.name).unwrap();
                        let (_, dtype) = Arc::make_mut(multi_scan_output_schema)
                            .shift_remove_index(row_index_post_position)
                            .unwrap();

                        if row_index_post_position != 0 {
                            reorder_after_row_index_post = true;
                            let mut schema =
                                Schema::with_capacity(multi_scan_output_schema.len() + 1);
                            schema.extend([(ri.name.clone(), dtype)]);
                            schema.extend(
                                multi_scan_output_schema
                                    .iter()
                                    .map(|(k, v)| (k.clone(), v.clone())),
                            );
                            schema_after_row_index_post = Arc::new(schema);
                        }
                    }

                    // If we have no predicate and no slice or positive slice, we can reorder the row index to after
                    // the slice by adjusting the offset. This can remove a serial synchronization step in multiscan
                    // and allow the reader to still skip rows.
                    let row_index_post_after_slice = (|| {
                        let mut row_index = row_index_post.take()?;

                        let positive_offset = match pre_slice_to_multiscan {
                            Some(Slice::Positive { offset, .. }) => Some(*offset),
                            None => Some(0),
                            Some(Slice::Negative { .. }) => unreachable!(),
                        }?;

                        row_index.offset = row_index.offset.saturating_add(
                            IdxSize::try_from(positive_offset).unwrap_or(IdxSize::MAX),
                        );

                        Some(row_index)
                    })();

                    let mut stream = {
                        let node_key = phys_sm.insert(PhysNode::new(
                            multi_scan_output_schema.clone(),
                            multi_scan_node,
                        ));
                        PhysStream::first(node_key)
                    };

                    if let Some(ri) = row_index_post {
                        let node = PhysNodeKind::WithRowIndex {
                            input: stream,
                            name: ri.name,
                            offset: Some(ri.offset),
                        };

                        let node_key = phys_sm.insert(PhysNode {
                            output_schema: schema_after_row_index_post.clone(),
                            kind: node,
                        });

                        stream = PhysStream::first(node_key);

                        if reorder_after_row_index_post {
                            let node = PhysNodeKind::SimpleProjection {
                                input: stream,
                                columns: output_schema.iter_names_cloned().collect(),
                            };

                            let node_key = phys_sm.insert(PhysNode {
                                output_schema: output_schema.clone(),
                                kind: node,
                            });

                            stream = PhysStream::first(node_key);
                        }
                    }

                    if let Some(ri) = row_index_post_after_slice {
                        let node = PhysNodeKind::WithRowIndex {
                            input: stream,
                            name: ri.name,
                            offset: Some(ri.offset),
                        };

                        let node_key = phys_sm.insert(PhysNode {
                            output_schema: schema_after_row_index_post,
                            kind: node,
                        });

                        stream = PhysStream::first(node_key);

                        if reorder_after_row_index_post {
                            let node = PhysNodeKind::SimpleProjection {
                                input: stream,
                                columns: output_schema.iter_names_cloned().collect(),
                            };

                            let node_key = phys_sm.insert(PhysNode {
                                output_schema: output_schema.clone(),
                                kind: node,
                            });

                            stream = PhysStream::first(node_key);
                        }
                    }

                    return Ok(stream);
                }
            }
        },

        #[cfg(feature = "python")]
        v @ IR::PythonScan { options } => {
            use polars_plan::dsl::python_dsl::PythonScanSource;

            match options.python_source {
                PythonScanSource::Pyarrow => {
                    // Fallback to in-memory engine.
                    let input = PhysNodeKind::InMemorySource {
                        df: Arc::new(DataFrame::default()),
                        disable_morsel_split: disable_morsel_split.unwrap_or(true),
                    };
                    let input_key =
                        phys_sm.insert(PhysNode::new(Arc::new(Schema::default()), input));
                    let phys_input = PhysStream::first(input_key);

                    let lmdf = Arc::new(LateMaterializedDataFrame::default());
                    let mut lp_arena = Arena::default();
                    let scan_lp_node = lp_arena.add(v.clone());

                    let executor = Mutex::new(create_physical_plan(
                        scan_lp_node,
                        &mut lp_arena,
                        expr_arena,
                        None,
                    )?);

                    let format_str = ctx.prepare_visualization.then(|| {
                        let mut buffer = String::new();
                        write_ir_non_recursive(
                            &mut buffer,
                            ir_arena.get(node),
                            expr_arena,
                            phys_sm.get(phys_input.node).unwrap().output_schema.as_ref(),
                            0,
                        )
                        .unwrap();
                        buffer
                    });

                    PhysNodeKind::InMemoryMap {
                        input: phys_input,
                        map: Arc::new(move |df| {
                            lmdf.set_materialized_dataframe(df);
                            let mut state = ExecutionState::new();
                            executor.lock().execute(&mut state)
                        }),
                        format_str,
                    }
                },
                _ => PhysNodeKind::PythonScan {
                    options: options.clone(),
                },
            }
        },
        IR::Cache { input, id } => {
            let id = *id;
            if let Some(cached) = cache_nodes.get(&id) {
                return Ok(*cached);
            }

            let phys_input = lower_ir!(*input)?;
            cache_nodes.insert(id, phys_input);
            return Ok(phys_input);
        },

        IR::GroupBy {
            input,
            keys,
            aggs,
            schema: output_schema,
            apply,
            maintain_order,
            options,
        } => {
            let input = *input;
            let keys = keys.clone();
            let aggs = aggs.clone();
            let output_schema = output_schema.clone();
            let apply = apply.clone();
            let maintain_order = *maintain_order;
            let options = options.clone();

            let phys_input = lower_ir!(input)?;

            let input_schema = &phys_sm[phys_input.node].output_schema;
            let are_keys_sorted = are_keys_sorted_any(
                is_sorted(input, ir_arena, expr_arena).as_ref(),
                &keys,
                expr_arena,
                input_schema,
            )
            .is_some();

            return build_group_by_stream(
                phys_input,
                &keys,
                &aggs,
                output_schema,
                maintain_order,
                options,
                apply,
                expr_arena,
                phys_sm,
                expr_cache,
                ctx,
                are_keys_sorted,
            );
        },
        IR::Join {
            input_left,
            input_right,
            schema: _,
            left_on,
            right_on,
            options,
        } => {
            let input_left = *input_left;
            let input_right = *input_right;
            let input_left_schema = IR::schema_with_cache(input_left, ir_arena, schema_cache);
            let input_right_schema = IR::schema_with_cache(input_right, ir_arena, schema_cache);
            let left_on = left_on.clone();
            let right_on = right_on.clone();
            let get_expr_name = |e: &ExprIR| e.output_name().clone();
            let left_on_names = left_on.iter().map(get_expr_name).collect_vec();
            let right_on_names = right_on.iter().map(get_expr_name).collect_vec();
            let args = options.args.clone();
            let options = options.options.clone();
            let left_df_sortedness = is_sorted(input_left, ir_arena, expr_arena);
            let left_on_sorted = are_keys_sorted_any(
                left_df_sortedness.as_ref(),
                &left_on,
                expr_arena,
                &input_left_schema,
            );
            let right_df_sortedness = is_sorted(input_right, ir_arena, expr_arena);
            let right_on_sorted = are_keys_sorted_any(
                right_df_sortedness.as_ref(),
                &right_on,
                expr_arena,
                &input_right_schema,
            );
            let join_keys_sorted_together =
                Option::zip(left_on_sorted.as_ref(), right_on_sorted.as_ref())
                    .is_some_and(|(ls, rs)| ls == rs);
            let use_streaming_merge_join = args.how.is_equi() && join_keys_sorted_together;
            #[cfg(feature = "asof_join")]
            let use_streaming_asof_join = if let JoinType::AsOf(ref asof_options) = args.how {
                // Grouped asof-join is not yet supported in the streaming engine.
                asof_options.left_by.is_none() && asof_options.right_by.is_none()
            } else {
                false
            };
            #[cfg(not(feature = "asof_join"))]
            let use_streaming_asof_join = false;

            let phys_left = lower_ir!(input_left)?;
            let phys_right = lower_ir!(input_right)?;

            if (args.how.is_equi() || args.how.is_semi_anti() || use_streaming_asof_join)
                && !args.validation.needs_checks()
            {
                // When lowering the expressions for the keys we need to ensure we keep around the
                // payload columns, otherwise the input nodes can get replaced by input-independent
                // nodes since the lowering code does not see we access any non-literal expressions.
                // So we add dummy expressions before lowering and remove them afterwards.

                let mut aug_left_on = left_on.clone();
                for name in phys_sm[phys_left.node].output_schema.iter_names() {
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    aug_left_on.push(ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone())));
                }
                let mut aug_right_on = right_on.clone();
                for name in phys_sm[phys_right.node].output_schema.iter_names() {
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    aug_right_on.push(ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone())));
                }

                let (mut trans_input_left, mut trans_left_on) = lower_exprs(
                    phys_left,
                    &aug_left_on,
                    expr_arena,
                    phys_sm,
                    expr_cache,
                    ctx,
                )?;
                let (mut trans_input_right, mut trans_right_on) = lower_exprs(
                    phys_right,
                    &aug_right_on,
                    expr_arena,
                    phys_sm,
                    expr_cache,
                    ctx,
                )?;

                trans_left_on.drain(left_on.len()..);
                trans_right_on.drain(right_on.len()..);

                let mut key_descending = left_on_sorted.as_ref().and_then(|v| v[0].descending);
                let key_nulls_last = left_on_sorted.as_ref().and_then(|v| v[0].nulls_last);
                let mut tmp_left_key_col = None;
                let mut tmp_right_key_col = None;
                if use_streaming_merge_join || use_streaming_asof_join {
                    (trans_input_left, trans_left_on, tmp_left_key_col) = append_sorted_key_column(
                        trans_input_left,
                        trans_left_on,
                        left_on_sorted.as_ref(),
                        Some(!args.nulls_equal),
                        expr_arena,
                        phys_sm,
                        expr_cache,
                        ctx,
                    )?;
                    (trans_input_right, trans_right_on, tmp_right_key_col) =
                        append_sorted_key_column(
                            trans_input_right,
                            trans_right_on,
                            right_on_sorted.as_ref(),
                            Some(!args.nulls_equal),
                            expr_arena,
                            phys_sm,
                            expr_cache,
                            ctx,
                        )?;
                }

                let node = if use_streaming_merge_join {
                    let keys_are_row_encoded = left_on_names.len() > 1;
                    if keys_are_row_encoded {
                        key_descending = Some(false);
                    }
                    phys_sm.insert(PhysNode::new(
                        output_schema,
                        PhysNodeKind::MergeJoin {
                            input_left: trans_input_left,
                            input_right: trans_input_right,
                            left_on: left_on_names,
                            right_on: right_on_names,
                            tmp_left_key_col,
                            tmp_right_key_col,
                            keys_row_encoded: keys_are_row_encoded,
                            descending: key_descending.unwrap(),
                            nulls_last: key_nulls_last.unwrap(),
                            args: args.clone(),
                        },
                    ))
                } else if args.how.is_equi() {
                    phys_sm.insert(PhysNode::new(
                        output_schema,
                        PhysNodeKind::EquiJoin {
                            input_left: trans_input_left,
                            input_right: trans_input_right,
                            left_on: trans_left_on,
                            right_on: trans_right_on,
                            args: args.clone(),
                        },
                    ))
                } else if use_streaming_asof_join {
                    assert!(left_on_names.len() == 1 && right_on_names.len() == 1);
                    phys_sm.insert(PhysNode::new(
                        output_schema,
                        PhysNodeKind::AsOfJoin {
                            input_left: trans_input_left,
                            input_right: trans_input_right,
                            left_on: left_on_names[0].clone(),
                            right_on: right_on_names[0].clone(),
                            tmp_left_key_col,
                            tmp_right_key_col,
                            args: args.clone(),
                        },
                    ))
                } else {
                    phys_sm.insert(PhysNode::new(
                        output_schema,
                        PhysNodeKind::SemiAntiJoin {
                            input_left: trans_input_left,
                            input_right: trans_input_right,
                            left_on: trans_left_on,
                            right_on: trans_right_on,
                            args: args.clone(),
                            output_bool: false,
                        },
                    ))
                };
                let mut stream = PhysStream::first(node);
                if let Some((offset, len)) = args.slice {
                    stream = build_slice_stream(stream, offset, len, phys_sm);
                }
                return Ok(stream);
            } else if args.how.is_cross() {
                let node = phys_sm.insert(PhysNode::new(
                    output_schema,
                    PhysNodeKind::CrossJoin {
                        input_left: phys_left,
                        input_right: phys_right,
                        args: args.clone(),
                    },
                ));
                let mut stream = PhysStream::first(node);
                if let Some((offset, len)) = args.slice {
                    stream = build_slice_stream(stream, offset, len, phys_sm);
                }
                return Ok(stream);
            } else {
                PhysNodeKind::InMemoryJoin {
                    input_left: phys_left,
                    input_right: phys_right,
                    left_on,
                    right_on,
                    args,
                    options,
                }
            }
        },

        IR::Distinct { input, options } => {
            let options = options.clone();
            let input = *input;
            let phys_input = lower_ir!(input)?;

            // We don't have a dedicated distinct operator (yet), lower to group
            // by with an aggregate for each column.
            let input_schema = &phys_sm[phys_input.node].output_schema;
            if input_schema.is_empty() {
                // Can't group (or have duplicates) if dataframe has zero-width.
                return Ok(phys_input);
            }

            if options.maintain_order && options.keep_strategy == UniqueKeepStrategy::Last {
                // Unfortunately the order-preserving groupby always orders by the first occurrence
                // of the group so we can't lower this and have to fallback.
                let input_schema = phys_sm[phys_input.node].output_schema.clone();
                let lmdf = Arc::new(LateMaterializedDataFrame::default());
                let mut lp_arena = Arena::default();
                let input_lp_node = lp_arena.add(lmdf.clone().as_ir_node(input_schema));
                let distinct_lp_node = lp_arena.add(IR::Distinct {
                    input: input_lp_node,
                    options,
                });
                let executor = Mutex::new(create_physical_plan(
                    distinct_lp_node,
                    &mut lp_arena,
                    expr_arena,
                    Some(crate::dispatch::build_streaming_query_executor),
                )?);

                let format_str = ctx.prepare_visualization.then(|| {
                    let mut buffer = String::new();
                    write_ir_non_recursive(
                        &mut buffer,
                        ir_arena.get(node),
                        expr_arena,
                        phys_sm.get(phys_input.node).unwrap().output_schema.as_ref(),
                        0,
                    )
                    .unwrap();
                    buffer
                });
                let distinct_node = PhysNode {
                    output_schema,
                    kind: PhysNodeKind::InMemoryMap {
                        input: phys_input,
                        map: Arc::new(move |df| {
                            lmdf.set_materialized_dataframe(df);
                            let mut state = ExecutionState::new();
                            executor.lock().execute(&mut state)
                        }),
                        format_str,
                    },
                };

                return Ok(PhysStream::first(phys_sm.insert(distinct_node)));
            }

            // Create the key and aggregate expressions.
            let all_col_names = input_schema.iter_names().cloned().collect_vec();
            let key_names = if let Some(subset) = options.subset {
                subset.to_vec()
            } else {
                all_col_names.clone()
            };
            let key_name_set: PlHashSet<_> = key_names.iter().cloned().collect();

            let mut group_by_output_schema = Schema::with_capacity(all_col_names.len() + 1);
            let keys = key_names
                .iter()
                .map(|name| {
                    group_by_output_schema
                        .insert(name.clone(), input_schema.get(name).unwrap().clone());
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone()))
                })
                .collect_vec();

            let mut aggs = all_col_names
                .iter()
                .filter(|name| !key_name_set.contains(*name))
                .map(|name| {
                    group_by_output_schema
                        .insert(name.clone(), input_schema.get(name).unwrap().clone());
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    use UniqueKeepStrategy::*;
                    let agg_expr = match options.keep_strategy {
                        First | None | Any => {
                            expr_arena.add(AExpr::Agg(IRAggExpr::First(col_expr)))
                        },
                        Last => expr_arena.add(AExpr::Agg(IRAggExpr::Last(col_expr))),
                    };
                    ExprIR::new(agg_expr, OutputName::ColumnLhs(name.clone()))
                })
                .collect_vec();

            if options.keep_strategy == UniqueKeepStrategy::None {
                // Track the length so we can filter out non-unique keys later.
                let name = unique_column_name();
                group_by_output_schema.insert(name.clone(), DataType::IDX_DTYPE);
                aggs.push(ExprIR::new(
                    expr_arena.add(AExpr::Len),
                    OutputName::Alias(name),
                ));
            }

            let are_keys_sorted = are_keys_sorted_any(
                is_sorted(input, ir_arena, expr_arena).as_ref(),
                &keys,
                expr_arena,
                input_schema,
            )
            .is_some();

            let mut stream = build_group_by_stream(
                phys_input,
                &keys,
                &aggs,
                Arc::new(group_by_output_schema),
                options.maintain_order,
                Arc::new(GroupbyOptions::default()),
                None,
                expr_arena,
                phys_sm,
                expr_cache,
                ctx,
                are_keys_sorted,
            )?;

            if options.keep_strategy == UniqueKeepStrategy::None {
                // Filter to keep only those groups with length 1.
                let unique_name = aggs.last().unwrap().output_name();
                let left = expr_arena.add(AExpr::Column(unique_name.clone()));
                let right = expr_arena.add(AExpr::Literal(LiteralValue::new_idxsize(1)));
                let predicate_aexpr = expr_arena.add(AExpr::BinaryExpr {
                    left,
                    op: polars_plan::dsl::Operator::Eq,
                    right,
                });
                let predicate =
                    ExprIR::new(predicate_aexpr, OutputName::ColumnLhs(unique_name.clone()));
                stream =
                    build_filter_stream(stream, predicate, expr_arena, phys_sm, expr_cache, ctx)?;
            }

            // Restore column order and drop the temporary length column if any.
            let exprs = all_col_names
                .iter()
                .map(|name| {
                    let col_expr = expr_arena.add(AExpr::Column(name.clone()));
                    ExprIR::new(col_expr, OutputName::ColumnLhs(name.clone()))
                })
                .collect_vec();
            stream = build_select_stream(stream, &exprs, expr_arena, phys_sm, expr_cache, ctx)?;

            // We didn't pass the slice earlier to build_group_by_stream because
            // we might have the intermediate keep = "none" filter.
            if let Some((offset, length)) = options.slice {
                stream = build_slice_stream(stream, offset, length, phys_sm);
            }

            return Ok(stream);
        },
        IR::ExtContext { .. } => todo!(),
        IR::Invalid => unreachable!(),
    };

    let node_key = phys_sm.insert(PhysNode::new(output_schema, node_kind));
    Ok(PhysStream::first(node_key))
}

/// Append a sorted key column to the DataFrame.
///
/// If keys_sorted is None, the sortedness of the key will be decided by the
/// default sortedness behavior of RowEncodingVariant::Ordered.
#[allow(clippy::too_many_arguments)]
fn append_sorted_key_column(
    phys_input: PhysStream,
    mut key_exprs: Vec<ExprIR>,
    keys_sorted: Option<&Vec<AExprSorted>>,
    broadcast_nulls: Option<bool>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    expr_cache: &mut ExprCache,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<(PhysStream, Vec<ExprIR>, Option<PlSmallStr>)> {
    let input_schema = &phys_sm[phys_input.node].output_schema.clone();
    let use_row_encoding =
        key_exprs.len() > 1 || key_exprs[0].dtype(input_schema, expr_arena)?.is_nested();
    let key_expr_is_trivial =
        |c: &ExprIR, ea: &mut Arena<AExpr>| matches!(ea.get(c.node()), AExpr::Column(_));
    let (phys_output, key_col_name) = if use_row_encoding {
        let key_col_name = unique_column_name();
        let tfc = ToFieldContext::new(expr_arena, input_schema);
        let sorted_descending =
            keys_sorted.and_then(|v| v.iter().map(|s| s.descending).collect::<Option<Vec<_>>>());
        let sorted_nulls_last =
            keys_sorted.and_then(|v| v.iter().map(|s| s.nulls_last).collect::<Option<Vec<_>>>());
        let expr_dtype = |e: &ExprIR| expr_arena.get(e.node()).to_dtype(&tfc);
        let row_encode_col_expr = AExprBuilder::row_encode(
            key_exprs.clone(),
            key_exprs.iter().map(expr_dtype).try_collect_vec()?,
            RowEncodingVariant::Ordered {
                descending: sorted_descending,
                nulls_last: sorted_nulls_last,
                broadcast_nulls,
            },
            expr_arena,
        )
        .expr_ir(key_col_name.clone());
        key_exprs.clear();
        key_exprs.push(row_encode_col_expr);
        let output =
            build_hstack_stream(phys_input, &key_exprs, expr_arena, phys_sm, expr_cache, ctx)?;
        (output, Some(key_col_name))
    } else if !key_expr_is_trivial(&key_exprs[0], expr_arena) {
        let key_col_name = unique_column_name();
        key_exprs[0] = key_exprs[0].with_alias(key_col_name.clone());
        let output =
            build_hstack_stream(phys_input, &key_exprs, expr_arena, phys_sm, expr_cache, ctx)?;
        (output, Some(key_col_name))
    } else {
        (phys_input, None)
    };
    Ok((phys_output, key_exprs, key_col_name))
}
