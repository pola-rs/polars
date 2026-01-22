use std::cell::LazyCell;
use std::sync::Arc;

use polars_core::config;
use polars_core::error::PolarsResult;
use polars_core::prelude::{IDX_DTYPE, IdxCa, InitHashMaps, PlHashMap, PlIndexMap, PlIndexSet};
use polars_core::schema::Schema;
use polars_error::polars_warn;
use polars_expr::{ExpressionConversionState, create_physical_expr};
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::default_values::{
    DefaultFieldValues, IcebergIdentityTransformedPartitionFields,
};
use polars_plan::dsl::deletion::DeletionFilesList;
use polars_plan::dsl::{
    FileScanIR, Operator, PredicateFileSkip, ScanSources, TableStatistics, UnifiedScanArgs,
};
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_plan::plans::predicates::{aexpr_to_column_predicates, aexpr_to_skip_batch_predicate};
use polars_plan::plans::{AExpr, ExprIRDisplay, FileInfo, IR, MintermIter};
use polars_plan::utils::aexpr_to_leaf_names_iter;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, format_pl_smallstr};

use crate::scan_predicate::skip_files_mask::SkipFilesMask;
use crate::scan_predicate::{PhysicalColumnPredicates, ScanPredicate};

pub fn create_scan_predicate(
    predicate: &ExprIR,
    expr_arena: &mut Arena<AExpr>,
    schema: &Arc<Schema>,
    hive_schema: Option<&Schema>,
    state: &mut ExpressionConversionState,
    create_skip_batch_predicate: bool,
    create_column_predicates: bool,
) -> PolarsResult<ScanPredicate> {
    let mut predicate = predicate.clone();

    let mut hive_predicate = None;
    let mut hive_predicate_is_full_predicate = false;

    #[expect(clippy::never_loop)]
    loop {
        let Some(hive_schema) = hive_schema else {
            break;
        };

        let mut hive_predicate_parts = vec![];
        let mut non_hive_predicate_parts = vec![];

        for predicate_part in MintermIter::new(predicate.node(), expr_arena) {
            if aexpr_to_leaf_names_iter(predicate_part, expr_arena)
                .all(|name| hive_schema.contains(name))
            {
                hive_predicate_parts.push(predicate_part)
            } else {
                non_hive_predicate_parts.push(predicate_part)
            }
        }

        if hive_predicate_parts.is_empty() {
            break;
        }

        if non_hive_predicate_parts.is_empty() {
            hive_predicate_is_full_predicate = true;
            break;
        }

        {
            let mut iter = hive_predicate_parts.into_iter();
            let mut node = iter.next().unwrap();

            for next_node in iter {
                node = expr_arena.add(AExpr::BinaryExpr {
                    left: node,
                    op: Operator::And,
                    right: next_node,
                });
            }

            hive_predicate = Some(create_physical_expr(
                &ExprIR::from_node(node, expr_arena),
                expr_arena,
                schema,
                state,
            )?)
        }

        {
            let mut iter = non_hive_predicate_parts.into_iter();
            let mut node = iter.next().unwrap();

            for next_node in iter {
                node = expr_arena.add(AExpr::BinaryExpr {
                    left: node,
                    op: Operator::And,
                    right: next_node,
                });
            }

            predicate = ExprIR::from_node(node, expr_arena);
        }

        break;
    }

    let phys_predicate = create_physical_expr(&predicate, expr_arena, schema, state)?;

    if hive_predicate_is_full_predicate {
        hive_predicate = Some(phys_predicate.clone());
    }

    let live_columns = Arc::new(PlIndexSet::from_iter(
        aexpr_to_leaf_names_iter(predicate.node(), expr_arena).cloned(),
    ));

    let mut skip_batch_predicate = None;

    if create_skip_batch_predicate {
        if let Some(node) = aexpr_to_skip_batch_predicate(predicate.node(), expr_arena, schema) {
            let expr = ExprIR::new(node, predicate.output_name_inner().clone());

            if std::env::var("POLARS_OUTPUT_SKIP_BATCH_PRED").as_deref() == Ok("1") {
                eprintln!("predicate: {}", predicate.display(expr_arena));
                eprintln!("skip_batch_predicate: {}", expr.display(expr_arena));
            }

            let mut skip_batch_schema = Schema::with_capacity(1 + live_columns.len());

            skip_batch_schema.insert(PlSmallStr::from_static("len"), IDX_DTYPE);
            for (col, dtype) in schema.iter() {
                if !live_columns.contains(col) {
                    continue;
                }

                skip_batch_schema.insert(format_pl_smallstr!("{col}_min"), dtype.clone());
                skip_batch_schema.insert(format_pl_smallstr!("{col}_max"), dtype.clone());
                skip_batch_schema.insert(format_pl_smallstr!("{col}_nc"), IDX_DTYPE);
            }

            skip_batch_predicate = Some(create_physical_expr(
                &expr,
                expr_arena,
                &Arc::new(skip_batch_schema),
                state,
            )?);
        }
    }

    let column_predicates = if create_column_predicates {
        let column_predicates = aexpr_to_column_predicates(predicate.node(), expr_arena, schema);
        if std::env::var("POLARS_OUTPUT_COLUMN_PREDS").as_deref() == Ok("1") {
            eprintln!("column_predicates: {{");
            eprintln!("  [");
            for (pred, spec) in column_predicates.predicates.values() {
                eprintln!(
                    "    {} ({spec:?}),",
                    ExprIRDisplay::display_node(*pred, expr_arena)
                );
            }
            eprintln!("  ],");
            eprintln!(
                "  is_sumwise_complete: {}",
                column_predicates.is_sumwise_complete
            );
            eprintln!("}}");
        }
        PhysicalColumnPredicates {
            predicates: column_predicates
                .predicates
                .into_iter()
                .map(|(n, (p, s))| {
                    PolarsResult::Ok((
                        n,
                        (
                            create_physical_expr(
                                &ExprIR::new(p, OutputName::Alias(PlSmallStr::EMPTY)),
                                expr_arena,
                                schema,
                                state,
                            )?,
                            s,
                        ),
                    ))
                })
                .collect::<PolarsResult<PlHashMap<_, _>>>()?,
            is_sumwise_complete: column_predicates.is_sumwise_complete,
        }
    } else {
        PhysicalColumnPredicates {
            predicates: PlHashMap::default(),
            is_sumwise_complete: false,
        }
    };

    PolarsResult::Ok(ScanPredicate {
        predicate: phys_predicate,
        live_columns,
        skip_batch_predicate,
        column_predicates,
        hive_predicate,
        hive_predicate_is_full_predicate,
    })
}

/// # Returns
/// (skip_files_mask, predicate)
pub fn initialize_scan_predicate<'a>(
    predicate: Option<&'a ScanIOPredicate>,
    hive_parts: Option<&HivePartitionsDf>,
    table_statistics: Option<&TableStatistics>,
    verbose: bool,
) -> PolarsResult<(Option<SkipFilesMask>, Option<&'a ScanIOPredicate>)> {
    #[expect(clippy::never_loop)]
    loop {
        let Some(predicate) = predicate else {
            break;
        };

        let expected_mask_len: usize;

        let (skip_files_mask, send_predicate_to_readers) = if let Some(hive_parts) = hive_parts
            && let Some(hive_predicate) = &predicate.hive_predicate
        {
            if verbose {
                eprintln!(
                    "initialize_scan_predicate: Source filter mask initialization via hive partitions"
                );
            }

            expected_mask_len = hive_parts.df().height();

            let inclusion_mask = hive_predicate
                .evaluate_io(hive_parts.df())?
                .bool()?
                .rechunk()
                .into_owned()
                .downcast_into_iter()
                .next()
                .unwrap()
                .values()
                .clone();

            (
                SkipFilesMask::Inclusion(inclusion_mask),
                !predicate.hive_predicate_is_full_predicate,
            )
        } else if let Some(table_statistics) = table_statistics
            && let Some(skip_batch_predicate) = &predicate.skip_batch_predicate
        {
            if verbose {
                eprintln!(
                    "initialize_scan_predicate: Source filter mask initialization via table statistics"
                );
            }

            expected_mask_len = table_statistics.0.height();

            let exclusion_mask = skip_batch_predicate.evaluate_with_stat_df(&table_statistics.0)?;

            (SkipFilesMask::Exclusion(exclusion_mask), true)
        } else {
            break;
        };

        if skip_files_mask.len() != expected_mask_len {
            polars_warn!(
                "WARNING: \
                initialize_scan_predicate: \
                filter mask length mismatch (length: {}, expected: {}). Files \
                will not be skipped. This is a bug; please open an issue with \
                a reproducible example if possible.",
                skip_files_mask.len(),
                expected_mask_len
            );
            return Ok((None, Some(predicate)));
        }

        if verbose {
            eprintln!(
                "initialize_scan_predicate: Predicate pushdown allows skipping {} / {} files",
                skip_files_mask.num_skipped_files(),
                skip_files_mask.len()
            );
        }

        return Ok((
            Some(skip_files_mask),
            send_predicate_to_readers.then_some(predicate),
        ));
    }

    Ok((None, predicate))
}

/// Filters the list of files in an `IR::Scan` based on the contained predicate. This is possible
/// if the predicate has components that refer to only the hive parts and there is no e.g.
/// row index / slice.
///
/// This also applies the projection onto the hive parts.
///
/// # Panics
/// Panics if `scan_ir_node` is not `IR::Scan`.
pub fn apply_scan_predicate_to_scan_ir(
    scan_ir_node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let scan_ir_schema = IR::schema(ir_arena.get(scan_ir_node), ir_arena).into_owned();
    let scan_ir = ir_arena.get_mut(scan_ir_node);

    let IR::Scan {
        sources,
        hive_parts,
        predicate,
        predicate_file_skip_applied,
        unified_scan_args,
        file_info,
        ..
    } = scan_ir
    else {
        unreachable!()
    };

    if let Some(hive_parts) = hive_parts.as_mut() {
        *hive_parts = hive_parts.filter_columns(&scan_ir_schema);
    }

    if unified_scan_args.has_row_index_or_slice() || predicate_file_skip_applied.is_some() {
        return Ok(());
    }

    let Some(predicate) = predicate else {
        return Ok(());
    };

    match sources {
        // Files cannot be `gather()`ed.
        ScanSources::Files(_) => return Ok(()),
        ScanSources::Paths(_) | ScanSources::Buffers(_) => {},
    }

    let verbose = config::verbose();

    let scan_predicate = create_scan_predicate(
        predicate,
        expr_arena,
        &scan_ir_schema,
        hive_parts.as_ref().map(|hp| hp.df().schema().as_ref()),
        &mut ExpressionConversionState::new(true),
        true,  // create_skip_batch_predicate
        false, // create_column_predicates
    )?
    .to_io(None, file_info.schema.clone());

    let (skip_files_mask, predicate_to_readers) = initialize_scan_predicate(
        Some(&scan_predicate),
        hive_parts.as_ref(),
        unified_scan_args.table_statistics.as_ref(),
        verbose,
    )?;

    if let Some(skip_files_mask) = skip_files_mask {
        assert_eq!(skip_files_mask.len(), sources.len());

        let predicate_file_skip = PredicateFileSkip {
            no_residual_predicate: predicate_to_readers.is_none(),
            original_len: sources.len(),
        };

        if verbose {
            eprintln!("apply_scan_predicate_to_scan_ir: {predicate_file_skip:?}");
        }

        *predicate_file_skip_applied = Some(predicate_file_skip);

        if skip_files_mask.num_skipped_files() > 0 {
            filter_scan_ir(scan_ir, skip_files_mask.non_skipped_files_idx_iter())
        }
    }

    Ok(())
}

/// Filters the paths for a scan IR. This also involves performing selections on
/// e.g. hive partitions, deletion files.
///
/// Note: `selected_path_indices` should be cheaply cloneable.
///
/// # Panics
/// Panics if `scan_ir` is not `IR::Scan`.
pub fn filter_scan_ir<I>(scan_ir: &mut IR, selected_path_indices: I)
where
    I: Iterator<Item = usize> + Clone,
{
    let IR::Scan {
        sources,
        file_info:
            FileInfo {
                schema: _,
                reader_schema,
                row_estimation,
            },
        hive_parts,
        predicate: _,
        predicate_file_skip_applied: _,
        output_schema: _,
        scan_type,
        unified_scan_args,
    } = scan_ir
    else {
        panic!("{:?}", scan_ir);
    };

    let size_hint = selected_path_indices.size_hint();

    if size_hint.0 == sources.len()
        && size_hint.1 == Some(sources.len())
        && selected_path_indices
            .clone()
            .enumerate()
            .all(|(i, x)| i == x)
    {
        return;
    }

    let UnifiedScanArgs {
        schema: _,
        cloud_options: _,
        hive_options: _,
        rechunk: _,
        cache: _,
        glob: _,
        hidden_file_prefix: _,
        projection: _,
        column_mapping: _,
        default_values,
        // Ensure these are None.
        row_index: None,
        pre_slice: None,
        cast_columns_policy: _,
        missing_columns_policy: _,
        extra_columns_policy: _,
        include_file_paths: _,
        table_statistics,
        deletion_files,
        row_count,
    } = unified_scan_args.as_mut()
    else {
        panic!("{unified_scan_args:?}")
    };

    *row_count = None;

    if selected_path_indices.clone().next() != Some(0) {
        *reader_schema = None;

        // Ensure the metadata is unset, otherwise it may incorrectly be used at
        // scan. This is especially important for Parquet as it requires the
        // correct `is_nullable` in the arrow field.
        match scan_type.as_mut() {
            #[cfg(feature = "parquet")]
            FileScanIR::Parquet {
                options: _,
                metadata,
            } => *metadata = None,

            #[cfg(feature = "ipc")]
            FileScanIR::Ipc {
                options: _,
                metadata,
            } => *metadata = None,

            #[cfg(feature = "csv")]
            FileScanIR::Csv { options: _ } => {},

            #[cfg(feature = "json")]
            FileScanIR::NDJson { options: _ } => {},

            #[cfg(feature = "python")]
            FileScanIR::PythonDataset {
                dataset_object: _,
                cached_ir,
            } => *cached_ir.lock().unwrap() = None,

            #[cfg(feature = "scan_lines")]
            FileScanIR::Lines { name: _ } => {},

            FileScanIR::Anonymous {
                options: _,
                function: _,
            } => {},
        }
    }

    let selected_path_indices_idxsize = LazyCell::new(|| {
        selected_path_indices
            .clone()
            .map(|i| IdxSize::try_from(i).unwrap())
            .collect::<Vec<_>>()
    });

    *deletion_files = deletion_files.as_ref().and_then(|x| match x {
        DeletionFilesList::IcebergPositionDelete(deletions) => {
            let mut out = None;

            for (out_idx, source_idx) in selected_path_indices.clone().enumerate() {
                if let Some(v) = deletions.get(&source_idx) {
                    out.get_or_insert_with(|| {
                        PlIndexMap::with_capacity(selected_path_indices.size_hint().0 - out_idx)
                    })
                    .insert(out_idx, v.clone());
                }
            }

            out.map(|x| DeletionFilesList::IcebergPositionDelete(Arc::new(x)))
        },
    });

    *table_statistics = table_statistics.as_ref().map(|x| {
        let df_height = IdxSize::try_from(x.0.height()).unwrap();

        assert!(selected_path_indices_idxsize.iter().all(|x| *x < df_height));

        TableStatistics(Arc::new(unsafe {
            x.0.take_slice_unchecked(&selected_path_indices_idxsize)
        }))
    });

    let original_sources_len = sources.len();
    *sources = sources.gather(selected_path_indices.clone()).unwrap();
    *row_estimation = (
        None,
        row_estimation
            .1
            .div_ceil(original_sources_len)
            .saturating_mul(sources.len()),
    );

    *hive_parts = hive_parts.as_ref().map(|hp| {
        let df = hp.df();
        let df_height = IdxSize::try_from(df.height()).unwrap();

        assert!(selected_path_indices_idxsize.iter().all(|x| *x < df_height));

        // Safety: Asserted all < df.height() above.
        unsafe { df.take_slice_unchecked(&selected_path_indices_idxsize) }.into()
    });

    *default_values = default_values.as_ref().map(|x| match x {
        DefaultFieldValues::Iceberg(v) => {
            let mut out = PlIndexMap::with_capacity(v.len());
            let mut gather_indices = PlHashMap::with_capacity(v.len());

            for (k, v) in v.iter() {
                out.insert(
                    *k,
                    v.as_ref().map_err(Clone::clone).map(|partition_values| {
                        if !gather_indices.contains_key(&partition_values.len()) {
                            gather_indices.insert(
                                partition_values.len(),
                                selected_path_indices
                                    .clone()
                                    .map(|i| {
                                        (i < partition_values.len())
                                            .then(|| IdxSize::try_from(i).unwrap())
                                    })
                                    .collect::<IdxCa>(),
                            );
                        }

                        unsafe {
                            partition_values.take_unchecked(
                                gather_indices.get(&partition_values.len()).unwrap(),
                            )
                        }
                    }),
                );
            }

            DefaultFieldValues::Iceberg(Arc::new(IcebergIdentityTransformedPartitionFields(out)))
        },
    });
}
