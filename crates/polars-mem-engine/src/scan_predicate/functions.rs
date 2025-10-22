use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::{IDX_DTYPE, PlHashMap, PlIndexSet};
use polars_core::schema::Schema;
use polars_expr::{ExpressionConversionState, create_physical_expr};
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{Operator, TableStatistics};
use polars_plan::plans::expr_ir::{ExprIR, OutputName};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_plan::plans::predicates::{aexpr_to_column_predicates, aexpr_to_skip_batch_predicate};
use polars_plan::plans::{AExpr, Context, ExprIRDisplay, MintermIter};
use polars_plan::utils::aexpr_to_leaf_names_iter;
use polars_utils::arena::Arena;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

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
                .all(|name| hive_schema.contains(&name))
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
                Context::Default,
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

    let phys_predicate =
        create_physical_expr(&predicate, Context::Default, expr_arena, schema, state)?;

    if hive_predicate_is_full_predicate {
        hive_predicate = Some(phys_predicate.clone());
    }

    let live_columns = Arc::new(PlIndexSet::from_iter(aexpr_to_leaf_names_iter(
        predicate.node(),
        expr_arena,
    )));

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
                Context::Default,
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
                                Context::Default,
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
    table_statsitics: Option<&TableStatistics>,
    verbose: bool,
) -> PolarsResult<(Option<SkipFilesMask>, Option<&'a ScanIOPredicate>)> {
    #[expect(clippy::never_loop)]
    loop {
        let Some(predicate) = predicate else {
            break;
        };

        let (skip_files_mask, send_predicate_to_readers) = if let Some(hive_parts) = hive_parts
            && let Some(hive_predicate) = &predicate.hive_predicate
        {
            if verbose {
                eprintln!(
                    "initialize_scan_predicate: Source filter mask initialization via hive partitions"
                );
            }

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
        } else if let Some(table_statsitics) = table_statsitics
            && let Some(skip_batch_predicate) = &predicate.skip_batch_predicate
        {
            if verbose {
                eprintln!(
                    "initialize_scan_predicate: Source filter mask initialization via table statistics"
                );
            }

            let exclusion_mask = skip_batch_predicate.evaluate_with_stat_df(&table_statsitics.0)?;

            (SkipFilesMask::Exclusion(exclusion_mask), true)
        } else {
            break;
        };

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
