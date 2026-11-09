//! Answer `min`, `max`, `count` and `null_count` over a whole dataset scan from its statistics.
//! Known gap, shared with file skipping: files without field IDs whose name was reused (#30689).

use polars_core::config;
use polars_core::prelude::*;
use polars_core::scalar::Scalar;
use polars_utils::aliases::{InitHashMaps, PlIndexMap, PlIndexSet};
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::expand_datasets::expand_dataset_scans_with_statistics;
use crate::dsl::python_dataset::PythonDatasetProvider;
use crate::dsl::{FileScanIR, StatisticsBounds};
use crate::plans::iterator::ArenaLpIter;
use crate::plans::stats::{counts, sum_if_all_known};
use crate::plans::{AExpr, IR, IRAggExpr, IRFunctionExpr, LiteralValue};

#[derive(Clone, Copy, Debug)]
enum Statistic {
    Min,
    Max,
    Count,
    NullCount,
    Len,
}

struct Aggregate {
    node: Node,
    statistic: Statistic,
    column: Option<PlSmallStr>,
}

struct Candidate {
    select: Node,
    scan: Node,
    aggregates: Vec<Aggregate>,
}

/// Replaces aggregates over a dataset scan by literals computed from its statistics.
/// Only providers that declare tight statistics (Iceberg) are asked for them.
pub(super) fn aggregate_from_statistics(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let mut candidates = vec![];
    // The columns each scan's provider is asked statistics for.
    let mut scans: PlIndexMap<Node, Vec<PlSmallStr>> = PlIndexMap::new();
    // A select under a duplicated subplan is reached once per parent.
    let mut seen = PlIndexSet::new();
    for (select, ir) in ir_arena.iter(root) {
        let IR::Select { input, expr, .. } = ir else {
            continue;
        };
        if !seen.insert(select) {
            continue;
        }
        let Some((provider, schema)) = whole_dataset_scan(ir_arena.get(*input)) else {
            continue;
        };
        let mut aggregates = vec![];
        if !expr
            .iter()
            .all(|e| collect_aggregates(e.node(), expr_arena, &mut aggregates))
            || !aggregates.iter().all(|a| answerable(a, schema))
        {
            continue;
        }
        let mut columns: Vec<PlSmallStr> = vec![];
        for column in aggregates.iter().filter_map(|a| a.column.as_ref()) {
            if !columns.contains(column) {
                columns.push(column.clone());
            }
        }
        // A lone `len()` is already answered from the row count at lowering.
        if columns.is_empty() || !provider.has_tight_statistics()? {
            continue;
        }
        let scan_columns = scans.entry(*input).or_default();
        for column in columns {
            if !scan_columns.contains(&column) {
                scan_columns.push(column);
            }
        }
        candidates.push(Candidate {
            select,
            scan: *input,
            aggregates,
        });
    }
    if candidates.is_empty() {
        return Ok(());
    }

    let scans = scans
        .into_iter()
        .map(|(scan, columns)| (scan, columns.into()))
        .collect();
    expand_dataset_scans_with_statistics(scans, ir_arena, expr_arena)?;

    for Candidate {
        select,
        scan,
        aggregates,
    } in candidates
    {
        let Some(values) = statistic_values(ir_arena.get(scan), &aggregates) else {
            continue;
        };
        for (aggregate, value) in aggregates.iter().zip(values) {
            expr_arena.replace(aggregate.node, AExpr::Literal(LiteralValue::Scalar(value)));
        }
        let one_row = ir_arena.add(IR::DataFrameScan {
            df: Arc::new(DataFrame::empty_with_height(1)),
            schema: Arc::new(Schema::default()),
            output_schema: None,
        });
        let IR::Select { input, expr, .. } = ir_arena.get_mut(select) else {
            unreachable!()
        };
        *input = one_row;
        // A literal is named `literal`: keep each output's name.
        for e in expr.iter_mut() {
            *e = e.with_alias(e.output_name().clone());
        }
        if config::verbose() {
            eprintln!(
                "aggregate_from_statistics: {} aggregate(s) answered from table statistics",
                aggregates.len()
            );
        }
    }
    Ok(())
}

/// The provider and schema of `ir`, if it scans every row of a dataset: no filter, slice or
/// row index.
fn whole_dataset_scan(ir: &IR) -> Option<(&PythonDatasetProvider, &Schema)> {
    let IR::Scan {
        scan_type,
        file_info,
        predicate: None,
        unified_scan_args: args,
        ..
    } = ir
    else {
        return None;
    };
    let FileScanIR::PythonDataset { dataset_object, .. } = scan_type.as_ref() else {
        return None;
    };
    (args.row_index.is_none() && args.pre_slice.is_none() && args.include_file_paths.is_none())
        .then_some((dataset_object.as_ref(), file_info.schema.as_ref()))
}

/// Collects the aggregates under `node`. Anything besides literals, casts and arithmetic
/// (e.g. `over`, `list.eval`, a bare column) returns false.
fn collect_aggregates(node: Node, arena: &Arena<AExpr>, out: &mut Vec<Aggregate>) -> bool {
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        let ae = arena.get(node);
        if let Some(aggregate) = aggregate(node, ae, arena) {
            out.push(aggregate);
            continue;
        }
        match ae {
            AExpr::Literal(_)
            | AExpr::Cast { .. }
            | AExpr::BinaryExpr { .. }
            | AExpr::Ternary { .. } => ae.inputs_rev(&mut stack),
            _ => return false,
        }
    }
    true
}

/// The eligible aggregate at `node`, if it is one.
fn aggregate(node: Node, ae: &AExpr, arena: &Arena<AExpr>) -> Option<Aggregate> {
    let column = |input: Node| match arena.get(input) {
        AExpr::Column(name) => Some(name.clone()),
        _ => None,
    };
    let (statistic, column) = match ae {
        AExpr::Len => (Statistic::Len, None),
        AExpr::Agg(IRAggExpr::Min {
            input,
            propagate_nans: false,
            propagate_nulls: false,
        }) => (Statistic::Min, Some(column(*input)?)),
        AExpr::Agg(IRAggExpr::Max {
            input,
            propagate_nans: false,
            propagate_nulls: false,
        }) => (Statistic::Max, Some(column(*input)?)),
        AExpr::Agg(IRAggExpr::Count {
            input,
            include_nulls: false,
        }) => (Statistic::Count, Some(column(*input)?)),
        AExpr::Function {
            input,
            function: IRFunctionExpr::NullCount,
            ..
        } if input.len() == 1 => (Statistic::NullCount, Some(column(input[0].node())?)),
        _ => return None,
    };
    Some(Aggregate {
        node,
        statistic,
        column,
    })
}

/// Whether statistics of the column's dtype can answer the aggregate, checked before asking
/// for them.
fn answerable(aggregate: &Aggregate, schema: &Schema) -> bool {
    let Some(column) = &aggregate.column else {
        return true;
    };
    let Some(dtype) = schema.get(column) else {
        return false;
    };
    match aggregate.statistic {
        Statistic::Min | Statistic::Max => has_exact_bounds(dtype),
        // Null counts of nested columns may count their elements.
        _ => !dtype.is_nested(),
    }
}

/// Whether the bounds of `dtype` are exact. Strings can be truncated, and floats lack NaN counts.
fn has_exact_bounds(dtype: &DataType) -> bool {
    dtype.is_integer()
        || dtype.is_bool()
        || dtype.is_decimal()
        || matches!(
            dtype,
            DataType::Date | DataType::Datetime(_, _) | DataType::Time
        )
}

/// Computes the value of each aggregate, if the statistics of `scan` describe every row exactly.
fn statistic_values(scan: &IR, aggregates: &[Aggregate]) -> Option<Vec<Scalar>> {
    let IR::Scan {
        scan_type,
        sources,
        file_info,
        unified_scan_args: args,
        ..
    } = scan
    else {
        return None;
    };
    // Filters, slices and row indices were refused before expansion.
    if matches!(scan_type.as_ref(), FileScanIR::PythonDataset { .. })
        || args.deletion_files.is_some()
    {
        return None;
    }
    let statistics = args
        .table_statistics
        .as_ref()
        .filter(|s| s.1 == StatisticsBounds::Tight)?;
    let schema = &file_info.schema;

    if sources.is_empty() {
        return aggregates
            .iter()
            .map(|a| match a.statistic {
                Statistic::Min | Statistic::Max => {
                    Some(Scalar::null(schema.get(a.column.as_ref()?)?.clone()))
                },
                _ => Some(Scalar::new(IDX_DTYPE, AnyValue::from(0 as IdxSize))),
            })
            .collect();
    }

    let df = statistics.0.as_ref();
    if df.height() != sources.len() {
        return None;
    }
    let lengths = counts(df, "len");
    let total = sum_if_all_known(&lengths)?;
    let to_idx = |n: u64| -> Option<Scalar> {
        Some(Scalar::new(
            IDX_DTYPE,
            AnyValue::from(IdxSize::try_from(n).ok()?),
        ))
    };

    aggregates
        .iter()
        .map(|a| match a.statistic {
            Statistic::Len => to_idx(total),
            Statistic::NullCount | Statistic::Count => {
                let column = a.column.as_ref()?;
                let nulls = sum_if_all_known(&counts(df, &format_pl_smallstr!("{column}_nc")))?;
                let count = match a.statistic {
                    Statistic::NullCount => nulls,
                    _ => total.checked_sub(nulls)?,
                };
                to_idx(count)
            },
            Statistic::Min | Statistic::Max => {
                let column = a.column.as_ref()?;
                let dtype = schema.get(column)?;
                let suffix = match a.statistic {
                    Statistic::Min => "min",
                    _ => "max",
                };
                let bounds = df
                    .column(&format_pl_smallstr!("{column}_{suffix}"))
                    .ok()?
                    .as_materialized_series();
                if bounds.dtype() != dtype {
                    return None;
                }
                let null_counts = counts(df, &format_pl_smallstr!("{column}_nc"));
                // Skip files without values, which can still carry a partition bound.
                let mut has_values = Vec::with_capacity(lengths.len());
                let files = lengths.into_no_null_iter().zip(null_counts.iter());
                for (i, (len, null_count)) in files.enumerate() {
                    let values = len > 0 && null_count != Some(len);
                    if values && bounds.get(i).ok()?.is_null() {
                        return None;
                    }
                    has_values.push(values);
                }
                let bounds = bounds
                    .filter(&BooleanChunked::from_slice(PlSmallStr::EMPTY, &has_values))
                    .ok()?;
                match a.statistic {
                    Statistic::Min => bounds.min_reduce().ok(),
                    _ => bounds.max_reduce().ok(),
                }
            },
        })
        .collect()
}
