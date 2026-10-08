//! Statistics of an expanded dataset scan, from the per-file statistics that its
//! provider read from table metadata.

use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;

use super::{Card, ScanColumnStats, ScanColumnStatsMap, ScanStats};
use crate::dsl::UnifiedScanArgs;

/// Statistics of a dataset scan over `num_sources` files.
///
/// `rows` is the row count the provider gave. When it is unknown, the file lengths in
/// the table statistics are summed, if every file has one.
pub(crate) fn dataset_scan_stats(
    num_sources: usize,
    rows: Card,
    unified_scan_args: &UnifiedScanArgs,
    schema: &Schema,
) -> ScanStats {
    if num_sources == 0 {
        return ScanStats::exact_rows(0);
    }
    let Some(statistics) = unified_scan_args
        .table_statistics
        .as_ref()
        .map(|s| s.0.as_ref())
        .filter(|df| df.height() == num_sources)
    else {
        return ScanStats::new(rows);
    };

    let lengths = counts(statistics, "len");
    let rows = match rows {
        Card::Unknown => sum_if_all_known(&lengths).map_or(Card::Unknown, Card::approx),
        rows => rows,
    };
    // Any values in the file statistics then belong to deleted rows.
    if rows.value() == Some(0) {
        return ScanStats::new(rows);
    }
    // Deleted rows are still counted in the statistics.
    let exact = unified_scan_args.deletion_files.is_none();
    let row_index = unified_scan_args.row_index.as_ref().map(|ri| &ri.name);

    let mut columns = ScanColumnStatsMap::default();
    for min in statistics.columns() {
        let Some(name) = min.name().strip_suffix("_min") else {
            continue;
        };
        let Some(dtype) = schema.get(name) else {
            continue;
        };
        if row_index.is_some_and(|ri| ri == name) {
            continue;
        }

        let null_counts = counts(statistics, &format_pl_smallstr!("{name}_nc"));
        let null_count = match sum_if_all_known(&null_counts) {
            Some(n) if exact => Card::Exact(n),
            Some(n) => Card::approx(n),
            None => Card::Unknown,
        };
        let int_range = statistics
            .column(&format_pl_smallstr!("{name}_max"))
            .ok()
            .and_then(|max| int_range(min, max, dtype, &lengths, &null_counts));

        if null_count == Card::Unknown && int_range.is_none() {
            continue;
        }
        columns.insert(
            name.into(),
            ScanColumnStats {
                distinct: Card::Unknown,
                null_count,
                avg_byte_width: None,
                int_range,
                int_range_partial: false,
            },
        );
    }

    ScanStats::new(rows).with_columns(columns)
}

/// Inclusive range of an integer, temporal or decimal column, on its physical values.
///
/// A file is left out only when it is known to hold no values. Any other file without
/// both bounds makes the range unknown.
fn int_range(
    min: &Column,
    max: &Column,
    dtype: &DataType,
    lengths: &UInt64Chunked,
    null_counts: &UInt64Chunked,
) -> Option<(i128, i128)> {
    if !(dtype.is_integer() || dtype.is_temporal() || dtype.is_decimal()) {
        return None;
    }
    let (min, max) = (physical(min, dtype)?, physical(max, dtype)?);

    let mut range: Option<(i128, i128)> = None;
    let files = min
        .iter()
        .zip(max.iter())
        .zip(lengths.iter())
        .zip(null_counts.iter());
    for (((min, max), len), null_count) in files {
        if len == Some(0) || (len.is_some() && null_count == len) {
            continue;
        }
        let (Some(min), Some(max)) = (min.extract::<i128>(), max.extract::<i128>()) else {
            return None;
        };
        range = Some(match range {
            Some((lo, hi)) => (lo.min(min), hi.max(max)),
            None => (min, max),
        });
    }
    range
}

/// The bounds in `bounds` as physical values of `dtype`.
fn physical(bounds: &Column, dtype: &DataType) -> Option<Series> {
    let s = bounds.as_materialized_series().cast(dtype).ok()?;
    Some(s.to_physical_repr().rechunk())
}

/// A count column, all unknown if it is missing or does not hold integer counts.
fn counts(statistics: &DataFrame, name: &str) -> UInt64Chunked {
    statistics
        .column(name)
        .ok()
        .and_then(|c| c.cast(&DataType::UInt64).ok())
        .and_then(|c| c.u64().ok().cloned())
        .unwrap_or_else(|| UInt64Chunked::full_null(name.into(), statistics.height()))
}

/// The sum of `counts`, unless any count is unknown.
fn sum_if_all_known(counts: &UInt64Chunked) -> Option<u64> {
    (counts.null_count() == 0).then(|| counts.into_no_null_iter().fold(0, u64::saturating_add))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::dsl::TableStatistics;

    fn args(statistics: Option<DataFrame>) -> UnifiedScanArgs {
        UnifiedScanArgs {
            table_statistics: statistics.map(|df| TableStatistics(Arc::new(df))),
            ..Default::default()
        }
    }

    fn schema() -> Schema {
        Schema::from_iter([Field::new("x".into(), DataType::Int64)])
    }

    fn stats(n: usize, statistics: DataFrame) -> ScanStats {
        dataset_scan_stats(n, Card::Unknown, &args(Some(statistics)), &schema())
    }

    #[test]
    fn empty_source_list_has_zero_rows() {
        let stats = dataset_scan_stats(0, Card::Unknown, &args(None), &schema());
        assert_eq!(stats.rows, Card::Exact(0));
    }

    #[test]
    fn deleted_rows_have_no_column_statistics() {
        let df = df!(
            "len" => [2u32],
            "x_nc" => [0u32],
            "x_min" => [1i64],
            "x_max" => [2i64],
        )
        .unwrap();
        let stats = dataset_scan_stats(1, Card::Exact(0), &args(Some(df)), &schema());
        assert_eq!(stats.rows, Card::Exact(0));
        assert!(stats.column("x").is_none());
    }

    #[test]
    fn rows_need_every_file_length() {
        let all_known = df!("len" => [Some(2u32), Some(3)]).unwrap();
        assert_eq!(stats(2, all_known).rows, Card::approx(5));

        let mixed = df!("len" => [Some(2i64), None]).unwrap();
        assert_eq!(stats(2, mixed).rows, Card::Unknown);

        let all_null = df!("len" => [None::<i64>, None]).unwrap();
        assert_eq!(stats(2, all_null).rows, Card::Unknown);

        let no_len = df!("x_nc" => [0u32, 0]).unwrap();
        assert_eq!(stats(2, no_len).rows, Card::Unknown);

        let provided = dataset_scan_stats(2, Card::Exact(4), &args(None), &schema());
        assert_eq!(provided.rows, Card::Exact(4));
    }

    #[test]
    fn int_range_folds_known_bounds() {
        let df = df!(
            "len" => [2u32, 3, 0],
            "x_nc" => [0u32, 1, 0],
            "x_min" => [Some(5i64), Some(1), None],
            "x_max" => [Some(9i64), Some(4), None],
        )
        .unwrap();
        let stats = stats(3, df);
        let x = stats.column("x").unwrap();
        assert_eq!(x.int_range, Some((1, 9)));
        assert_eq!(x.null_count, Card::Exact(1));
    }

    #[test]
    fn missing_bounds_need_a_proven_all_null_file() {
        // The second file holds only nulls.
        let all_null = df!(
            "len" => [2u32, 3],
            "x_nc" => [0u32, 3],
            "x_min" => [Some(5i64), None],
            "x_max" => [Some(9i64), None],
        )
        .unwrap();
        let x_range = |df| stats(2, df).column("x").and_then(|x| x.int_range);
        assert_eq!(x_range(all_null), Some((5, 9)));

        let unknown_nulls = df!(
            "len" => [2u32, 3],
            "x_nc" => [Some(0u32), None],
            "x_min" => [Some(5i64), None],
            "x_max" => [Some(9i64), None],
        )
        .unwrap();
        assert_eq!(x_range(unknown_nulls), None);

        let unknown_len = df!(
            "len" => [Some(2i64), None],
            "x_nc" => [0u32, 3],
            "x_min" => [Some(5i64), None],
            "x_max" => [Some(9i64), None],
        )
        .unwrap();
        assert_eq!(x_range(unknown_len), None);
    }
}
