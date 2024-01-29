use std::fmt::Write;

use arrow::legacy::index::IdxSize;
use num_traits::ToPrimitive;
use polars_core::datatypes::PolarsNumericType;
use polars_core::prelude::{
    ChunkCast, ChunkSort, ChunkedArray, DataType, StringChunkedBuilder, StructChunked, UInt32Type,
    *,
};
use polars_core::with_match_physical_numeric_polars_type;
use polars_error::PolarsResult;
use polars_utils::float::IsFloat;
use polars_utils::total_ord::TotalOrdWrap;

fn compute_hist<T>(
    ca: &ChunkedArray<T>,
    bin_count: Option<usize>,
    bins: Option<&[f64]>,
    include_category: bool,
    include_breakpoint: bool,
) -> Series
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    let (breaks, count) = if let Some(bins) = bins {
        let mut breaks = Vec::with_capacity(bins.len() + 1);
        breaks.extend_from_slice(bins);
        breaks.sort_unstable_by_key(|k| TotalOrdWrap(*k));
        breaks.push(f64::INFINITY);

        let sorted = ca.sort(false);

        let mut count: Vec<IdxSize> = Vec::with_capacity(breaks.len());
        let mut current_count: IdxSize = 0;
        let mut breaks_iter = breaks.iter();

        // We start with the lower garbage bin.
        // (-inf, B0]
        let mut lower_bound = f64::NEG_INFINITY;
        let mut upper_bound = *breaks_iter.next().unwrap();

        for chunk in sorted.downcast_iter() {
            for item in chunk.non_null_values_iter() {
                let item = item.to_f64().unwrap();

                // Not a member of current interval
                if !(item <= upper_bound && item > lower_bound) {
                    loop {
                        // So we push the previous interval
                        count.push(current_count);
                        current_count = 0;
                        lower_bound = upper_bound;
                        upper_bound = *breaks_iter.next().unwrap();
                        if item <= upper_bound && item > lower_bound {
                            break;
                        }
                    }
                }
                current_count += 1;
            }
        }
        // Add last value, this is the garbage bin. E.g. anything that doesn't fit in the bounds.
        count.push(current_count);
        // Add the remaining buckets
        while count.len() < breaks.len() {
            count.push(0)
        }
        (breaks, count)
    } else {
        let min = ChunkAgg::min(ca).unwrap().to_f64().unwrap();
        let max = ChunkAgg::max(ca).unwrap().to_f64().unwrap();

        let start = min.floor() - 1.0;
        let end = max.ceil() + 1.0;

        // If bin_count is omitted, default to the difference between start and stop (unit bins)
        let bin_count = if let Some(bin_count) = bin_count {
            bin_count
        } else {
            (end - start).round() as usize
        };

        // Calculate the breakpoints and make the array
        let interval = (end - start) / (bin_count as f64);

        let breaks_iter = (0..(bin_count)).map(|b| start + (b as f64) * interval);

        let mut breaks = Vec::with_capacity(breaks_iter.size_hint().0 + 1);
        breaks.extend(breaks_iter);
        breaks.push(f64::INFINITY);

        let mut count: Vec<IdxSize> = vec![0; breaks.len()];
        let end_idx = count.len() - 1;

        // start is the closed rhs of the interval, so we subtract the bucket width
        let start_range = start - interval;
        for chunk in ca.downcast_iter() {
            for item in chunk.non_null_values_iter() {
                let item = item.to_f64().unwrap() - start_range;

                // This is needed for numeric stability.
                // Only for integers.
                // we can fall directly on a boundary with an integer.
                let item = item / interval;
                let item = if !T::Native::is_float() && (item.round() - item).abs() < 0.0000001 {
                    item.round() - 1.0
                } else {
                    item.ceil() - 1.0
                };

                let idx = item as usize;
                let idx = std::cmp::min(idx, end_idx);
                count[idx] += 1;
            }
        }
        (breaks, count)
    };
    let mut fields = Vec::with_capacity(3);
    if include_category {
        // Use AnyValue for formatting.
        let mut lower = AnyValue::Float64(f64::NEG_INFINITY);
        let mut categories = StringChunkedBuilder::new("category", breaks.len());

        let mut buf = String::new();
        for br in &breaks {
            let br = AnyValue::Float64(*br);
            buf.clear();
            write!(buf, "({lower}, {br}]").unwrap();
            categories.append_value(buf.as_str());
            lower = br;
        }
        let categories = categories
            .finish()
            .cast(&DataType::Categorical(None, Default::default()))
            .unwrap();
        fields.push(categories);
    };
    if include_breakpoint {
        fields.insert(0, Series::new("break_point", breaks))
    }

    let count = Series::new("count", count);
    fields.push(count);

    if fields.len() == 1 {
        let out = fields.pop().unwrap();
        out.with_name(ca.name())
    } else {
        StructChunked::new(ca.name(), &fields)
            .unwrap()
            .into_series()
    }
}

pub fn hist_series(
    s: &Series,
    bin_count: Option<usize>,
    bins: Option<Series>,
    include_category: bool,
    include_breakpoint: bool,
) -> PolarsResult<Series> {
    let mut bins_arg = None;

    let owned_bins;
    if let Some(bins) = bins {
        polars_ensure!(bins.null_count() == 0, InvalidOperation: "nulls not supported in 'bins' argument");
        let bins = bins.cast(&DataType::Float64)?;
        let bins_s = bins.rechunk();
        owned_bins = bins_s;
        let bins = owned_bins.f64().unwrap();
        let bins = bins.cont_slice().unwrap();
        bins_arg = Some(bins);
    };
    polars_ensure!(s.dtype().is_numeric(), InvalidOperation: "'hist' is only supported for numeric data");

    let out = with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
         let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
         compute_hist(ca, bin_count, bins_arg, include_category, include_breakpoint)
    });
    Ok(out)
}
