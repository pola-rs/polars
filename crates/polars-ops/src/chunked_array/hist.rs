use std::fmt::Write;

use num_traits::ToPrimitive;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::ToTotalOrd;

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
    let mut lower_bound: f64;
    let (breaks, count) = if let Some(bins) = bins {
        let mut breaks = Vec::with_capacity(bins.len() + 1);
        breaks.extend_from_slice(bins);
        breaks.sort_unstable_by_key(|k| k.to_total_ord());
        breaks.push(f64::INFINITY);

        let sorted = ca.sort(false);

        let mut count: Vec<IdxSize> = Vec::with_capacity(breaks.len());
        let mut current_count: IdxSize = 0;
        let mut breaks_iter = breaks.iter();

        // We start with the lower garbage bin.
        // (-inf, B0]
        lower_bound = f64::NEG_INFINITY;
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
        // Push lower bound to infinity
        lower_bound = f64::NEG_INFINITY;
        (breaks, count)
    } else if ca.null_count() == ca.len() {
        lower_bound = f64::NEG_INFINITY;
        let breaks: Vec<f64> = vec![f64::INFINITY];
        let count: Vec<IdxSize> = vec![0];
        (breaks, count)
    } else {
        let start = ChunkAgg::min(ca).unwrap().to_f64().unwrap();
        let end = ChunkAgg::max(ca).unwrap().to_f64().unwrap();

        // If bin_count is omitted, default to the difference between start and stop (unit bins)
        let bin_count = if let Some(bin_count) = bin_count {
            bin_count
        } else {
            (end - start).round() as usize
        };

        // Calculate the breakpoints and make the array. The breakpoints form the RHS of the bins.
        let interval = (end - start) / (bin_count as f64);
        let breaks_iter = (1..(bin_count)).map(|b| start + (b as f64) * interval);
        let mut breaks = Vec::with_capacity(breaks_iter.size_hint().0 + 1);
        breaks.extend(breaks_iter);

        // Extend the left-most edge by 0.1% of the total range to include the minimum value.
        let margin = (end - start) * 0.001;
        lower_bound = start - margin;
        breaks.push(end);

        let mut count: Vec<IdxSize> = vec![0; bin_count];
        let max_bin = breaks.len() - 1;
        for chunk in ca.downcast_iter() {
            for item in chunk.non_null_values_iter() {
                let item = item.to_f64().unwrap();
                let bin = ((((item - start) / interval).ceil() - 1.0) as usize).min(max_bin);
                count[bin] += 1;
            }
        }
        (breaks, count)
    };
    let mut fields = Vec::with_capacity(3);
    if include_category {
        // Use AnyValue for formatting.
        let mut lower = AnyValue::Float64(lower_bound);
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
        fields.insert(0, Series::new("breakpoint", breaks))
    }

    let count = Series::new("count", count);
    fields.push(count);

    if fields.len() == 1 {
        let out = fields.pop().unwrap();
        out.with_name(ca.name())
    } else {
        StructChunked::from_series(ca.name(), &fields)
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
