use std::cmp;
use std::fmt::Write;

use num_traits::ToPrimitive;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

const DEFAULT_BIN_COUNT: usize = 10;

fn get_breaks<T>(
    ca: &ChunkedArray<T>,
    bin_count: Option<usize>,
    bins: Option<&[f64]>,
    temporal_check: bool,
) -> PolarsResult<(Vec<f64>, bool)>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    let (bins, uniform) = match (bin_count, bins) {
        (Some(_), Some(_)) => {
            return Err(PolarsError::ComputeError(
                "can only provide one of `bin_count` or `bins`".into(),
            ));
        },
        (None, Some(bins)) => {
            // User-supplied bins. Note these are actually bin edges. Check for monotonicity.
            // If we only have one edge, we have no bins.
            let bin_len = bins.len();
            if bin_len > 1 {
                for i in 1..bin_len {
                    if (bins[i] - bins[i - 1]) <= 0.0 {
                        return Err(PolarsError::ComputeError(
                            "bins must increase monotonically".into(),
                        ));
                    }
                }
                (bins.to_vec(), false)
            } else {
                (Vec::<f64>::new(), false)
            }
        },
        (bin_count, None) => {
            // User-supplied bin count, or 10 by default. Compute edges from the data.
            let bin_count = bin_count.unwrap_or(DEFAULT_BIN_COUNT);
            let n = ca.len() - ca.null_count();
            let (offset, width, upper_limit) = if n == 0 {
                // No non-null items; supply unit interval.
                (0.0, 1.0 / bin_count as f64, 1.0)
            } else if n == 1 {
                // Unit interval around single point
                let idx = ca.first_non_null().unwrap();
                // SAFETY: idx is guaranteed to contain an element.
                let center = unsafe { ca.get_unchecked(idx) }.unwrap().to_f64().unwrap();
                (center - 0.5, 1.0 / bin_count as f64, center + 0.5)
            } else {
                // Determine outer bin edges from the data itself
                let min_value = ca.min().unwrap().to_f64().unwrap();
                let max_value = ca.max().unwrap().to_f64().unwrap();

                // If we're dealing with temporals, ensure that the bin count isn't larger than the
                // number of values between the upper and lower bin limits.
                // Note that this check is not completely accurate for large floats.
                if temporal_check {
                    polars_ensure!(
                        (max_value - min_value) as usize >= bin_count,
                        ComputeError: "data type is too coarse to create {bin_count} bins in specified data range"
                    )
                }

                // All data points are identical--use unit interval.
                if min_value == max_value {
                    (min_value - 0.5, 1.0 / bin_count as f64, max_value + 0.5)
                } else {
                    (
                        min_value,
                        (max_value - min_value) / bin_count as f64,
                        max_value,
                    )
                }
            };
            // Manually set the final value to the maximum value to ensure the final value isn't
            // missed due to floating-point precision.
            let out = (0..bin_count)
                .map(|x| (x as f64 * width) + offset)
                .chain(std::iter::once(upper_limit))
                .collect::<Vec<f64>>();
            (out, true)
        },
    };
    Ok((bins, uniform))
}

// O(n) implementation when buckets are fixed-size.
// We deposit items directly into their buckets.
fn uniform_hist_count<T>(breaks: &[f64], ca: &ChunkedArray<T>) -> Vec<IdxSize>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    let num_bins = breaks.len() - 1;
    let mut count: Vec<IdxSize> = vec![0; num_bins];
    let min_break: f64 = breaks[0];
    let max_break: f64 = breaks[num_bins];
    let scale = num_bins as f64 / (max_break - min_break);
    let max_idx = num_bins - 1;

    for chunk in ca.downcast_iter() {
        for item in chunk.non_null_values_iter() {
            let item = item.to_f64().unwrap();
            if item > min_break && item <= max_break {
                // idx > (num_bins - 1) may happen due to floating point representation imprecision
                let mut idx = cmp::min((scale * (item - min_break)) as usize, max_idx);

                // Adjust for float imprecision providing idx > 1 ULP of the breaks
                if item <= breaks[idx] {
                    idx -= 1;
                } else if item > breaks[idx + 1] {
                    idx += 1;
                }

                count[idx] += 1;
            } else if item == min_break {
                count[0] += 1;
            }
        }
    }
    count
}

// Variable-width bucketing. We sort the items and then move linearly through buckets.
fn hist_count<T>(breaks: &[f64], ca: &ChunkedArray<T>) -> Vec<IdxSize>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    let num_bins = breaks.len() - 1;
    let mut breaks_iter = breaks.iter().skip(1); // Skip the first lower bound
    let (min_break, max_break) = (breaks[0], breaks[breaks.len() - 1]);
    let mut upper_bound = *breaks_iter.next().unwrap();
    let mut sorted = ca.sort(false);
    sorted.rechunk_mut();
    let mut current_count: IdxSize = 0;
    let chunk = sorted.downcast_as_array();
    let mut count: Vec<IdxSize> = Vec::with_capacity(num_bins);

    'item: for item in chunk.non_null_values_iter() {
        let item = item.to_f64().unwrap();

        // Cycle through items until we hit the first bucket.
        if item.is_nan() || item < min_break {
            continue;
        }

        while item > upper_bound {
            if item > max_break {
                // No more items will fit in any buckets
                break 'item;
            }

            // Finished with prior bucket; push, reset, and move to next.
            count.push(current_count);
            current_count = 0;
            upper_bound = *breaks_iter.next().unwrap();
        }

        // Item is in bound.
        current_count += 1;
    }
    count.push(current_count);
    count.resize(num_bins, 0); // If we left early, fill remainder with 0.
    count
}

fn compute_hist<T>(
    ca: &ChunkedArray<T>,
    bin_count: Option<usize>,
    bins: Option<&[f64]>,
    temporal_check: bool, // Ensure break range is valid
) -> PolarsResult<(Vec<IdxSize>, Vec<f64>, usize)>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    let (breaks, uniform) = get_breaks(ca, bin_count, bins, temporal_check)?;
    let num_bins = std::cmp::max(breaks.len(), 1) - 1;
    let count = if num_bins > 0 && ca.len() > ca.null_count() {
        if uniform {
            uniform_hist_count(&breaks, ca)
        } else {
            hist_count(&breaks, ca)
        }
    } else {
        vec![0; num_bins]
    };
    Ok((count, breaks, num_bins))
}

fn build_categories(num_bins: usize, breaks: &Series) -> Series {
    let mut categories =
        StringChunkedBuilder::new(PlSmallStr::from_static("category"), breaks.len());
    if num_bins > 0 {
        let mut lower = breaks.get(0).unwrap();
        let mut buf = String::new();
        let mut open_bracket = "[";
        for value in breaks.iter().skip(1) {
            buf.clear();
            write!(buf, "{open_bracket}{lower}, {value}]").unwrap();
            open_bracket = "(";
            categories.append_value(buf.as_str());
            lower = value;
        }
    }
    categories
        .finish()
        .cast(&DataType::from_categories(Categories::global()))
        .unwrap()
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
    let dt = s.dtype();
    let dt_is_temporal = dt.is_temporal();
    let s = if dt_is_temporal {
        &s.to_physical_repr().into_owned()
    } else if dt.is_primitive_numeric() {
        s
    } else {
        polars_bail!(InvalidOperation: "'hist' is only supported for numeric or temporal data")
    };

    let (count, breaks, num_bins) = with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        compute_hist(ca, bin_count, bins_arg, dt_is_temporal)?
    });

    // Generate output columns: breakpoint (optional), breaks (optional) count
    let mut fields = Vec::with_capacity(3);

    if include_breakpoint || include_category {
        let mut breaks = if num_bins > 0 {
            Series::new(PlSmallStr::from_static("breakpoint"), &breaks)
        } else {
            let empty: &[f64; 0] = &[];
            Series::new(PlSmallStr::from_static("breakpoint"), empty)
        };
        if dt_is_temporal {
            breaks = breaks.cast(&DataType::Int64)?.cast(dt)?.into_series();
        }
        if include_breakpoint {
            fields.push(breaks.slice(1, breaks.len()));
        }
        if include_category {
            let categories = build_categories(num_bins, &breaks);
            fields.push(categories);
        };
    }

    let count = Series::new(PlSmallStr::from_static("count"), count);
    fields.push(count);

    Ok(if fields.len() == 1 {
        fields.pop().unwrap().with_name(s.name().clone())
    } else {
        StructChunked::from_series(s.name().clone(), fields[0].len(), fields.iter())
            .unwrap()
            .into_series()
    })
}
