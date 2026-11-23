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

fn ensure_hist_dtype(s: &Series) -> PolarsResult<()> {
    polars_ensure!(
        s.dtype().is_primitive_numeric(),
        opq = hist,
        got = s.dtype(),
        expected = "integer or float"
    );
    Ok(())
}

/// The bin edges of `hist`, which for a bin count depend only on the min and max of the column.
#[derive(Clone, Debug)]
pub struct HistBreaks {
    edges: Vec<f64>,
    // Equal-width edges allow counting without sorting.
    uniform: bool,
}

impl HistBreaks {
    /// The edges of `bin_count` bins over `s`, or the given `bins`.
    pub fn new(s: &Series, bin_count: Option<usize>, bins: Option<&Series>) -> PolarsResult<Self> {
        let bins = bins
            .map(|bins| {
                polars_ensure!(bins.null_count() == 0, InvalidOperation: "nulls not supported in 'bins' argument");
                Ok(bins.cast(&DataType::Float64)?.rechunk())
            })
            .transpose()?;
        let bins = bins
            .as_ref()
            .map(|bins| bins.f64().unwrap().cont_slice().unwrap());
        ensure_hist_dtype(s)?;

        let (edges, uniform) = with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            get_breaks(ca, bin_count, bins)?
        });
        Ok(Self { edges, uniform })
    }

    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    pub fn num_bins(&self) -> usize {
        self.edges.len().saturating_sub(1)
    }

    /// The number of values of `s` in each bin.
    pub fn count(&self, s: &Series) -> PolarsResult<Vec<IdxSize>> {
        ensure_hist_dtype(s)?;
        let num_bins = self.num_bins();
        if num_bins == 0 || s.len() == s.null_count() {
            return Ok(vec![0; num_bins]);
        }
        Ok(with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            if self.uniform {
                uniform_hist_count(&self.edges, ca)
            } else {
                hist_count(&self.edges, ca)
            }
        }))
    }

    /// The `hist` output named `name` for the `counts` of each bin.
    pub fn finish(
        &self,
        name: PlSmallStr,
        counts: &[IdxSize],
        include_category: bool,
        include_breakpoint: bool,
    ) -> PolarsResult<Series> {
        let breaks = &self.edges;
        let num_bins = self.num_bins();
        polars_ensure!(
            counts.len() == num_bins,
            ShapeMismatch: "`hist` has {} bins but got {} counts", num_bins, counts.len()
        );

        // Generate output: breakpoint (optional), breaks (optional), count
        let mut fields = Vec::with_capacity(3);

        if include_breakpoint {
            let breakpoints = if num_bins > 0 {
                Series::new(PlSmallStr::from_static("breakpoint"), &breaks[1..])
            } else {
                let empty: &[f64; 0] = &[];
                Series::new(PlSmallStr::from_static("breakpoint"), empty)
            };
            fields.push(breakpoints)
        }

        if include_category {
            let mut categories =
                StringChunkedBuilder::new(PlSmallStr::from_static("category"), breaks.len());
            if num_bins > 0 {
                let mut lower = AnyValue::Float64(breaks[0]);
                let mut buf = String::new();
                let mut open_bracket = "[";
                for br in &breaks[1..] {
                    let br = AnyValue::Float64(*br);
                    buf.clear();
                    write!(buf, "{open_bracket}{lower}, {br}]").unwrap();
                    open_bracket = "(";
                    categories.append_value(buf.as_str());
                    lower = br;
                }
            }
            let categories = categories
                .finish()
                .cast(&DataType::from_categories(Categories::global()))
                .unwrap();
            fields.push(categories);
        };

        let count = Series::new(PlSmallStr::from_static("count"), counts);
        fields.push(count);

        Ok(if fields.len() == 1 {
            fields.pop().unwrap().with_name(name)
        } else {
            StructChunked::from_series(name, fields[0].len(), fields.iter())
                .unwrap()
                .into_series()
        })
    }
}

pub fn hist_series(
    s: &Series,
    bin_count: Option<usize>,
    bins: Option<Series>,
    include_category: bool,
    include_breakpoint: bool,
) -> PolarsResult<Series> {
    let breaks = HistBreaks::new(s, bin_count, bins.as_ref())?;
    let counts = breaks.count(s)?;
    breaks.finish(
        s.name().clone(),
        &counts,
        include_category,
        include_breakpoint,
    )
}
