use std::cmp;
use std::fmt::Write;
use std::marker::PhantomData;

use polars_core::prelude::*;

pub(crate) const DEFAULT_BIN_COUNT: usize = 10;

struct UniformWidthHistogram<T: PolarsNumericType> {
    lower: T::Native,
    upper: T::Native,
    num_bins: usize,
}

struct VariableWidthHistogram<T: PolarsNumericType> {
    edges: Series,
    phantom: PhantomData<T>,
}

trait Histogram {
    fn hist(&self, s: &Series) -> PolarsResult<Series>;
    fn get_breakpoints(&self) -> Series;
    fn get_categories(&self) -> Series;
}

fn new_histogram(
    s: &Series,
    bin_count: Option<usize>,
    edges: Option<Series>,
) -> PolarsResult<Box<dyn Histogram>> {
    let hist = match (bin_count, edges) {
        (Some(_), Some(_)) => {
            polars_bail!(InvalidOperation: "cannot supply both 'bins' and 'bin_count' to hist")
        },
        (None, Some(edges)) => match s.dtype() {
            &DataType::Float64 => {
                let hist = VariableWidthHistogram::<Float64Type>::new(edges)?;
                Box::new(hist) as Box<dyn Histogram>
            },
            _ => unimplemented!(),
        },
        _ => match s.dtype() {
            &DataType::Float64 => {
                let hist = UniformWidthHistogram::<Float64Type>::new(s.f64()?, bin_count);
                Box::new(hist) as Box<dyn Histogram>
            },
            _ => unimplemented!(),
        },
    };
    Ok(hist)
}

impl UniformWidthHistogram<Float64Type> {
    fn new(ca: &ChunkedArray<Float64Type>, num_bins: Option<usize>) -> Self {
        let n = ca.len() - ca.null_count();
        let (lower, upper) = if n == 0 {
            // No non-null items; supply the unit interval.
            (0.0, 1.0)
        } else if n == 1 {
            // One non-null item; create a unit interval centered around the single value.
            let idx = ca.first_non_null().unwrap();
            // SAFETY: idx is guaranteed to contain an element.
            let center = unsafe { ca.get_unchecked(idx) }.unwrap();
            (center - 0.5, center + 0.5)
        } else {
            // Determine outer bin edges from the data itself
            let lower = ca.min().unwrap();
            let upper = ca.max().unwrap();

            // All data points are identical--use unit interval.
            if lower == upper {
                (lower - 0.5, upper + 0.5)
            } else {
                (lower, upper)
            }
        };

        Self {
            lower,
            upper,
            num_bins: num_bins.unwrap_or(DEFAULT_BIN_COUNT),
        }
    }
}

impl<T: PolarsNumericType> VariableWidthHistogram<T> {
    // Given a set of user-provided edges, ensure they are valid, i.e. monotonic increasing.
    // In the future, we may consider detecting whether user-provided bins are uniformly
    // distributed.
    fn check_bins(edges: &Series) -> PolarsResult<()> {
        let bin_len = edges.len();
        if bin_len > 1 {
            for idx in 1..bin_len {
                // SAFETY: idx is within bounds.
                unsafe {
                    if edges.get_unchecked(idx) < edges.get_unchecked(idx - 1) {
                        return Err(PolarsError::ComputeError(
                            "bins must increase monotonically".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn new(edges: Series) -> PolarsResult<Self> {
        // We cannot have only one edge, this creates zero bins.
        let edges = if edges.len() < 2 {
            Series::new_empty(PlSmallStr::EMPTY, &DataType::Float64)
        } else {
            edges.rechunk()
        };
        Self::check_bins(&edges)?;
        Ok(Self {
            edges,
            phantom: PhantomData,
        })
    }
}

impl Histogram for UniformWidthHistogram<Float64Type> {
    fn hist(&self, s: &Series) -> PolarsResult<Series> {
        let name = PlSmallStr::from_static("count");
        if self.num_bins == 0 {
            return Ok(Series::new_empty(name, &IDX_DTYPE));
        }
        let ca = s.f64()?;
        let scale = self.num_bins as f64 / (self.upper - self.lower);
        let max_idx = self.num_bins - 1;
        let bin_width = (self.upper - self.lower) / self.num_bins as f64;

        let mut count: Vec<IdxSize> = vec![0; self.num_bins];
        for chunk in ca.downcast_iter() {
            for item in chunk.non_null_values_iter() {
                if item > self.lower && item <= self.upper {
                    let mut idx = (scale * (item - self.lower)) as usize;

                    // Adjust for float imprecision providing idx > 1 ULP of the breaks
                    let current_break = self.lower + bin_width * idx as f64;
                    if item <= current_break {
                        idx -= 1;
                    } else if item > current_break + bin_width {
                        idx += 1;
                    }

                    // idx > (num_bins - 1) may happen due to floating point representation imprecision
                    idx = cmp::min(idx, max_idx);
                    count[idx] += 1;
                } else if item == self.lower {
                    count[0] += 1;
                }
            }
        }
        Ok(Series::from_vec(name, count))
    }

    fn get_breakpoints(&self) -> Series {
        let name = PlSmallStr::from_static("breakpoint");
        if self.num_bins == 0 {
            return Series::new_empty(name, &DataType::Float64);
        }
        let bin_width = (self.upper - self.lower) / self.num_bins as f64;
        Series::from_iter(
            (1..self.num_bins)
                .map(|idx| self.lower + (idx as f64) * bin_width)
                .chain(std::iter::once(self.upper)),
        )
        .with_name(name)
    }

    fn get_categories(&self) -> Series {
        let name = PlSmallStr::from_static("category");
        let dt = DataType::Categorical(Categories::global(), Categories::global().mapping());
        if self.num_bins == 0 {
            return Series::new_empty(name, &dt);
        }

        let mut categories = StringChunkedBuilder::new(name, self.num_bins);
        let width = (self.upper - self.lower) / self.num_bins as f64;
        let mut lower = AnyValue::Float64(self.lower);
        let upper = AnyValue::Float64(self.lower + width);

        // Write first fully-closed interval.
        let mut buf = format!("[{}, {}]", lower, upper);
        categories.append_value(buf.as_str());
        lower = upper;

        // Write remaining right-closed intervals.
        for idx in 1..self.num_bins {
            let upper = AnyValue::Float64(self.lower + (idx + 1) as f64 * width);
            buf.clear();
            write!(buf, "({lower}, {upper}]").unwrap();
            categories.append_value(buf.as_str());
            lower = upper;
        }
        categories
            .finish()
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap()
    }
}

impl Histogram for VariableWidthHistogram<Float64Type> {
    // Variable-width bucketing. We sort the items and then move linearly through buckets.
    fn hist(&self, s: &Series) -> PolarsResult<Series> {
        let name = PlSmallStr::from_static("count");
        if self.edges.is_empty() {
            return Ok(Series::new_empty(name, &IDX_DTYPE));
        }
        let ca = s.f64()?;
        let edges = self.edges.f64().unwrap().cont_slice().unwrap();
        let num_edges = edges.len();
        let mut edges_iter = edges.iter().skip(1); // Skip the first lower bound
        let (min_break, max_break) = (edges[0], edges[num_edges - 1]);
        let mut upper_bound = *edges_iter.next().unwrap();
        let mut sorted = ca.sort(false);
        sorted.rechunk_mut();
        let mut current_count: IdxSize = 0;
        let chunk = sorted.downcast_as_array();
        let mut count: Vec<IdxSize> = Vec::with_capacity(num_edges - 1);

        'item: for item in chunk.non_null_values_iter() {
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
                upper_bound = *edges_iter.next().unwrap();
            }

            // Item is in bound.
            current_count += 1;
        }
        count.push(current_count);
        count.resize(num_edges - 1, 0); // If we left early, fill remainder with 0.
        Ok(Series::from_vec(name, count))
    }

    fn get_breakpoints(&self) -> Series {
        let name = PlSmallStr::from_static("breakpoint");
        if self.edges.is_empty() {
            return Series::new_empty(name, &DataType::Float64);
        }
        self.edges.slice(1, self.edges.len() - 1).with_name(name)
    }

    fn get_categories(&self) -> Series {
        let name = PlSmallStr::from_static("category");
        let dt = DataType::Categorical(Categories::global(), Categories::global().mapping());
        if self.edges.is_empty() {
            return Series::new_empty(name, &dt);
        }
        let mut categories =
            StringChunkedBuilder::new(PlSmallStr::from_static("category"), self.edges.len() - 1);
        let edges = self.edges.f64().unwrap().cont_slice().unwrap();
        let mut lower = AnyValue::Float64(edges[0]);
        let mut buf = String::new();
        let mut change_bracket = true;
        let mut bracket = "[";
        for br in &edges[1..] {
            let br = AnyValue::Float64(*br);
            buf.clear();
            write!(buf, "{bracket}{lower}, {br}]").unwrap();
            if change_bracket {
                bracket = "(";
                change_bracket = false;
            }
            categories.append_value(buf.as_str());
            lower = br;
        }
        categories.finish().cast(&dt).unwrap()
    }
}

pub fn hist_series(
    s: &Series,
    bin_count: Option<usize>,
    bins: Option<Series>,
    include_category: bool,
    include_breakpoint: bool,
) -> PolarsResult<Series> {
    let histogram = new_histogram(s, bin_count, bins)?;

    // Generate output: breakpoint (optional), breaks (optional), count
    let count = histogram.hist(s)?;
    Ok(if include_category || include_breakpoint {
        let mut fields = Vec::with_capacity(3);
        if include_breakpoint {
            fields.push(histogram.get_breakpoints());
        }
        if include_category {
            fields.push(histogram.get_categories());
        }
        fields.push(count);
        StructChunked::from_series(s.name().clone(), fields[0].len(), fields.iter())
            .unwrap()
            .into_series()
    } else {
        count
    })
}
