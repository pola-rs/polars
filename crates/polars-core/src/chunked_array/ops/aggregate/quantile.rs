use num_traits::AsPrimitive;
use polars_compute::filter::filter_values;
use polars_compute::rolling::QuantileMethod;
use polars_compute::selection::{
    MaskedSlice, SELECT_BLOCK_LEN, SELECT_MIN_PAR_LEN, select_many_by, select_many_seq_by,
};
#[cfg(feature = "dtype-f16")]
use polars_utils::float16::pf16;
use rayon::prelude::*;

use super::*;
use crate::runtime::RAYON;

pub trait QuantileAggSeries {
    /// Get the median of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn median_reduce(&self) -> Scalar;

    /// Get the quantile of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn quantile_reduce(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Scalar>;

    /// Get the quantiles of the [`ChunkedArray`] as a new [`Series`] of same length as quantiles
    fn quantiles_reduce(&self, _quantiles: &[f64], _method: QuantileMethod)
    -> PolarsResult<Scalar>;
}

/// Returns the 0-based indices into `n_nonnull` sorted non-null values that `quantile` needs, and
/// the fraction of the way from the lower to the upper one: `(lower_idx, upper_idx, fraction)`.
///
/// `upper_idx == lower_idx` unless `method` combines two distinct values. Only `Linear` uses
/// `fraction`; `Midpoint` averages the two values.
pub fn quantile_indices(
    n_nonnull: usize,
    quantile: f64,
    method: QuantileMethod,
) -> (usize, usize, f64) {
    assert!(n_nonnull > 0);
    let float_idx = (n_nonnull as f64 - 1.0) * quantile;
    let floor_idx = float_idx as usize;
    let ceil_idx = (float_idx.ceil() as usize).min(n_nonnull - 1);
    match method {
        QuantileMethod::Nearest => {
            let idx = float_idx.round() as usize;
            (idx, idx, 0.0)
        },
        QuantileMethod::Lower => (floor_idx, floor_idx, 0.0),
        QuantileMethod::Higher => (ceil_idx, ceil_idx, 0.0),
        QuantileMethod::Midpoint | QuantileMethod::Linear => {
            (floor_idx, ceil_idx, float_idx - floor_idx as f64)
        },
        QuantileMethod::Equiprobable => {
            let idx = ((n_nonnull as f64 * quantile).ceil() - 1.0).max(0.0) as usize;
            (idx, idx, 0.0)
        },
    }
}

/// Extracts the quantiles of the `quantile` expression input, a single quantile or a list of them.
pub fn extract_quantiles(quantile: &Series) -> PolarsResult<Vec<f64>> {
    polars_ensure!(!quantile.is_empty(), ComputeError:
        "the 'quantile' expression input should produce a single quantile or a list of quantiles, \
        got an empty input"
    );
    polars_ensure!(quantile.len() == 1, ComputeError:
        "polars does not support varying quantiles yet, \
        make sure the 'quantile' expression input produces a single quantile or a list of quantiles"
    );
    let quantiles = match quantile.dtype() {
        DataType::List(_) => {
            let inner = quantile.list()?.get_as_series(0).ok_or_else(
                || polars_err!(ComputeError: "quantile expression contains null values"),
            )?;
            polars_ensure!(!inner.has_nulls(), ComputeError: "quantile expression contains null values");
            inner
                .cast(&DataType::Float64)?
                .f64()?
                .into_no_null_iter()
                .collect()
        },
        _ => vec![quantile.get(0)?.try_extract()?],
    };
    check_quantiles(&quantiles)?;
    Ok(quantiles)
}

fn check_quantiles(quantiles: &[f64]) -> PolarsResult<()> {
    for &q in quantiles {
        polars_ensure!(
            (0.0..=1.0).contains(&q),
            ComputeError: "`quantile` should be between 0.0 and 1.0, got {}", q,
        );
    }
    Ok(())
}

fn linear_interpol<T: Float>(lower: T, upper: T, fraction: f64) -> T {
    if lower == upper {
        lower
    } else {
        let proportion: T = T::from(fraction).unwrap();
        proportion * (upper - lower) + lower
    }
}
fn midpoint_interpol<T: Float>(lower: T, upper: T) -> T {
    if lower == upper {
        lower
    } else {
        (lower + upper) / (T::one() + T::one())
    }
}

/// Combines the values at the ranks returned by [`quantile_indices`] into the quantile.
fn interpolate<T: ToPrimitive>(
    lower: T,
    upper: T,
    fraction: f64,
    method: QuantileMethod,
) -> Option<f64> {
    let lower = lower.to_f64()?;
    match method {
        QuantileMethod::Midpoint => Some(midpoint_interpol(lower, upper.to_f64()?)),
        QuantileMethod::Linear => Some(linear_interpol(lower, upper.to_f64()?, fraction)),
        _ => Some(lower),
    }
}

/// Computes `quantile` of `n_nonnull` values, where `order_stats(ranks, out)` writes the values at
/// the given ascending ranks of the sorted values to `out`.
fn quantile_with<T: ToPrimitive + Copy + Default>(
    n_nonnull: usize,
    quantile: f64,
    method: QuantileMethod,
    order_stats: impl FnOnce(&[usize], &mut [T]),
) -> Option<f64> {
    if n_nonnull == 0 {
        return None;
    }
    let (lower, upper, fraction) = quantile_indices(n_nonnull, quantile, method);
    let k = if lower == upper { 1 } else { 2 };
    let mut vals = [T::default(); 2];
    order_stats(&[lower, upper][..k], &mut vals[..k]);
    interpolate(vals[0], vals[k - 1], fraction, method)
}

/// [`quantile_with`] for several quantiles, fetching all the order statistics they need at once.
fn quantiles_with<T: ToPrimitive + Copy + Default>(
    n_nonnull: usize,
    quantiles: &[f64],
    method: QuantileMethod,
    order_stats: impl FnOnce(&[usize], &mut [T]),
) -> Vec<Option<f64>> {
    if n_nonnull == 0 {
        return vec![None; quantiles.len()];
    }
    let indices: Vec<_> = quantiles
        .iter()
        .map(|&q| quantile_indices(n_nonnull, q, method))
        .collect();
    let mut ranks: Vec<usize> = indices.iter().flat_map(|&(l, u, _)| [l, u]).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let mut vals = vec![T::default(); ranks.len()];
    order_stats(&ranks, &mut vals);
    let at = |r| vals[ranks.binary_search(&r).unwrap()];
    indices
        .iter()
        .map(|&(lower, upper, fraction)| interpolate(at(lower), at(upper), fraction, method))
        .collect()
}

/// The `quantile` of `vals`, non-null values in any order, which it reorders on the calling
/// thread, since group-by already runs its groups in parallel.
pub(crate) fn quantile_of_nonnull<T: ToPrimitive + TotalOrd + Copy + Default>(
    vals: &mut [T],
    quantile: f64,
    method: QuantileMethod,
) -> Option<f64> {
    quantile_with(vals.len(), quantile, method, |ranks, out| {
        select_many_seq_by(vals, ranks, out, TotalOrd::tot_cmp)
    })
}

/// Reads the values at the ascending `ranks` of the non-null values of `ca` into `out`, given that
/// `ca` is flagged sorted (descending if `descending`), which keeps all its nulls at one end.
fn sorted_order_stats<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    descending: bool,
    ranks: &[usize],
    out: &mut [T::Native],
) {
    let n_nonnull = ca.len() - ca.null_count();
    let start = ca.first_non_null().unwrap_or(0);
    debug_assert!(ranks.last().is_none_or(|&r| r < n_nonnull));

    // Visit the ranks in ascending row order, so a single pass over the chunks finds them.
    let mut chunks = ca.downcast_iter();
    let mut chunk = chunks.next().unwrap();
    let mut chunk_start = 0;
    for j in 0..ranks.len() {
        let (i, idx) = if descending {
            let i = ranks.len() - 1 - j;
            (i, start + n_nonnull - 1 - ranks[i])
        } else {
            (j, start + ranks[j])
        };
        while idx >= chunk_start + chunk.len() {
            chunk_start += chunk.len();
            chunk = chunks.next().unwrap();
        }
        out[i] = chunk.value(idx - chunk_start);
    }
}

/// The `quantile` of `ca`, which is sorted (descending if `descending`) with its nulls at one end.
pub(crate) fn sorted_quantile<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    descending: bool,
    quantile: f64,
    method: QuantileMethod,
) -> Option<f64> {
    let n_nonnull = ca.len() - ca.null_count();
    quantile_with(n_nonnull, quantile, method, |ranks, out| {
        sorted_order_stats(ca, descending, ranks, out)
    })
}

/// The non-null values of `arr`, filtered in parallel blocks.
fn filter_in_blocks<T: NativeType>(arr: &PrimitiveArray<T>) -> Vec<Vec<T>> {
    let validity = arr.validity().unwrap();
    (0..arr.len().div_ceil(SELECT_BLOCK_LEN))
        .into_par_iter()
        .map(|block| {
            let start = block * SELECT_BLOCK_LEN;
            let len = SELECT_BLOCK_LEN.min(arr.len() - start);
            let values = &arr.values()[start..start + len];
            filter_values(values, &validity.clone().sliced(start, len))
        })
        .collect()
}

/// Writes the values at the strictly ascending `ranks` of the sorted non-null values of `ca` to
/// `out`, reading them directly if `ca` is flagged sorted and selecting them otherwise.
pub fn order_statistics<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    ranks: &[usize],
    out: &mut [T::Native],
) {
    if ranks.is_empty() {
        return;
    }
    let n_valid = ca.len() - ca.null_count();
    match ca.is_sorted_flag() {
        // Entering the pool costs a thread wake-up, more than a small selection. On one thread,
        // copying and selecting sequentially beats classifying every value against a sampled
        // bracket.
        IsSorted::Not if n_valid < SELECT_MIN_PAR_LEN || RAYON.current_num_threads() == 1 => {
            select_many_seq_by(&mut ca.to_vec_non_null(), ranks, out, TotalOrd::tot_cmp)
        },
        IsSorted::Not => RAYON.install(|| {
            // Mostly-null chunks are filtered first: their copy is small, while masking them
            // would make the selection's sampling reject most values.
            let filtered: Vec<Option<Vec<Vec<T::Native>>>> = ca
                .downcast_iter()
                .map(|arr| (2 * arr.null_count() > arr.len()).then(|| filter_in_blocks(arr)))
                .collect();
            let mut chunks: Vec<MaskedSlice<'_, T::Native>> = Vec::new();
            for (arr, filtered) in ca.downcast_iter().zip(&filtered) {
                match filtered {
                    Some(blocks) => chunks.extend(blocks.iter().map(|values| MaskedSlice {
                        values,
                        validity: None,
                    })),
                    None => {
                        let validity = arr.validity().filter(|v| v.unset_bits() > 0).map(|v| {
                            let (bytes, offset, _) = v.as_slice();
                            (bytes, offset)
                        });
                        chunks.push(MaskedSlice {
                            values: arr.values(),
                            validity,
                        });
                    },
                }
            }
            select_many_by(&chunks, n_valid, ranks, out, TotalOrd::tot_cmp)
        }),
        sorted => sorted_order_stats(ca, sorted == IsSorted::Descending, ranks, out),
    }
}

/// The `quantile` of `ca`, after validating it.
fn ca_quantile<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    quantile: f64,
    method: QuantileMethod,
) -> PolarsResult<Option<f64>> {
    check_quantiles(&[quantile])?;
    let n_nonnull = ca.len() - ca.null_count();
    Ok(quantile_with(n_nonnull, quantile, method, |ranks, out| {
        order_statistics(ca, ranks, out)
    }))
}

/// The `quantiles` of `ca`, after validating them.
fn ca_quantiles<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    quantiles: &[f64],
    method: QuantileMethod,
) -> PolarsResult<Vec<Option<f64>>> {
    check_quantiles(quantiles)?;
    let n_nonnull = ca.len() - ca.null_count();
    Ok(quantiles_with(
        n_nonnull,
        quantiles,
        method,
        |ranks, out| order_statistics(ca, ranks, out),
    ))
}

/// The `quantiles` of the non-null values of `ca` that are not NaN, after validating them, or
/// `None` if there are no such values.
pub fn quantiles_without_nan(
    ca: &Float64Chunked,
    quantiles: &[f64],
    method: QuantileMethod,
) -> PolarsResult<Option<Vec<f64>>> {
    check_quantiles(quantiles)?;
    let n_nan: usize = ca
        .downcast_iter()
        .map(|arr| arr.non_null_values_iter().filter(|v| v.is_nan()).count())
        .sum();
    // NaN orders above every other value, so the lowest ranks are exactly the values without NaN.
    let n = ca.len() - ca.null_count() - n_nan;
    if n == 0 {
        return Ok(None);
    }
    let qs = quantiles_with(n, quantiles, method, |ranks, out| {
        order_statistics(ca, ranks, out)
    });
    Ok(Some(qs.into_iter().map(Option::unwrap).collect()))
}

impl<T> ChunkQuantile<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
{
    fn quantile(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Option<f64>> {
        ca_quantile(self, quantile, method)
    }

    fn quantiles(
        &self,
        quantiles: &[f64],
        method: QuantileMethod,
    ) -> PolarsResult<Vec<Option<f64>>> {
        ca_quantiles(self, quantiles, method)
    }

    fn median(&self) -> Option<f64> {
        self.quantile(0.5, QuantileMethod::Linear).unwrap() // unwrap fine since quantile in range
    }
}

macro_rules! impl_chunk_quantile_for_float_chunked {
    ($T:ty, $CA:ty) => {
        impl ChunkQuantile<$T> for $CA {
            fn quantile(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Option<$T>> {
                Ok(ca_quantile(self, quantile, method)?.map(AsPrimitive::as_))
            }

            fn quantiles(
                &self,
                quantiles: &[f64],
                method: QuantileMethod,
            ) -> PolarsResult<Vec<Option<$T>>> {
                Ok(ca_quantiles(self, quantiles, method)?
                    .into_iter()
                    .map(|opt| opt.map(AsPrimitive::as_))
                    .collect())
            }

            fn median(&self) -> Option<$T> {
                self.quantile(0.5, QuantileMethod::Linear).unwrap() // unwrap fine since quantile in range
            }
        }
    };
}

#[cfg(feature = "dtype-f16")]
impl_chunk_quantile_for_float_chunked!(pf16, Float16Chunked);
impl_chunk_quantile_for_float_chunked!(f32, Float32Chunked);
impl_chunk_quantile_for_float_chunked!(f64, Float64Chunked);

impl ChunkQuantile<String> for StringChunked {}
impl ChunkQuantile<Series> for ListChunked {}
#[cfg(feature = "dtype-array")]
impl ChunkQuantile<Series> for ArrayChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkQuantile<Series> for ObjectChunked<T> {}
impl ChunkQuantile<bool> for BooleanChunked {}
