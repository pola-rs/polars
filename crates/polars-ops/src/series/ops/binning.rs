use arrow::compute::concatenate::concatenate_validities;
use polars_core::chunked_array::ops::binning::{FractionSpec, IntervalSpec};
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;

use crate::series::{SearchSortedSide, search_sorted};

/// Assign every element of `s` to a bin delimited by `breaks`.
///
/// `breaks` must be ascending, free of nulls, and of exactly `s`'s dtype. The result is
/// in `0..=breaks.len()`, and null wherever `s` is null.
fn bins_from_breaks(s: &Series, breaks: &Series, right_closed: bool) -> PolarsResult<IdxCa> {
    polars_ensure!(
        s.dtype() == breaks.dtype(),
        ComputeError: "binning expects the breakpoints to be of the input dtype `{}`, got `{}`",
        s.dtype(), breaks.dtype()
    );

    let side = if right_closed {
        SearchSortedSide::Left
    } else {
        SearchSortedSide::Right
    };

    // `search_sorted` gives a null needle the position where a null would sort rather
    // than a null, so the input validity is reattached here.
    Ok(search_sorted(breaks, s, side, false)?
        .with_name(s.name().clone())
        .with_validity(concatenate_validities(s.chunks())))
}

/// Gather the value at each given position within the non-null values in sorted order.
///
/// Positions at or past the non-null count yield null, as required by a trailing empty bin.
fn gather_at_sorted_positions(
    s: &Series,
    sort_idx: &IdxCa,
    positions: &[IdxSize],
) -> PolarsResult<Series> {
    let non_null_len = sort_idx.len() as IdxSize;

    let wanted: IdxCa = positions
        .iter()
        .map(|p| (*p < non_null_len).then_some(*p))
        .collect_ca(PlSmallStr::EMPTY);

    let phys_idx = sort_idx.take(&wanted)?;
    s.take(&phys_idx)
}

/// Measure and apply an offset in the unsigned domain so full-width signed spans do not
/// overflow.
trait BinWidth: Copy {
    /// `self - min`, non-negative because `self` is the column's max.
    fn span_from(self, min: Self) -> u128;
    /// `self + offset`, where `offset` lies within the span and so cannot overflow, nor
    /// lose anything on the way back down to this width.
    fn offset_by(self, offset: u128) -> Self;
}

macro_rules! impl_bin_width {
    (signed: $($t:ty),* $(,)?) => {
        $(
            impl BinWidth for $t {
                fn span_from(self, min: Self) -> u128 {
                    self.cast_unsigned().wrapping_sub(min.cast_unsigned()).into()
                }
                fn offset_by(self, offset: u128) -> Self {
                    self.cast_unsigned().wrapping_add(offset as _).cast_signed()
                }
            }
        )*
    };
    (unsigned: $($t:ty),* $(,)?) => {
        $(
            impl BinWidth for $t {
                fn span_from(self, min: Self) -> u128 {
                    self.wrapping_sub(min).into()
                }
                fn offset_by(self, offset: u128) -> Self {
                    self.wrapping_add(offset as _)
                }
            }
        )*
    };
}

impl_bin_width!(signed: i8, i16, i32, i64, i128);
impl_bin_width!(unsigned: u8, u16, u32, u64, u128);

/// Offsets from `min` of the `n_bins - 1` thresholds of equal-width bins over a span.
///
/// The exact breakpoint `i * span / n_bins` need not be an integer. For left-closed bins
/// the equivalent threshold is its ceiling; for right-closed bins it is its floor. So
/// carry the division as a quotient plus a running remainder: `offset` is the floor and
/// `error` is the numerator of the fraction still owed, which is non-zero exactly when
/// the ceiling is one higher.
fn uniform_threshold_offsets(
    span: u128,
    n_bins: usize,
    right_closed: bool,
) -> impl Iterator<Item = u128> {
    let n = n_bins as u128;
    let (step, remainder) = (span / n, span % n);
    let mut offset = 0;
    let mut error = 0;

    (1..n_bins).map(move |_| {
        offset += step;
        error += remainder;
        if error >= n {
            offset += 1;
            error -= n;
        }

        let round_up = !right_closed && error != 0;
        offset + u128::from(round_up)
    })
}

/// Representable thresholds for equal-width bins over `[min, max]`, in `N`'s own width.
fn uniform_integer_thresholds<N: BinWidth>(
    min: N,
    max: N,
    n_bins: usize,
    right_closed: bool,
) -> Vec<N> {
    uniform_threshold_offsets(max.span_from(min), n_bins, right_closed)
        .map(|offset| min.offset_by(offset))
        .collect()
}

/// Equal-width breakpoints `min + (i + 1)/n_bins * (max - min)` for `0 <= i < n_bins - 1`,
/// in the input dtype, or `None` when the column has no usable `min`/`max`.
///
/// Floats go through `f64` and are narrowed back afterwards; integers and `Decimal` (via
/// its `Int128` physical) use [`uniform_integer_thresholds`].
fn uniform_interval_breaks(
    s: &Series,
    n_bins: usize,
    right_closed: bool,
) -> PolarsResult<Option<Series>> {
    let n_breaks = n_bins - 1;
    let dtype = s.dtype();

    if dtype.is_float() {
        let f = s.cast(&DataType::Float64)?;
        let Some((min, max)) = f.f64()?.min_max() else {
            return Ok(None);
        };
        let breaks: Vec<f64> = (1..=n_breaks)
            .map(|i| {
                let t = i as f64 / n_bins as f64;
                min * (1.0 - t) + max * t
            })
            .collect();
        return Float64Chunked::from_vec(s.name().clone(), breaks)
            .into_series()
            .cast(dtype)
            .map(Some);
    }

    let phys = s.to_physical_repr();
    let phys: &Series = phys.as_ref();
    let breaks = with_match_physical_integer_polars_type!(phys.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = phys.as_ref().as_ref();
        let Some((min, max)) = ca.min_max() else {
            return Ok(None);
        };
        ChunkedArray::<$T>::from_vec(
            s.name().clone(),
            uniform_integer_thresholds(min, max, n_bins, right_closed),
        )
        .into_series()
    });

    // Reattach the logical dtype: a no-op for plain integers, and restores the precision
    // and scale for `Decimal`, whose breakpoints were computed on its i128 mantissas.
    // SAFETY: the values came out of the column's own physical range.
    unsafe { breaks.from_physical_unchecked(dtype) }.map(Some)
}

/// Breakpoint positions for explicit quantile probabilities: `floor(q * (len - 1))`,
/// matching `QuantileMethod::Lower`.
fn quantile_break_positions(non_null_len: usize, probs: &[f64]) -> Vec<IdxSize> {
    if non_null_len == 0 {
        return vec![0; probs.len()];
    }
    let span = (non_null_len - 1) as f64;
    probs
        .iter()
        .map(|q| (span * q).floor() as IdxSize)
        .collect()
}

/// Breakpoint positions for `n_bins` equiprobable bins. Integer arithmetic avoids
/// off-by-one errors from expanding the cuts to inexact `f64` probabilities.
fn quantile_break_positions_uniform(non_null_len: usize, n_bins: usize) -> Vec<IdxSize> {
    let n_breaks = n_bins - 1;
    if non_null_len == 0 {
        return vec![0; n_breaks];
    }
    let span = non_null_len - 1;
    (1..=n_breaks)
        .map(|i| ((i * span) / n_bins) as IdxSize)
        .collect()
}

/// Turn a column of bin indices into the final output.
///
/// Without labels the bins are returned as `UInt32`, with labels as an `Enum` over them.
/// When `breaks` is given the result is instead a struct of `bin`, `left` and `right`,
/// where `left` is null for the first bin and `right` null for the last. `breaks` holds
/// the `n_bins - 1` boundary values; for rank-based binning these are the gathered
/// boundary *values*, not the rank positions.
fn finish_bins(
    out_name: PlSmallStr,
    bin_idx: &IdxCa,
    n_bins: usize,
    breaks: Option<&Series>,
    labels: Option<&[PlSmallStr]>,
) -> PolarsResult<Series> {
    let bin = match labels {
        None => bin_idx.cast(&DataType::UInt32)?,
        Some(labels) => {
            polars_ensure!(
                labels.len() == n_bins,
                ShapeMismatch: "binning produces {} bins but got {} labels",
                n_bins, labels.len()
            );
            let fcats = FrozenCategories::new(labels.iter().map(|s| s.as_str()))?;
            let dtype = DataType::from_frozen_categories(fcats.clone());
            with_match_categorical_physical_type!(fcats.physical(), |$C| {
                let cats: Vec<<$C as PolarsCategoricalType>::Native> = bin_idx
                    .iter()
                    .map(|opt| {
                        <$C as PolarsCategoricalType>::Native::from_cat(
                            opt.unwrap_or(0) as CatSize,
                        )
                    })
                    .collect();
                let phys = ChunkedArray::<<$C as PolarsCategoricalType>::PolarsPhysical>::from_vec_validity(
                    PlSmallStr::EMPTY,
                    cats,
                    concatenate_validities(bin_idx.chunks()),
                );
                // SAFETY: every index is `< n_bins`, which is the number of frozen
                // categories, and the physical width was taken from `fcats` itself.
                unsafe {
                    CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(phys, dtype)
                }
                .into_series()
            })
        },
    };

    let Some(breaks) = breaks else {
        return Ok(bin.with_name(out_name));
    };
    debug_assert_eq!(breaks.len() + 1, n_bins);

    // `left[i]` is `(null ++ breaks)[bin_idx[i]]` and `right[i]` is
    // `(breaks ++ null)[bin_idx[i]]`. A null index gathers to null, so null inputs and
    // the open ends of the first and last bin all fall out of the gather for free.
    let null = Series::full_null(PlSmallStr::EMPTY, 1, breaks.dtype());
    let mut left_lookup = null.clone();
    left_lookup.append(breaks)?;
    let mut right_lookup = breaks.clone().with_name(PlSmallStr::EMPTY);
    right_lookup.append(&null)?;

    let bin = bin.with_name(PlSmallStr::from_static("bin"));
    let left = left_lookup
        .take(bin_idx)?
        .with_name(PlSmallStr::from_static("left"));
    let right = right_lookup
        .take(bin_idx)?
        .with_name(PlSmallStr::from_static("right"));

    Ok(
        StructChunked::from_series(out_name, bin_idx.len(), [&bin, &left, &right].into_iter())?
            .into_series(),
    )
}

/// Output for an input with no usable values, keeping the dtype and struct shape stable.
fn empty_bins(
    s: &Series,
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let breaks =
        include_intervals.then(|| Series::full_null(PlSmallStr::EMPTY, n_bins - 1, s.dtype()));
    let bin_idx = IdxCa::full_null(s.name().clone(), s.len());
    finish_bins(s.name().clone(), &bin_idx, n_bins, breaks.as_ref(), labels)
}

pub fn bin_intervals(
    s: &Series,
    spec: &IntervalSpec,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let breaks = match spec {
        // Both sides already share a dtype: the DSL to IR conversion casts the
        // breakpoints to the input, or wraps the input in a `Cast` to their supertype.
        // `bins_from_breaks` rejects a mismatch rather than papering over it here.
        IntervalSpec::Breaks(breaks) => Series::clone(breaks),
        IntervalSpec::Count(n_bins) => {
            let n_bins = n_bins.get();
            // No usable min/max: an empty or all-null column.
            let Some(breaks) = uniform_interval_breaks(s, n_bins, right_closed)? else {
                return empty_bins(s, n_bins, labels, include_intervals);
            };
            breaks
        },
    };

    let bin_idx = bins_from_breaks(s, &breaks, right_closed)?;
    finish_bins(
        s.name().clone(),
        &bin_idx,
        breaks.len() + 1,
        include_intervals.then_some(&breaks),
        labels,
    )
}

pub fn bin_quantiles(
    s: &Series,
    spec: &FractionSpec,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    let positions = match spec {
        FractionSpec::Explicit(probs) => quantile_break_positions(non_null_len, probs),
        FractionSpec::Count(n_bins) => quantile_break_positions_uniform(non_null_len, n_bins.get()),
    };
    bin_at_positions(
        s,
        &positions,
        spec.n_bins(),
        labels,
        include_intervals,
        right_closed,
    )
}

/// Shared tail of the position-derived forms: sort the non-null values once, gather the
/// boundary values sitting at the given positions, and bin against them.
fn bin_at_positions(
    s: &Series,
    positions: &[IdxSize],
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    if non_null_len == 0 {
        return empty_bins(s, n_bins, labels, include_intervals);
    }

    let sort_idx = s
        .arg_sort(SortOptions {
            descending: false,
            nulls_last: true,
            ..Default::default()
        })
        .slice(0, non_null_len);

    let breaks = gather_at_sorted_positions(s, &sort_idx, positions)?;
    let bin_idx = bins_from_breaks(s, &breaks, right_closed)?;

    finish_bins(
        s.name().clone(),
        &bin_idx,
        n_bins,
        include_intervals.then_some(&breaks),
        labels,
    )
}
