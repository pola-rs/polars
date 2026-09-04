use arrow::compute::concatenate::concatenate_validities;
use polars_core::chunked_array::ops::binning::IntervalSpec;
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
