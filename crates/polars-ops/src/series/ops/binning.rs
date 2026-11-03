use polars_arrow::array::Array;
use polars_arrow::compute::concatenate::concatenate_validities;
use polars_core::chunked_array::ops::binning::{FractionSpec, IntervalSpec};
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::series::IsSorted;
use polars_core::{
    with_match_physical_integer_polars_type, with_match_physical_numeric_polars_type,
};
use polars_utils::total_ord::{TotalEq, TotalOrd};
use rayon::prelude::*;

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

/// Assign every element the bin its rank falls in -- its 0-based position within the
/// non-null values in sorted order -- and null wherever `s` is null.
///
/// `cum_bin_sizes[i]` is the number of elements in bins `0..=i`.
fn bins_from_rank_cuts(s: &Series, sort_idx: &IdxCa, cum_bin_sizes: &[IdxSize]) -> IdxCa {
    let mut out = vec![0 as IdxSize; s.len()];
    let mut rank: IdxSize = 0;
    let mut bin = 0;
    for arr in sort_idx.downcast_iter() {
        for i in arr.values_iter() {
            while bin < cum_bin_sizes.len() && cum_bin_sizes[bin] <= rank {
                bin += 1;
            }
            out[*i as usize] = bin as IdxSize;
            rank += 1;
        }
    }
    IdxCa::from_vec_validity(s.name().clone(), out, concatenate_validities(s.chunks()))
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

/// Whether `dtype` sorts like its numeric physical type, so that its order statistics can be
/// selected on the physical values.
fn sorts_as_physical_numeric(dtype: &DataType) -> bool {
    dtype.is_numeric() || dtype.is_temporal()
}

/// The value at each position within the sorted non-null values of `ca`, or `None` past them as a
/// trailing empty bin requires.
fn select_at_sorted_positions<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    positions: &[IdxSize],
) -> Vec<Option<T::Native>> {
    let n_nonnull = ca.len() - ca.null_count();
    let mut ranks: Vec<usize> = positions
        .iter()
        .map(|&p| p as usize)
        .filter(|&p| p < n_nonnull)
        .collect();
    ranks.sort_unstable();
    ranks.dedup();
    let mut vals = vec![T::Native::default(); ranks.len()];
    order_statistics(ca, &ranks, &mut vals);
    positions
        .iter()
        .map(|&p| ranks.binary_search(&(p as usize)).ok().map(|i| vals[i]))
        .collect()
}

/// [`bins_from_rank_cuts`] without a sort, given the cumulative bin sizes `cuts` and the value at
/// each cut rank as `boundaries` (`None` past the non-null values).
fn bins_from_rank_cuts_select<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    cuts: &[IdxSize],
    boundaries: &[Option<T::Native>],
) -> IdxCa {
    // A value that equals no boundary lands in the bin after every boundary below it. The values
    // equal to a boundary occupy a contiguous run of ranks, which a stable sort hands out in input
    // order; counting them gives each its rank, and so its bin.
    //
    // The cuts ascend (explicit fractions are validated as sorted), so those past the non-null
    // values, which no rank reaches and which have no boundary, come last.
    let n_valid = boundaries.iter().take_while(|b| b.is_some()).count();
    let cuts = &cuts[..n_valid];
    let boundaries: Vec<T::Native> = boundaries[..n_valid].iter().map(|b| b.unwrap()).collect();

    // `distinct[m]` is a distinct boundary value, and `first_cut[m]` the first cut with it.
    let mut distinct: Vec<T::Native> = Vec::new();
    let mut first_cut: Vec<usize> = Vec::new();
    for (j, b) in boundaries.iter().enumerate() {
        if distinct.last().is_none_or(|d| d.tot_ne(b)) {
            distinct.push(*b);
            first_cut.push(j);
        }
    }
    first_cut.push(n_valid);

    // Index of the first distinct boundary not less than `v`, and whether it equals `v`.
    let locate = |v: &T::Native| {
        let m = distinct.partition_point(|d| d.tot_lt(v));
        (m, m < distinct.len() && distinct[m].tot_eq(v))
    };

    // Both passes run in parallel over ranges of rows, which may span chunks. Pass 1 stores each
    // row's located boundary as `2 * m + equal` in `out` (m <= k <= the number of cuts, which
    // `bin_at_positions` keeps below `IdxSize::MAX / 2`) and counts per range; the counts give each range the rank of its first value
    // equal to every boundary. A range is at least as long as there are boundaries, so the
    // per-range counts and starting ranks (3k + 1 per range) take at most about three times the
    // memory of `out`.
    let k = distinct.len();
    let range_len = (1 << 16).max(k);
    let arrs: Vec<_> = ca.downcast_iter().collect();
    let mut chunk_starts = Vec::with_capacity(arrs.len() + 1);
    chunk_starts.push(0);
    for arr in &arrs {
        chunk_starts.push(chunk_starts.last().unwrap() + arr.len());
    }
    let mut out = vec![0 as IdxSize; ca.len()];
    let mut parts: Vec<&mut [IdxSize]> = out.chunks_mut(range_len).collect();

    let counts: Vec<(Vec<IdxSize>, Vec<IdxSize>)> = RAYON.install(|| {
        parts
            .par_iter_mut()
            .enumerate()
            .map(|(r, part)| {
                let mut n_between = vec![0 as IdxSize; k + 1];
                let mut n_equal = vec![0 as IdxSize; k];
                let (lo, hi) = (r * range_len, r * range_len + part.len());
                let mut c = chunk_starts.partition_point(|&start| start <= lo) - 1;
                let mut row = lo;
                while row < hi {
                    let (arr, chunk_start) = (arrs[c], chunk_starts[c]);
                    let end = hi.min(chunk_starts[c + 1]);
                    let values = &arr.values()[row - chunk_start..end - chunk_start];
                    let out_piece = &mut part[row - lo..end - lo];
                    for (i, (v, o)) in values.iter().zip(out_piece.iter_mut()).enumerate() {
                        if arr.is_valid(row - chunk_start + i) {
                            let (m, equal) = locate(v);
                            *o = (2 * m + equal as usize) as IdxSize;
                            if equal {
                                n_equal[m] += 1;
                            } else {
                                n_between[m] += 1;
                            }
                        }
                    }
                    row = end;
                    c += 1;
                }
                (n_between, n_equal)
            })
            .collect()
    });

    // Turn each range's tie counts into the rank its first value equal to each boundary takes.
    let mut next_rank = vec![0 as IdxSize; k];
    let mut below: IdxSize = 0;
    for m in 0..k {
        below += counts
            .iter()
            .map(|(n_between, _)| n_between[m])
            .sum::<IdxSize>();
        next_rank[m] = below;
        below += counts
            .iter()
            .map(|(_, n_equal)| n_equal[m])
            .sum::<IdxSize>();
    }
    let mut starts: Vec<Vec<IdxSize>> = Vec::with_capacity(counts.len());
    for (_, n_equal) in &counts {
        starts.push(next_rank.clone());
        for (r, n) in next_rank.iter_mut().zip(n_equal) {
            *r += n;
        }
    }

    // Null rows still hold 0, which decodes as "below boundary 0, not equal": it takes no tie
    // rank, and the validity masks the bin.
    RAYON.install(|| {
        parts
            .into_par_iter()
            .zip(starts.into_par_iter())
            .for_each(|(part, mut next_rank)| {
                for o in part.iter_mut() {
                    let (m, equal) = (*o as usize / 2, *o % 2 == 1);
                    let bin = if equal {
                        let rank = next_rank[m];
                        next_rank[m] += 1;
                        let tied_cuts = &cuts[first_cut[m]..first_cut[m + 1]];
                        first_cut[m] + tied_cuts.partition_point(|&c| c <= rank)
                    } else {
                        first_cut[m]
                    };
                    *o = bin as IdxSize;
                }
            })
    });
    IdxCa::from_vec_validity(ca.name().clone(), out, concatenate_validities(ca.chunks()))
}

/// [`bins_from_rank_cuts`] for `ca` flagged sorted ascending, whose non-null rows are already in
/// rank order, so each run of rows between two cuts fills one bin.
fn bins_from_rank_cuts_sorted<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    cuts: &[IdxSize],
) -> IdxCa {
    let n_nonnull = ca.len() - ca.null_count();
    let start = ca.first_non_null().unwrap_or(0);
    let mut out = vec![0 as IdxSize; ca.len()];
    let ranked = &mut out[start..start + n_nonnull];
    let ends = cuts.iter().map(|&c| (c as usize).min(n_nonnull));
    let mut lo = 0;
    for (bin, hi) in ends.chain([n_nonnull]).enumerate() {
        ranked[lo..hi].fill(bin as IdxSize);
        lo = hi;
    }
    IdxCa::from_vec_validity(ca.name().clone(), out, concatenate_validities(ca.chunks()))
}

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

/// Equal-width breakpoints `min + (i + 1)/n_bins * (max - min)` for `0 <= i < n_bins - 1`,
/// in the input dtype, or `None` when the column has no usable `min`/`max`.
///
/// Floats go through `f64` and are narrowed back afterwards; integers and `Decimal` (via
/// its `Int128` physical) derive representable thresholds in their own width.
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
        let breaks: Vec<f64> = if min == max {
            // Special case so that we match the integer path -- we would round otherwise
            vec![min; n_breaks]
        } else {
            (1..=n_breaks)
                .map(|i| {
                    let t = i as f64 / n_bins as f64;
                    min * (1.0 - t) + max * t
                })
                .collect()
        };
        return Float64Chunked::from_vec(s.name().clone(), breaks)
            .into_series()
            .cast(dtype)
            .map(Some);
    }

    let phys = s.to_physical_repr();
    let phys: &Series = phys.as_ref();
    let breaks = with_match_physical_integer_polars_type!(phys.dtype(), |$T| {
        type N = <$T as PolarsNumericType>::Native;
        let ca: &ChunkedArray<$T> = phys.as_ref().as_ref();
        let Some((min, max)) = ca.min_max() else {
            return Ok(None);
        };
        // we are exact for all signed/unsigned integer types
        let span = (max as i128).wrapping_sub(min as i128) as u128;
        let breaks: Vec<N> = uniform_threshold_offsets(span, n_bins, right_closed)
            .map(|offset| (min as i128).wrapping_add(offset as i128) as N)
            .collect();
        ChunkedArray::<$T>::from_vec(s.name().clone(), breaks).into_series()
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

/// Cumulative sizes of `n_bins` bins of near-equal size.
///
/// Bin `i` receives `k + 1` elements while `i < len % n_bins` and `k` afterwards, so the
/// earlier bins are the larger ones: 14 elements over 4 bins gives 4 + 4 + 3 + 3.
fn cum_bin_sizes_uniform(non_null_len: usize, n_bins: usize) -> Vec<IdxSize> {
    let k = non_null_len / n_bins;
    let r = non_null_len % n_bins;
    (1..n_bins).map(|i| (i * k + i.min(r)) as IdxSize).collect()
}

/// Cumulative bin sizes for explicit cumulative fractions: `round(f * len)`, rounding half
/// away from zero so that, as in [`cum_bin_sizes_uniform`], earlier bins are the larger.
fn cum_bin_sizes_fractions(non_null_len: usize, fractions: &[f64]) -> Vec<IdxSize> {
    fractions
        .iter()
        .map(|f| (f * non_null_len as f64).round() as IdxSize)
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
            let physical = fcats.physical();
            let dtype = DataType::from_frozen_categories(fcats);
            with_match_categorical_physical_type!(physical, |$C| {
                CategoricalChunked::<$C>::from_str_iter(
                    PlSmallStr::EMPTY,
                    dtype,
                    bin_idx
                        .iter()
                        .map(|opt| opt.map(|i| labels[i as usize].as_str())),
                )?
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
        Some(right_closed),
    )
}

pub fn bin_ranks(
    s: &Series,
    spec: &FractionSpec,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    let cum_bin_sizes = match spec {
        FractionSpec::Explicit(fractions) => cum_bin_sizes_fractions(non_null_len, fractions),
        FractionSpec::Count(n_bins) => cum_bin_sizes_uniform(non_null_len, n_bins.get()),
    };
    bin_at_positions(
        s,
        &cum_bin_sizes,
        spec.n_bins(),
        labels,
        include_intervals,
        None,
    )
}

/// Shared tail of the four position-derived forms.
///
/// `right_closed` is `Some` for quantile binning, which assigns bins by comparing values
/// against the gathered breakpoints, and `None` for rank binning, which instead compares
/// each row's ordinal rank against the cut positions. The rank form therefore splits ties
/// across adjacent bins, which is the whole point of it -- and never needs the boundary
/// values unless they are actually reported.
///
/// Either way `positions` indexes into the non-null values in sorted order, but the two
/// forms arrive at it differently: for quantile binning it is the breakpoint's own sorted
/// position, while for rank binning it is the cumulative bin sizes -- the rank at which
/// each bin starts, whose value happens to be that bin's left boundary.
fn bin_at_positions(
    s: &Series,
    positions: &[IdxSize],
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: Option<bool>,
) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    if non_null_len == 0 {
        return empty_bins(s, n_bins, labels, include_intervals);
    }

    // `bins_from_rank_cuts_select` packs a boundary index and a flag into one `IdxSize`.
    if sorts_as_physical_numeric(s.dtype()) && positions.len() < IdxSize::MAX as usize / 2 {
        let phys = s.to_physical_repr();
        let (breaks, rank_bins) = with_match_physical_numeric_polars_type!(phys.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = phys.as_ref().as_ref().as_ref();
            let breaks = select_at_sorted_positions(ca, positions);
            let rank_bins = right_closed.is_none().then(|| {
                if ca.is_sorted_flag() == IsSorted::Ascending {
                    bins_from_rank_cuts_sorted(ca, positions)
                } else {
                    bins_from_rank_cuts_select(ca, positions, &breaks)
                }
            });
            let breaks: ChunkedArray<$T> = breaks.into_iter().collect_ca(s.name().clone());
            (breaks.into_series(), rank_bins)
        });
        // SAFETY: the values came out of the column's own physical values.
        let breaks = unsafe { breaks.from_physical_unchecked(s.dtype()) }?;
        let bin_idx = match (right_closed, rank_bins) {
            (Some(right_closed), _) => bins_from_breaks(s, &breaks, right_closed)?,
            (None, Some(rank_bins)) => rank_bins,
            (None, None) => unreachable!(),
        };
        let breaks = include_intervals.then_some(breaks);
        return finish_bins(s.name().clone(), &bin_idx, n_bins, breaks.as_ref(), labels);
    }

    // Rank binning may split ties, so the stable sort preserves their input order.
    let sort_idx = s
        .arg_sort(SortOptions {
            descending: false,
            nulls_last: true,
            maintain_order: true,
            ..Default::default()
        })
        .slice(0, non_null_len);

    // `finish_bins` reads `Some` as "emit the struct", so each arm keeps the boundaries
    // only when they are actually reported -- the value-based path needs them to bin at
    // all, the rank-based one only to report them.
    let (bin_idx, breaks) = match right_closed {
        Some(right_closed) => {
            let breaks = gather_at_sorted_positions(s, &sort_idx, positions)?;
            let bin_idx = bins_from_breaks(s, &breaks, right_closed)?;
            (bin_idx, include_intervals.then_some(breaks))
        },
        None => {
            let bin_idx = bins_from_rank_cuts(s, &sort_idx, positions);
            let breaks = include_intervals
                .then(|| gather_at_sorted_positions(s, &sort_idx, positions))
                .transpose()?;
            (bin_idx, breaks)
        },
    };

    finish_bins(s.name().clone(), &bin_idx, n_bins, breaks.as_ref(), labels)
}
