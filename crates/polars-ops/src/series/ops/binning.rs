use std::ops::Range;

use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::compute::concatenate::concatenate_validities;
use polars_arrow::types::NativeType;
use polars_core::chunked_array::ops::binning::{
    FractionSpec, IntervalSpec, order_statistics_physical,
};
use polars_core::prelude::gather::check_bounds_ca;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::series::IsSorted;
use polars_core::{
    with_match_physical_integer_polars_type, with_match_physical_numeric_polars_type,
};
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
    let vals = order_statistics_physical(&ca.clone().into_series(), &ranks);
    let vals: &ChunkedArray<T> = vals.as_ref().as_ref();
    let vals = vals.cont_slice().unwrap();
    positions
        .iter()
        .map(|&p| ranks.binary_search(&(p as usize)).ok().map(|i| vals[i]))
        .collect()
}

/// The non-null breakpoints of rank binning, ascending with nulls last, grouped by value.
struct RankBreaks<N> {
    distinct: Vec<N>,
    /// The index of the first breakpoint equal to each distinct value, then the number of non-null
    /// breakpoints.
    first_cut: Vec<usize>,
}

impl<N: NumericNative> RankBreaks<N> {
    fn new(breaks: &[Option<N>]) -> PolarsResult<Self> {
        let n_valid = breaks.iter().take_while(|b| b.is_some()).count();
        // `bins_from_rank_ties_impl` packs a distinct index and a flag into one `IdxSize`.
        polars_ensure!(
            breaks[n_valid..].iter().all(Option::is_none) && n_valid < IdxSize::MAX as usize / 2,
            ComputeError: "rank binning expects at most {} breakpoints with nulls last",
            IdxSize::MAX / 2 - 1
        );
        let mut distinct: Vec<N> = Vec::new();
        let mut first_cut = Vec::new();
        for (j, b) in breaks[..n_valid].iter().map(|b| b.unwrap()).enumerate() {
            match distinct.last() {
                Some(d) if d.tot_eq(&b) => {},
                Some(d) if b.tot_lt(d) => {
                    polars_bail!(ComputeError: "rank binning expects ascending breakpoints")
                },
                _ => {
                    distinct.push(b);
                    first_cut.push(j);
                },
            }
        }
        first_cut.push(n_valid);
        Ok(Self {
            distinct,
            first_cut,
        })
    }

    fn n_valid(&self) -> usize {
        *self.first_cut.last().unwrap()
    }

    /// Rows per parallel range. All ranges but the last hold at least `k` rows, so their 3k + 1
    /// counters take at most about three times the memory of one `IdxSize` per row.
    fn range_len(&self) -> usize {
        (1 << 16).max(self.distinct.len())
    }

    /// Index of the first distinct breakpoint not less than `v`, and whether it equals `v`.
    fn locate(&self, v: &N) -> (usize, bool) {
        let m = self.distinct.partition_point(|d| d.tot_lt(v));
        (m, m < self.distinct.len() && self.distinct[m].tot_eq(v))
    }

    /// Locates every non-null row in `rows` of the column, passing `(row, m, equal)` to `f`, and
    /// counts how many values lie below each distinct breakpoint (and above the one before) and
    /// how many equal it.
    fn count(
        &self,
        chunks: &ChunkIndex<'_, N>,
        rows: Range<usize>,
        mut f: impl FnMut(usize, usize, bool),
    ) -> (Vec<IdxSize>, Vec<IdxSize>) {
        let k = self.distinct.len();
        let mut n_between = vec![0 as IdxSize; k + 1];
        let mut n_equal = vec![0 as IdxSize; k];
        chunks.for_each_valid(rows, |row, v| {
            let (m, equal) = self.locate(v);
            f(row, m, equal);
            if equal {
                n_equal[m] += 1;
            } else {
                n_between[m] += 1;
            }
        });
        (n_between, n_equal)
    }

    /// [`Self::count`] over all of `ca`, in parallel.
    fn count_all<T: PolarsNumericType<Native = N>>(
        &self,
        ca: &ChunkedArray<T>,
    ) -> (Vec<IdxSize>, Vec<IdxSize>) {
        let range_len = self.range_len();
        let chunks = ChunkIndex::new(ca);
        let counts: Vec<_> = RAYON.install(|| {
            (0..ca.len().div_ceil(range_len))
                .into_par_iter()
                .map(|r| {
                    let rows = r * range_len..ca.len().min((r + 1) * range_len);
                    self.count(&chunks, rows, |_, _, _| {})
                })
                .collect()
        });
        let k = self.distinct.len();
        let (mut n_between, mut n_equal) = (vec![0 as IdxSize; k + 1], vec![0 as IdxSize; k]);
        for (b, e) in &counts {
            n_between.iter_mut().zip(b).for_each(|(t, n)| *t += n);
            n_equal.iter_mut().zip(e).for_each(|(t, n)| *t += n);
        }
        (n_between, n_equal)
    }
}

/// The chunks of a column and the row each starts at, which locate a range of rows without walking
/// the chunks before it.
struct ChunkIndex<'a, N: NativeType> {
    arrs: Vec<&'a PrimitiveArray<N>>,
    starts: Vec<usize>,
}

impl<'a, N: NativeType> ChunkIndex<'a, N> {
    fn new<T: PolarsNumericType<Native = N>>(ca: &'a ChunkedArray<T>) -> Self {
        let arrs: Vec<_> = ca.downcast_iter().collect();
        let mut starts = Vec::with_capacity(arrs.len() + 1);
        starts.push(0);
        for arr in &arrs {
            starts.push(starts.last().unwrap() + arr.len());
        }
        Self { arrs, starts }
    }

    /// Calls `f(row, value)` for every non-null row in `rows`.
    fn for_each_valid(&self, rows: Range<usize>, mut f: impl FnMut(usize, &N)) {
        let mut c = self.starts.partition_point(|&start| start <= rows.start) - 1;
        let mut row = rows.start;
        while row < rows.end {
            let (arr, start) = (self.arrs[c], self.starts[c]);
            let end = rows.end.min(self.starts[c + 1]);
            for i in row - start..end - start {
                if arr.is_valid(i) {
                    f(start + i, &arr.values()[i]);
                }
            }
            row = end;
            c += 1;
        }
    }
}

/// The bin of each row of `ca` whose `m`-th distinct breakpoint has the ranks `cuts`, given
/// `first_rank`, which turns the per-range counts of [`RankBreaks::count`] into the rank of the
/// first value of `ca` equal to each distinct breakpoint.
///
/// A value that equals no breakpoint lands in the bin after every breakpoint below it. The values
/// equal to a breakpoint occupy a contiguous run of ranks, which a stable sort hands out in input
/// order; counting them gives each its rank, and so its bin.
fn bins_from_rank_ties_impl<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    rb: &RankBreaks<T::Native>,
    cuts: &[IdxSize],
    first_rank: impl FnOnce(&[(Vec<IdxSize>, Vec<IdxSize>)]) -> Vec<IdxSize>,
) -> PolarsResult<IdxCa> {
    // Pass 1 stores each row's located breakpoint as `2 * m + equal` in `out`, and counts per
    // range. Prefix sums of the counts then give, per range and breakpoint, the rank of the range's
    // first value equal to it.
    let range_len = rb.range_len();
    let mut out = vec![0 as IdxSize; ca.len()];
    let mut parts: Vec<&mut [IdxSize]> = out.chunks_mut(range_len).collect();
    let chunks = ChunkIndex::new(ca);
    let counts: Vec<(Vec<IdxSize>, Vec<IdxSize>)> = RAYON.install(|| {
        parts
            .par_iter_mut()
            .enumerate()
            .map(|(r, part)| {
                let lo = r * range_len;
                rb.count(&chunks, lo..lo + part.len(), |row, m, equal| {
                    part[row - lo] = (2 * m + equal as usize) as IdxSize
                })
            })
            .collect()
    });

    let mut next_rank = first_rank(&counts);
    let mut starts: Vec<Vec<IdxSize>> = Vec::with_capacity(counts.len());
    for (_, n_equal) in &counts {
        starts.push(next_rank.clone());
        for (r, n) in next_rank.iter_mut().zip(n_equal) {
            // A caller's tie rank can be anything; past this check, pass 2 stays in range.
            *r = r.checked_add(*n).ok_or_else(
                || polars_err!(ComputeError: "rank binning tie ranks overflow `IdxSize`"),
            )?;
        }
    }

    // Null rows still hold 0, which decodes as "below breakpoint 0, not equal": it takes no tie
    // rank, and the validity masks the bin.
    let first_cut = &rb.first_cut;
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
    Ok(IdxCa::from_vec_validity(
        ca.name().clone(),
        out,
        concatenate_validities(ca.chunks()),
    ))
}

/// [`bins_from_rank_cuts`] without a sort, given the cumulative bin sizes `cuts` and the value at
/// each cut rank as `boundaries` (`None` past the non-null values).
fn bins_from_rank_cuts_select<T: PolarsNumericType>(
    ca: &ChunkedArray<T>,
    cuts: &[IdxSize],
    boundaries: &[Option<T::Native>],
) -> PolarsResult<IdxCa> {
    let rb = RankBreaks::new(boundaries)?;
    bins_from_rank_ties_impl(ca, &rb, cuts, |counts| {
        let mut next_rank = vec![0 as IdxSize; rb.distinct.len()];
        let mut below: IdxSize = 0;
        for (m, r) in next_rank.iter_mut().enumerate() {
            below += counts.iter().map(|(b, _)| b[m]).sum::<IdxSize>();
            *r = below;
            below += counts.iter().map(|(_, e)| e[m]).sum::<IdxSize>();
        }
        next_rank
    })
}

/// The physical values of `s` and `breaks`, which rank binning on parts of a column compares.
fn rank_break_physicals(s: &Series, breaks: &Series) -> PolarsResult<(Series, Series)> {
    polars_ensure!(
        s.dtype() == breaks.dtype(),
        ComputeError: "binning expects the breakpoints to be of the input dtype `{}`, got `{}`",
        s.dtype(), breaks.dtype()
    );
    polars_ensure!(
        s.dtype().is_numeric() || s.dtype().is_temporal(),
        InvalidOperation: "rank binning of parts of a column is not supported for dtype `{}`",
        s.dtype()
    );
    Ok((
        s.to_physical_repr().into_owned(),
        breaks.to_physical_repr().into_owned(),
    ))
}

/// For each of the `bin_ranks` breakpoints `breaks`, ascending with nulls last, the number of
/// non-null values of `s` less than it and equal to it (zero for a null breakpoint).
///
/// Summing these over the parts of a column gives the global ranks [`bins_from_rank_ties`] needs.
pub fn bin_rank_break_counts(
    s: &Series,
    breaks: &Series,
) -> PolarsResult<(Vec<IdxSize>, Vec<IdxSize>)> {
    let (s, breaks) = rank_break_physicals(s, breaks)?;
    with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        let b: &ChunkedArray<$T> = breaks.as_ref().as_ref().as_ref();
        let rb = RankBreaks::new(&b.iter().collect::<Vec<_>>())?;
        let (n_between, n_equal) = rb.count_all(ca);
        let mut lt = vec![0 as IdxSize; b.len()];
        let mut eq = vec![0 as IdxSize; b.len()];
        let mut below: IdxSize = 0;
        for (m, w) in rb.first_cut.windows(2).enumerate() {
            below += n_between[m];
            lt[w[0]..w[1]].fill(below);
            eq[w[0]..w[1]].fill(n_equal[m]);
            below += n_equal[m];
        }
        Ok((lt, eq))
    })
}

/// The `bin_ranks` bins of the part `s` of a column whose breakpoints `breaks`, ascending with
/// nulls last, sit at the ranks `cuts`, where `tie_ranks[i]` is the rank within the column of
/// the first value of `s` equal to `breaks[i]`.
///
/// Entries of `cuts` and `tie_ranks` at null breakpoints are ignored.
pub fn bins_from_rank_ties(
    s: &Series,
    breaks: &Series,
    cuts: &[IdxSize],
    tie_ranks: &[IdxSize],
) -> PolarsResult<IdxCa> {
    polars_ensure!(
        cuts.len() == breaks.len() && tie_ranks.len() == breaks.len(),
        ComputeError: "rank binning expects one cut and tie rank per breakpoint, got {} breakpoints, {} cuts and {} tie ranks",
        breaks.len(), cuts.len(), tie_ranks.len()
    );
    let (phys, breaks) = rank_break_physicals(s, breaks)?;
    with_match_physical_numeric_polars_type!(phys.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = phys.as_ref().as_ref().as_ref();
        let b: &ChunkedArray<$T> = breaks.as_ref().as_ref().as_ref();
        let rb = RankBreaks::new(&b.iter().collect::<Vec<_>>())?;
        let cuts = &cuts[..rb.n_valid()];
        polars_ensure!(
            cuts.is_sorted(),
            ComputeError: "rank binning expects ascending cuts"
        );
        let bins = bins_from_rank_ties_impl(ca, &rb, cuts, |_| {
            rb.first_cut[..rb.distinct.len()].iter().map(|&j| tie_ranks[j]).collect()
        })?;
        Ok(bins.with_name(s.name().clone()))
    })
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

/// The `bin_intervals` breakpoints of `n_bins` equal-width bins, which depend only on the dtype,
/// min and max of `s`, or `None` when `s` has no non-null values.
//
// Floats go through `f64` and are narrowed back afterwards; integers and `Decimal` (via
// its `Int128` physical) derive representable thresholds in their own width.
/// Makes `breaks` non-decreasing and within `[min, max)` when right-closed or `(min, max]` when
/// left-closed, so that `min` and `max` land in the first and last bin.
///
/// The bounds are the neighbors of `max` and `min` in the float dtype `dtype`, so a break that
/// rounds onto an endpoint moves as little as possible. A NaN break, which the interpolation over
/// `[-inf, inf]` produces, repeats the break before it.
fn fix_float_breaks(breaks: &mut [f64], min: f64, max: f64, dtype: &DataType, right_closed: bool) {
    let (mut lo, hi) = if right_closed {
        (min, float_step(max, dtype, false))
    } else {
        (float_step(min, dtype, true), max)
    };
    for b in breaks.iter_mut() {
        *b = b.max(lo).min(hi);
        lo = *b;
    }
}

/// The neighbor of `x` toward `+inf` if `up`, else toward `-inf`, among the values of the float
/// dtype `dtype`, which holds `x`.
fn float_step(x: f64, dtype: &DataType, up: bool) -> f64 {
    match dtype {
        DataType::Float32 => {
            let x = x as f32;
            (if up { x.next_up() } else { x.next_down() }) as f64
        },
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            use polars_utils::float16::pf16;
            let x = pf16::from(x);
            f64::from(if up { x.next_up() } else { x.next_down() })
        },
        _ => {
            if up {
                x.next_up()
            } else {
                x.next_down()
            }
        },
    }
}

pub fn bin_interval_breaks(
    s: &Series,
    n_bins: usize,
    right_closed: bool,
) -> PolarsResult<Option<Series>> {
    polars_ensure!(n_bins > 0, ComputeError: "binning requires at least one bin");
    let dtype = s.dtype();
    polars_ensure!(
        dtype.is_numeric(),
        InvalidOperation: "`bin_intervals` requires a numeric input, got `{}`", dtype
    );
    let n_breaks = n_bins - 1;

    if dtype.is_float() {
        let f = s.cast(&DataType::Float64)?;
        let Some((min, max)) = f.f64()?.min_max() else {
            return Ok(None);
        };
        if min == max {
            // The bounds of `fix_float_breaks` cross without a span. This matches the integer path.
            return Ok(Some(
                Series::new(s.name().clone(), vec![min; n_breaks]).cast(dtype)?,
            ));
        }
        let lerp: Vec<f64> = (1..=n_breaks)
            .map(|i| {
                let t = i as f64 / n_bins as f64;
                min * (1.0 - t) + max * t
            })
            .collect();
        // Fix the breaks up after rounding them to the input dtype, which can undo a fixup in
        // `f64`. `min` and `max` come from the column, so they and the fixed breaks round-trip.
        let narrowed = Float64Chunked::from_vec(s.name().clone(), lerp)
            .into_series()
            .cast(dtype)?
            .cast(&DataType::Float64)?;
        let mut breaks: Vec<f64> = narrowed.f64()?.into_no_null_iter().collect();
        fix_float_breaks(&mut breaks, min, max, dtype, right_closed);
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

/// The binning output for the bin indices `bin_idx` in `0..n_bins`, labelled by the `n_bins`
/// `labels` and with intervals bounded by the `n_bins - 1` boundary values `breaks`.
//
// Without labels the bins are returned as `UInt32`, with labels as an `Enum` over them.
// When `breaks` is given the result is instead a struct of `bin`, `left` and `right`,
// where `left` is null for the first bin and `right` null for the last. For rank-based
// binning `breaks` are the gathered boundary *values*, not the rank positions.
pub fn finish_bins(
    out_name: PlSmallStr,
    bin_idx: &IdxCa,
    n_bins: usize,
    breaks: Option<&Series>,
    labels: Option<&[PlSmallStr]>,
) -> PolarsResult<Series> {
    polars_ensure!(n_bins > 0, ComputeError: "binning requires at least one bin");
    check_bounds_ca(bin_idx, n_bins.try_into().unwrap_or(IdxSize::MAX))?;

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
    polars_ensure!(
        breaks.len() + 1 == n_bins,
        ShapeMismatch: "binning produces {} bins but got {} breakpoints",
        n_bins, breaks.len()
    );

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
            let Some(breaks) = bin_interval_breaks(s, n_bins, right_closed)? else {
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

/// Sorted positions among the `non_null_len` non-null values of the `bin_quantiles` breakpoints.
pub fn bin_quantile_positions(non_null_len: usize, spec: &FractionSpec) -> Vec<IdxSize> {
    match spec {
        FractionSpec::Explicit(probs) => quantile_break_positions(non_null_len, probs),
        FractionSpec::Count(n_bins) => quantile_break_positions_uniform(non_null_len, n_bins.get()),
    }
}

pub fn bin_quantiles(
    s: &Series,
    spec: &FractionSpec,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let positions = bin_quantile_positions(s.len() - s.null_count(), spec);
    bin_at_positions(
        s,
        &positions,
        spec.n_bins(),
        labels,
        include_intervals,
        Some(right_closed),
    )
}

/// The `bin_ranks` cuts among `non_null_len` non-null values: bin `i` holds the ranks from
/// `cuts[i - 1]` up to but excluding `cuts[i]`.
pub fn bin_rank_cuts(non_null_len: usize, spec: &FractionSpec) -> Vec<IdxSize> {
    match spec {
        FractionSpec::Explicit(fractions) => cum_bin_sizes_fractions(non_null_len, fractions),
        FractionSpec::Count(n_bins) => cum_bin_sizes_uniform(non_null_len, n_bins.get()),
    }
}

pub fn bin_ranks(
    s: &Series,
    spec: &FractionSpec,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let cum_bin_sizes = bin_rank_cuts(s.len() - s.null_count(), spec);
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
/// across adjacent bins, which is the whole point of it.
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

    // Numeric and temporal values sort like their physical values, so their boundaries can be
    // selected on those. `bins_from_rank_cuts_select` packs a boundary index and a flag into one
    // `IdxSize`.
    let dtype = s.dtype();
    if (dtype.is_numeric() || dtype.is_temporal()) && positions.len() < IdxSize::MAX as usize / 2 {
        let phys = s.to_physical_repr();
        let (breaks, bin_idx) = with_match_physical_numeric_polars_type!(phys.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = phys.as_ref().as_ref().as_ref();
            let values = select_at_sorted_positions(ca, positions);
            let breaks: ChunkedArray<$T> = values.iter().copied().collect_ca(s.name().clone());
            // SAFETY: the values came out of the column's own physical values.
            let breaks = unsafe { breaks.into_series().from_physical_unchecked(dtype) }?;
            let bin_idx = match right_closed {
                Some(right_closed) => bins_from_breaks(s, &breaks, right_closed)?,
                None if ca.is_sorted_flag() == IsSorted::Ascending => {
                    bins_from_rank_cuts_sorted(ca, positions)
                },
                None => bins_from_rank_cuts_select(ca, positions, &values)?,
            };
            (breaks, bin_idx)
        });
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
        .slice(Slice::head(non_null_len));

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
