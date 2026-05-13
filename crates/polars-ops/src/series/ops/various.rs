use arrow::bitmap::Bitmap;
use arrow::bitmap::aligned::AlignedBitmapSlice;
use num_traits::Bounded;
#[cfg(feature = "dtype-struct")]
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::with_match_physical_numeric_polars_type;
#[cfg(feature = "hash")]
use polars_utils::aliases::PlSeedableRandomStateQuality;
use polars_utils::total_ord::TotalOrd;

use crate::series::ops::SeriesSealed;

pub trait SeriesMethods: SeriesSealed {
    /// Create a [`DataFrame`] with the unique `values` of this [`Series`] and a column `"counts"`
    /// with dtype [`IdxType`]
    fn value_counts(
        &self,
        sort: bool,
        parallel: bool,
        name: PlSmallStr,
        normalize: bool,
    ) -> PolarsResult<DataFrame> {
        let s = self.as_series();
        polars_ensure!(
            s.name() != &name,
            Duplicate: "using `value_counts` on a column/series named '{}' would lead to duplicate \
            column names; change `name` to fix", name,
        );
        // we need to sort here as well in case of `maintain_order` because duplicates behavior is undefined
        let groups = s.group_tuples(parallel, sort)?;
        let values = unsafe { s.agg_first(&groups) }
            .with_name(s.name().clone())
            .into();
        let counts = groups.group_count().with_name(name.clone());

        let counts = if normalize {
            let len = s.len() as f64;
            let counts: Float64Chunked =
                unary_elementwise_values(&counts, |count| count as f64 / len);
            counts.into_column()
        } else {
            counts.into_column()
        };

        let height = counts.len();
        let cols = vec![values, counts];
        let df = unsafe { DataFrame::new_unchecked(height, cols) };
        if sort {
            df.sort(
                [name],
                SortMultipleOptions::default()
                    .with_order_descending(true)
                    .with_multithreaded(parallel),
            )
        } else {
            Ok(df)
        }
    }

    #[cfg(feature = "hash")]
    fn hash(&self, build_hasher: PlSeedableRandomStateQuality) -> UInt64Chunked {
        let s = self.as_series();
        let mut h = vec![];
        s.0.vec_hash(build_hasher, &mut h).unwrap();
        UInt64Chunked::from_vec(s.name().clone(), h)
    }

    fn ensure_sorted_arg(&self, operation: &str) -> PolarsResult<()> {
        polars_ensure!(self.is_sorted(Default::default())?, InvalidOperation: "argument in operation '{}' is not sorted, please sort the 'expr/series/column' first", operation);
        Ok(())
    }

    /// Checks if a [`Series`] is sorted. Tries to fail fast.
    fn is_sorted(&self, options: SortOptions) -> PolarsResult<bool> {
        let s = self.as_series();
        let null_count = s.null_count();

        // fast paths
        if (options.descending
            && (options.nulls_last || null_count == 0)
            && matches!(s.is_sorted_flag(), IsSorted::Descending))
            || (!options.descending
                && (!options.nulls_last || null_count == 0)
                && matches!(s.is_sorted_flag(), IsSorted::Ascending))
        {
            return Ok(true);
        }

        // for struct types we row-encode and recurse
        #[cfg(feature = "dtype-struct")]
        if matches!(s.dtype(), DataType::Struct(_)) {
            let encoded = _get_rows_encoded_ca(
                PlSmallStr::EMPTY,
                &[s.clone().into()],
                &[options.descending],
                &[options.nulls_last],
                false,
            )?;
            return encoded.into_series().is_sorted(options);
        }

        let s_len = s.len();
        if null_count == s_len {
            // All nulls is all equal
            return Ok(true);
        }
        // Check if nulls are in the right location.
        if null_count > 0 {
            // The slice triggers a fast null count
            if options.nulls_last {
                if s.slice((s_len - null_count) as i64, null_count)
                    .null_count()
                    != null_count
                {
                    return Ok(false);
                }
            } else if s.slice(0, null_count).null_count() != null_count {
                return Ok(false);
            }
        }

        if s.dtype().is_primitive_numeric() {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                return Ok(is_sorted_ca_num::<$T>(ca, options))
            })
        }

        // Logical non-primitive types (e.g. String, Categorical, List, …): take only the contiguous
        // non-null values (`non_null`). For ordinary `Categorical` use `iter_str` (below); otherwise
        // `to_physical_repr`, then
        // (1) reuse `is_sorted_ca_num` when the physical type is primitive numeric (temporal /
        //     Decimal, Enum-as-integer, …) after `to_physical_repr`;
        // (2) for ordinary [`DataType::Categorical`], compare adjacent **decoded strings** (`iter_str`),
        // (3) else scan booleans via a bitmap/`u64` word kernel / string+binary with `TotalOrd`,
        // (4) else fall back to pairwise `Series::lt_eq` / `gt_eq` (nested types, etc.).
        let non_null_len = s_len - null_count;
        if non_null_len <= 1 {
            return Ok(true);
        }

        let offset = (!options.nulls_last as i64) * (null_count as i64);
        let non_null = s.slice(offset, non_null_len);
        polars_ensure!(non_null.null_count() == 0, ComputeError: "internal error: `is_sorted` non-null slice contains nulls");

        #[cfg(feature = "dtype-categorical")]
        if matches!(non_null.dtype(), DataType::Categorical(_, _)) {
            return is_sorted_categorical_lexical_adjacent(&non_null, options);
        }

        let phys = non_null.to_physical_repr();
        let s_phys = phys.as_ref();
        if s_phys.dtype().is_primitive_numeric() {
            with_match_physical_numeric_polars_type!(s_phys.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s_phys.as_ref().as_ref().as_ref();
                return Ok(is_sorted_ca_num::<$T>(ca, options))
            })
        }

        match s_phys.dtype() {
            DataType::Boolean => {
                let ca = s_phys.bool()?;
                Ok(is_sorted_ca_bool(ca, options.descending))
            },
            DataType::String => {
                let ca = s_phys.str()?;
                Ok(is_sorted_adjacent_total_ord(
                    ca.into_no_null_iter(),
                    options.descending,
                ))
            },
            DataType::Binary => {
                let ca = s_phys.binary()?;
                Ok(is_sorted_adjacent_total_ord(
                    ca.into_no_null_iter(),
                    options.descending,
                ))
            },
            DataType::BinaryOffset => {
                let ca = s_phys.binary_offset()?;
                Ok(is_sorted_adjacent_total_ord(
                    ca.into_no_null_iter(),
                    options.descending,
                ))
            },
            _ => {
                // `non_null` excludes nulls already; compare `non_null[..-1]` with `non_null[1..]`.
                let cmp_len = non_null_len - 1;
                let s1 = non_null.slice(0, cmp_len);
                let s2 = non_null.slice(1, cmp_len);
                let cmp_op = if options.descending {
                    Series::gt_eq
                } else {
                    Series::lt_eq
                };
                Ok(cmp_op(&s1, &s2)?.all())
            },
        }
    }
}

/// Returns whether iterator elements are non-decreasing (`descending == false`) or non-increasing
/// (`descending == true`) under [`TotalOrd`].
///
/// Assumes the iterator `it` yields **only** the non-null values in row order (one item per row). An empty
/// iterator is considered sorted. Stops at the first pair that violates the ordering.
fn is_sorted_adjacent_total_ord<T: TotalOrd>(
    it: impl Iterator<Item = T>,
    descending: bool,
) -> bool {
    let mut it = it;
    // Sliding window: `prev` is always the previous element; seed with the first value.
    let Some(mut prev) = it.next() else {
        return true;
    };
    if descending {
        for v in it {
            if !prev.tot_ge(&v) {
                return false;
            }
            prev = v;
        }
    } else {
        for v in it {
            if !prev.tot_le(&v) {
                return false;
            }
            prev = v;
        }
    }
    true
}

/// Ordinary [`DataType::Categorical`] uses **lexical string order** in `Series::lt_eq` / `gt_eq` (decoded
/// labels via `iter_str`). This scans adjacent decoded strings—same semantics, no pairwise boolean mask.
#[cfg(feature = "dtype-categorical")]
fn is_sorted_categorical_lexical_adjacent(s: &Series, options: SortOptions) -> PolarsResult<bool> {
    polars_ensure!(
        matches!(s.dtype(), DataType::Categorical(_, _)),
        ComputeError: "internal error: expected Categorical in lexical `is_sorted` path",
    );
    polars_ensure!(
        s.null_count() == 0,
        ComputeError: "internal error: lexical categorical `is_sorted` expects no nulls in slice"
    );

    with_match_categorical_physical_type!(s.dtype().cat_physical().unwrap(), |$C| {
        let ca = s.cat::<$C>()?;
        polars_ensure!(
            ca.null_count() == 0,
            ComputeError: "internal error: categorical physical array unexpectedly contains nulls"
        );

        // `ca.null_count() == 0` implies each `phys` row decodes via `iter_str` to `Some(..)` (see
        // [`CategoricalChunked::iter_str`]).
        Ok(is_sorted_adjacent_total_ord(
            ca.iter_str().map(|opt| {
                opt.expect(
                    "`iter_str` produced None while categorical null_count reported 0 (`is_sorted`)"
                )
            }),
            options.descending,
        ))
    })
}

/// Returns [`true`] if the adjacent pair `prev` → `curr` violates monotonic order for booleans.
///
/// Ascending (**non‑decreasing**): violates iff `prev && !curr` (`true` then `false`).
/// Descending (**non‑increasing**): violates iff `!prev && curr` (`false` then `true`).
#[inline(always)]
fn bool_pair_is_unsorted(prev: bool, curr: bool, descending: bool) -> bool {
    if descending {
        !prev && curr
    } else {
        prev && !curr
    }
}

/// Scan one contiguous boolean **values** [`Bitmap`] for monotonic adjacent order (`false < true`).
///
/// Bit **`i`** is logical row **`i`** within this slice (Arrow / Polars LSB‑first indexing, same notion as `get_bit_unchecked`).
///
/// On success, updates **`prev`** to [`Some`] with this slice's **last** row value so callers can enforce the pair
/// across **(last row here → first row of the next bitmap or chunk)**.
///
/// Implemented as **`AlignedBitmapSlice<u64>`**: scalar **prefix**, **`u64` bulk** bitmask checks for interior adjacent
/// pairs, then scalar **suffix**. Used by boolean `Series::is_sorted` in this module.
fn is_sorted_boolean_bitmap_slice(
    bits: &Bitmap,
    descending: bool,
    prev: &mut Option<bool>,
) -> bool {
    if bits.is_empty() {
        return true;
    }

    // Split bitmap into prefix | aligned u64 chunks | suffix — see `AlignedBitmapSlice`.
    let (bytes, offset, length) = bits.as_slice();
    let aligned = AlignedBitmapSlice::<u64>::new(bytes, offset, length);

    // Prefix: first `< 64` logical bits, already packed into one u64 (LSB = lowest index in this slice).
    let p = aligned.prefix();
    for i in 0..aligned.prefix_bitlen() {
        // the boolean at index i in this prefix slice
        let b = ((p >> i) & 1) != 0;
        if let Some(pb) = *prev {
            if bool_pair_is_unsorted(pb, b, descending) {
                return false;
            }
        }
        *prev = Some(b);
    }

    // Bulk: 64 rows per word. Bit i = row i within the word (LSB-first).
    // Descending interior check must ignore bit 0: its left neighbor is `prev`, not bit 63 of the same word.
    const DESC_INTERIOR_MASK: u64 = u64::MAX << 1;

    for w in aligned.bulk_iter() {
        // Bridge (last row before this word) -> (row 0 of this word).
        if let Some(pb) = *prev {
            let first = (w & 1) != 0;
            if bool_pair_is_unsorted(pb, first, descending) {
                return false;
            }
        }
        if descending {
            // Nonzero iff some adjacent pair wholly inside bits 1..=63 forms `01` (forbidden descending).
            if (w & !(w << 1)) & DESC_INTERIOR_MASK != 0 {
                return false;
            }
        } else if (!w) & (w << 1) != 0 {
            // Nonzero iff some adjacent pair wholly inside bits 1..=63 forms `10` (forbidden ascending).
            return false;
        }
        // Last row in this word; pairs with row 0 of the next bulk word / suffix via `prev`.
        *prev = Some(((w >> 63) & 1) != 0);
    }

    // Suffix: trailing bits after the last full u64, same scalar walk as prefix.
    let s = aligned.suffix();
    for i in 0..aligned.suffix_bitlen() {
        // the boolean at index i in this suffix slice
        let b = ((s >> i) & 1) != 0;
        if let Some(pb) = *prev {
            if bool_pair_is_unsorted(pb, b, descending) {
                return false;
            }
        }
        *prev = Some(b);
    }

    true
}

/// Booleans ordered as [`false`] < [`true`] (same as inequality comparisons on [`BooleanChunked`]).
///
/// Caller must ensure **`ca` has no nulls** on the flattened series (see `non_null` slice above).
fn is_sorted_ca_bool(ca: &BooleanChunked, descending: bool) -> bool {
    let len = ca.len();
    if len <= 1 {
        return true;
    }
    // Defensive fallback if chunked metadata disagrees with per-chunk values.
    if ca.null_count() != 0 {
        return is_sorted_adjacent_total_ord(ca.into_no_null_iter(), descending);
    }

    let mut prev_row = None::<bool>;
    for arr in ca.downcast_iter() {
        let bits = arr.values();
        if bits.is_empty() {
            continue;
        }
        if !is_sorted_boolean_bitmap_slice(bits, descending, &mut prev_row) {
            return false;
        }
    }
    true
}

fn check_cmp<T: NumericNative, Cmp: Fn(&T, &T) -> bool>(
    vals: &[T],
    f: Cmp,
    previous: &mut T,
) -> bool {
    let mut sorted = true;

    // Outer loop so we can fail fast
    // Inner loop will auto vectorize
    for c in vals.chunks(1024) {
        // don't early stop or branch
        // so it autovectorizes
        for v in c {
            sorted &= f(previous, v);
            *previous = *v;
        }
        if !sorted {
            return false;
        }
    }
    sorted
}

// Assumes nulls last/first is already checked.
fn is_sorted_ca_num<T: PolarsNumericType>(ca: &ChunkedArray<T>, options: SortOptions) -> bool {
    if let Ok(vals) = ca.cont_slice() {
        let mut previous = vals[0];
        return if options.descending {
            check_cmp(vals, |prev, c| prev.tot_ge(c), &mut previous)
        } else {
            check_cmp(vals, |prev, c| prev.tot_le(c), &mut previous)
        };
    };

    if ca.null_count() == 0 {
        let mut previous = if options.descending {
            T::Native::max_value()
        } else {
            T::Native::min_value()
        };
        for arr in ca.downcast_iter() {
            let vals = arr.values();

            let sorted = if options.descending {
                check_cmp(vals, |prev, c| prev.tot_ge(c), &mut previous)
            } else {
                check_cmp(vals, |prev, c| prev.tot_le(c), &mut previous)
            };
            if !sorted {
                return false;
            }
        }
        return true;
    };

    // Slice off nulls and recurse.
    let null_count = ca.null_count();
    if options.nulls_last {
        let ca = ca.slice(0, ca.len() - null_count);
        is_sorted_ca_num(&ca, options)
    } else {
        let ca = ca.slice(null_count as i64, ca.len() - null_count);
        is_sorted_ca_num(&ca, options)
    }
}

impl SeriesMethods for Series {}

#[cfg(test)]
mod is_sorted_tests {
    use polars_core::prelude::*;

    use super::SeriesMethods;

    #[test]
    fn is_sorted_non_primitive_len_one_short_circuits() {
        let s = Series::new("b".into(), &[true]);
        assert_eq!(s.len(), 1);
        assert!(!s.dtype().is_primitive_numeric());

        assert!(s.is_sorted(SortOptions::default()).unwrap());
    }
}
