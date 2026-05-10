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
        // (3) else scan bool / string / binary values with `TotalOrd` (no full boolean compare series),
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
                Ok(is_sorted_adjacent_total_ord(
                    ca.into_no_null_iter(),
                    options.descending,
                ))
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
                let cmp_len = s_len - null_count - 1;
                let offset = (!options.nulls_last as i64) * (null_count as i64);
                let (s1, s2) = (s.slice(offset, cmp_len), s.slice(offset + 1, cmp_len));
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

        // SAFETY CONTRACT: physical null count is zero, so each `phys` row resolves to `Some(..)` via
        // `iter_str` (see [`CategoricalChunked::iter_str`]).
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
