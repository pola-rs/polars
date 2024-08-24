use num_traits::Bounded;
use polars_core::prelude::arity::unary_elementwise_values;
#[cfg(feature = "dtype-struct")]
use polars_core::prelude::sort::arg_sort_multiple::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::TotalOrd;

use crate::series::ops::SeriesSealed;

pub trait SeriesMethods: SeriesSealed {
    /// Create a [`DataFrame`] with the unique `values` of this [`Series`] and a column `"counts"`
    /// with dtype [`IdxType`]
    fn value_counts(
        &self,
        sort: bool,
        parallel: bool,
        name: String,
        normalize: bool,
    ) -> PolarsResult<DataFrame> {
        let s = self.as_series();
        polars_ensure!(
            s.name() != name,
            Duplicate: "using `value_counts` on a column/series named '{}' would lead to duplicate column names; change `name` to fix", name,
        );
        // we need to sort here as well in case of `maintain_order` because duplicates behavior is undefined
        let groups = s.group_tuples(parallel, sort)?;
        let values = unsafe { s.agg_first(&groups) };
        let counts = groups.group_count().with_name(name.as_str());

        let counts = if normalize {
            let len = s.len() as f64;
            let counts: Float64Chunked =
                unary_elementwise_values(&counts, |count| count as f64 / len);
            counts.into_series()
        } else {
            counts.into_series()
        };

        let cols = vec![values, counts.into_series()];
        let df = unsafe { DataFrame::new_no_checks(cols) };
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
    fn hash(&self, build_hasher: PlRandomState) -> UInt64Chunked {
        let s = self.as_series().to_physical_repr();
        match s.dtype() {
            DataType::List(_) => {
                let mut ca = s.list().unwrap().clone();
                crate::chunked_array::hash::hash(&mut ca, build_hasher)
            },
            _ => {
                let mut h = vec![];
                s.0.vec_hash(build_hasher, &mut h).unwrap();
                UInt64Chunked::from_vec(s.name(), h)
            },
        }
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
                "",
                &[s.clone()],
                &[options.descending],
                &[options.nulls_last],
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

        if s.dtype().is_numeric() {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                return Ok(is_sorted_ca_num::<$T>(ca, options))
            })
        }

        let cmp_len = s_len - null_count - 1; // Number of comparisons we might have to do
                                              // TODO! Change this, allocation of a full boolean series is too expensive and doesn't fail fast.
                                              // Compare adjacent elements with no-copy slices that don't include any nulls
        let offset = !options.nulls_last as i64 * null_count as i64;
        let (s1, s2) = (s.slice(offset, cmp_len), s.slice(offset + 1, cmp_len));
        let cmp_op = if options.descending {
            Series::gt_eq
        } else {
            Series::lt_eq
        };
        Ok(cmp_op(&s1, &s2)?.all())
    }
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
