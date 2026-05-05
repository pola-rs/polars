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
        polars_ensure!(
            self.is_sorted(SortOptions::default())?,
            InvalidOperation: "argument in operation '{}' is not sorted, please sort the 'expr/series/column' first",
            operation
        );
        Ok(())
    }

    /// Checks if a [`Series`] is sorted with concrete options. Tries to fail fast.
    ///
    /// For inference of `descending` / `nulls_last`, see [`Self::is_sorted_any`].
    fn is_sorted(&self, options: SortOptions) -> PolarsResult<bool> {
        is_sorted_impl(self.as_series(), options)
    }

    fn is_sorted_any(
        &self,
        descending: Option<bool>,
        nulls_last: Option<bool>,
    ) -> PolarsResult<bool> {
        let s = self.as_series();
        match resolve_sort_options(s, descending, nulls_last)? {
            None => Ok(true),
            Some(opts) => is_sorted_impl(s, opts),
        }
    }
}

fn is_sorted_impl(s: &Series, options: SortOptions) -> PolarsResult<bool> {
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
        return is_sorted_impl(&encoded.into_series(), options);
    }

    let s_len = s.len();
    if null_count == s_len {
        return Ok(true);
    }
    if null_count > 0 {
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

    let cmp_len = s_len - null_count - 1;
    let offset = !options.nulls_last as i64 * null_count as i64;
    let (s1, s2) = (s.slice(offset, cmp_len), s.slice(offset + 1, cmp_len));
    let cmp_op = if options.descending {
        Series::gt_eq
    } else {
        Series::lt_eq
    };
    Ok(cmp_op(&s1, &s2)?.all())
}

pub fn resolve_sort_options(
    s: &Series,
    descending: Option<bool>,
    nulls_last: Option<bool>,
) -> PolarsResult<Option<SortOptions>> {
    let null_count = s.null_count();
    let s_len = s.len();

    if null_count == s_len {
        return Ok(None);
    }

    let nulls_actually_last: Option<bool> = if null_count == 0 {
        None
    } else if s
        .slice((s_len - null_count) as i64, null_count)
        .null_count()
        == null_count
    {
        Some(true)
    } else if s.slice(0, null_count).null_count() == null_count {
        Some(false)
    } else {
        return Ok(Some(SortOptions {
            descending: descending.unwrap_or(false),
            nulls_last: false,
            ..Default::default()
        }));
    };

    let nulls_last = match (nulls_last, nulls_actually_last) {
        (Some(n), _) => n,
        (None, Some(actual)) => actual,
        (None, None) => descending.unwrap_or(false),
    };

    let descending = match descending {
        Some(d) => d,
        None => match infer_descending(s, nulls_last)? {
            Some(d) => d,
            None => return Ok(None),
        },
    };

    Ok(Some(SortOptions {
        descending,
        nulls_last,
        ..Default::default()
    }))
}

fn infer_descending(s: &Series, nulls_last: bool) -> PolarsResult<Option<bool>> {
    let null_count = s.null_count();
    let non_null_len = s.len() - null_count;
    if non_null_len < 2 {
        return Ok(None);
    }

    let non_null_start = if nulls_last { 0 } else { null_count };
    let non_null = s.slice(non_null_start as i64, non_null_len);

    let a = non_null.slice(0, non_null_len - 1);
    let b = non_null.slice(1, non_null_len - 1);

    let lt = a.lt(&b)?;
    let gt = a.gt(&b)?;

    for (lt_v, gt_v) in lt.iter().zip(gt.iter()) {
        match (lt_v, gt_v) {
            (Some(true), _) => return Ok(Some(false)),
            (_, Some(true)) => return Ok(Some(true)),
            _ => {},
        }
    }
    Ok(None)
}

fn check_cmp<T: NumericNative, Cmp: Fn(&T, &T) -> bool>(
    vals: &[T],
    f: Cmp,
    previous: &mut T,
) -> bool {
    let mut sorted = true;
    for c in vals.chunks(1024) {
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
