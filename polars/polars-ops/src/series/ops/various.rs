#[cfg(feature = "hash")]
use polars_core::export::ahash;
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::series::ops::SeriesSealed;

pub trait SeriesMethods: SeriesSealed {
    /// Create a [`DataFrame`] with the unique `values` of this [`Series`] and a column `"counts"`
    /// with dtype [`IdxType`]
    fn value_counts(&self, multithreaded: bool, sorted: bool) -> PolarsResult<DataFrame> {
        let s = self.as_series();
        // we need to sort here as well in case of `maintain_order` because duplicates behavior is undefined
        let groups = s.group_tuples(multithreaded, sorted)?;
        let values = unsafe { s.agg_first(&groups) };
        let counts = groups.group_lengths("counts");
        let cols = vec![values, counts.into_series()];
        let df = DataFrame::new_no_checks(cols);
        if sorted {
            df.sort(["counts"], true, false)
        } else {
            Ok(df)
        }
    }

    #[cfg(feature = "hash")]
    fn hash(&self, build_hasher: ahash::RandomState) -> UInt64Chunked {
        let s = self.as_series().to_physical_repr();
        match s.dtype() {
            DataType::List(_) => {
                let mut ca = s.list().unwrap().clone();
                crate::chunked_array::hash::hash(&mut ca, build_hasher)
            }
            _ => {
                let mut h = vec![];
                s.0.vec_hash(build_hasher, &mut h).unwrap();
                UInt64Chunked::from_vec(s.name(), h)
            }
        }
    }

    fn is_sorted(&self, options: SortOptions) -> PolarsResult<bool> {
        let s = self.as_series();

        // fast paths
        if (options.descending
            && options.nulls_last
            && matches!(s.is_sorted_flag(), IsSorted::Descending))
            || (!options.descending
                && !options.nulls_last
                && matches!(s.is_sorted_flag(), IsSorted::Ascending))
        {
            return Ok(true);
        }
        let nc = s.null_count();
        let slen = s.len() - nc - 1; // Number of comparisons we might have to do
        if nc == s.len() {
            // All nulls is all equal
            return Ok(true);
        }
        if nc > 0 {
            let nulls = s.chunks().iter().flat_map(|c| c.validity().unwrap());
            let mut npairs = nulls.clone().zip(nulls.skip(1));
            // A null never precedes (follows) a non-null iff all nulls are at the end (beginning)
            if (options.nulls_last && npairs.any(|(a, b)| !a && b)) || npairs.any(|(a, b)| a && !b)
            {
                return Ok(false);
            }
        }
        // Compare adjacent elements with no-copy slices that don't include any nulls
        let offset = !options.nulls_last as i64 * nc as i64;
        let (s1, s2) = (s.slice(offset, slen), s.slice(offset + 1, slen));
        let cmp_op = match options.descending {
            true => Series::gt_eq,
            false => Series::lt_eq,
        };
        match s.dtype() {
            // For structs compare per-field. We don't have to check any types or field names though
            // since we're just comparing two offset slices of the same Series. The loop is to both
            // short-circuit on false and propagate errors. Maybe there's a way with iterators?
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => {
                let mut struct_cmp = true;
                for (l, r) in s1.struct_()?.fields().iter().zip(s2.struct_()?.fields()) {
                    struct_cmp &= cmp_op(l, r)?.all();
                    if !struct_cmp {
                        break;
                    }
                }
                Ok(struct_cmp)
            }
            _ => Ok(cmp_op(&s1, &s2)?.all()),
        }
    }
}

impl SeriesMethods for Series {}
