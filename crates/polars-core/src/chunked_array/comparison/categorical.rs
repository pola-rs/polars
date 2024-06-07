use arrow::bitmap::Bitmap;
use arrow::legacy::utils::FromTrustedLenIterator;
use polars_compute::comparisons::TotalOrdKernel;

use crate::chunked_array::cast::CastOptions;
use crate::prelude::nulls::replace_non_null;
use crate::prelude::*;

#[cfg(feature = "dtype-categorical")]
fn cat_equality_helper<'a, Compare, Missing>(
    lhs: &'a CategoricalChunked,
    rhs: &'a CategoricalChunked,
    missing_function: Missing,
    compare_function: Compare,
) -> PolarsResult<BooleanChunked>
where
    Compare: Fn(&'a UInt32Chunked, &'a UInt32Chunked) -> BooleanChunked,
    Missing: Fn(&'a CategoricalChunked) -> BooleanChunked,
{
    let rev_map_l = lhs.get_rev_map();
    polars_ensure!(rev_map_l.same_src(rhs.get_rev_map()), string_cache_mismatch);
    let rhs = rhs.physical();

    // Fast path for globals
    if rhs.len() == 1 && rhs.null_count() == 0 {
        let rhs = rhs.get(0).unwrap();
        if rev_map_l.get_optional(rhs).is_none() {
            return Ok(missing_function(lhs));
        }
    }
    Ok(compare_function(lhs.physical(), rhs))
}

fn cat_compare_helper<'a, Compare, CompareString>(
    lhs: &'a CategoricalChunked,
    rhs: &'a CategoricalChunked,
    compare_function: Compare,
    compare_str_function: CompareString,
) -> PolarsResult<BooleanChunked>
where
    Compare: Fn(&'a UInt32Chunked, &'a UInt32Chunked) -> BooleanChunked,
    CompareString: Fn(&str, &str) -> bool,
{
    let rev_map_l = lhs.get_rev_map();
    let rev_map_r = rhs.get_rev_map();
    polars_ensure!(rev_map_l.same_src(rev_map_r), ComputeError: "can only compare categoricals of the same type with the same categories");

    if lhs.is_enum() || !lhs.uses_lexical_ordering() {
        Ok(compare_function(lhs.physical(), rhs.physical()))
    } else {
        match (lhs.len(), rhs.len()) {
            (lhs_len, 1) => {
                // SAFETY: physical is in range of revmap
                let v = unsafe {
                    rhs.physical()
                        .get(0)
                        .map(|phys| rev_map_r.get_unchecked(phys))
                };
                let Some(v) = v else {
                    return Ok(BooleanChunked::full_null(lhs.name(), lhs_len));
                };

                Ok(lhs
                    .iter_str()
                    .map(|opt_s| opt_s.map(|s| compare_str_function(s, v)))
                    .collect_ca_trusted(lhs.name()))
            },
            (1, rhs_len) => {
                // SAFETY: physical is in range of revmap
                let v = unsafe {
                    lhs.physical()
                        .get(0)
                        .map(|phys| rev_map_l.get_unchecked(phys))
                };
                let Some(v) = v else {
                    return Ok(BooleanChunked::full_null(lhs.name(), rhs_len));
                };
                Ok(rhs
                    .iter_str()
                    .map(|opt_s| opt_s.map(|s| compare_str_function(v, s)))
                    .collect_ca_trusted(lhs.name()))
            },
            (lhs_len, rhs_len) if lhs_len == rhs_len => Ok(lhs
                .iter_str()
                .zip(rhs.iter_str())
                .map(|(l, r)| match (l, r) {
                    (None, _) => None,
                    (_, None) => None,
                    (Some(l), Some(r)) => Some(compare_str_function(l, r)),
                })
                .collect_ca_trusted(lhs.name())),
            (lhs_len, rhs_len) => {
                polars_bail!(ComputeError: "Columns are of unequal length: {} vs {}",lhs_len,rhs_len)
            },
        }
    }
}

impl ChunkCompare<&CategoricalChunked> for CategoricalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(
            self,
            rhs,
            |lhs| replace_non_null(lhs.name(), &lhs.physical().chunks, false),
            UInt32Chunked::equal,
        )
    }

    fn equal_missing(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(
            self,
            rhs,
            |lhs| BooleanChunked::full(lhs.name(), false, lhs.len()),
            UInt32Chunked::equal_missing,
        )
    }

    fn not_equal(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(
            self,
            rhs,
            |lhs| replace_non_null(lhs.name(), &lhs.physical().chunks, true),
            UInt32Chunked::not_equal,
        )
    }

    fn not_equal_missing(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(
            self,
            rhs,
            |lhs| BooleanChunked::full(lhs.name(), true, lhs.len()),
            UInt32Chunked::not_equal_missing,
        )
    }

    fn gt(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::gt, |l, r| l > r)
    }

    fn gt_eq(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::gt_eq, |l, r| l >= r)
    }

    fn lt(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::lt, |l, r| l < r)
    }

    fn lt_eq(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::lt_eq, |l, r| l <= r)
    }
}

fn cat_str_equality_helper<'a, Missing, CompareNone, CompareCat, ComparePhys, CompareString>(
    lhs: &'a CategoricalChunked,
    rhs: &'a StringChunked,
    missing_function: Missing,
    compare_to_none: CompareNone,
    cat_compare_function: CompareCat,
    phys_compare_function: ComparePhys,
    str_compare_function: CompareString,
) -> PolarsResult<BooleanChunked>
where
    Missing: Fn(&CategoricalChunked) -> BooleanChunked,
    CompareNone: Fn(&CategoricalChunked) -> BooleanChunked,
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
    CompareCat: Fn(&CategoricalChunked, &CategoricalChunked) -> PolarsResult<BooleanChunked>,
    CompareString: Fn(&StringChunked, &'a StringChunked) -> BooleanChunked,
{
    if lhs.is_enum() {
        let rhs_cat = rhs.clone().into_series().strict_cast(lhs.dtype())?;
        cat_compare_function(lhs, rhs_cat.categorical().unwrap())
    } else if rhs.len() == 1 {
        match rhs.get(0) {
            None => Ok(compare_to_none(lhs)),
            Some(s) => {
                cat_single_str_equality_helper(lhs, s, missing_function, phys_compare_function)
            },
        }
    } else {
        let lhs_string = lhs.cast_with_options(&DataType::String, CastOptions::NonStrict)?;
        Ok(str_compare_function(lhs_string.str().unwrap(), rhs))
    }
}

fn cat_str_compare_helper<'a, CompareCat, ComparePhys, CompareStringSingle, CompareString>(
    lhs: &'a CategoricalChunked,
    rhs: &'a StringChunked,
    cat_compare_function: CompareCat,
    phys_compare_function: ComparePhys,
    str_single_compare_function: CompareStringSingle,
    str_compare_function: CompareString,
) -> PolarsResult<BooleanChunked>
where
    CompareStringSingle: Fn(&Utf8ViewArray, &str) -> Bitmap,
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
    CompareCat: Fn(&CategoricalChunked, &CategoricalChunked) -> PolarsResult<BooleanChunked>,
    CompareString: Fn(&StringChunked, &'a StringChunked) -> BooleanChunked,
{
    if lhs.is_enum() {
        let rhs_cat = rhs.clone().into_series().strict_cast(lhs.dtype())?;
        cat_compare_function(lhs, rhs_cat.categorical().unwrap())
    } else if rhs.len() == 1 {
        match rhs.get(0) {
            None => Ok(BooleanChunked::full_null(lhs.name(), lhs.len())),
            Some(s) => cat_single_str_compare_helper(
                lhs,
                s,
                phys_compare_function,
                str_single_compare_function,
            ),
        }
    } else {
        let lhs_string = lhs.cast_with_options(&DataType::String, CastOptions::NonStrict)?;
        Ok(str_compare_function(lhs_string.str().unwrap(), rhs))
    }
}

impl ChunkCompare<&StringChunked> for CategoricalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |lhs| replace_non_null(lhs.name(), &lhs.physical().chunks, false),
            |lhs| BooleanChunked::full_null(lhs.name(), lhs.len()),
            |s1, s2| CategoricalChunked::equal(s1, s2),
            UInt32Chunked::equal,
            StringChunked::equal,
        )
    }
    fn equal_missing(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |lhs| BooleanChunked::full(lhs.name(), false, lhs.len()),
            |lhs| lhs.physical().is_null(),
            |s1, s2| CategoricalChunked::equal_missing(s1, s2),
            UInt32Chunked::equal_missing,
            StringChunked::equal_missing,
        )
    }

    fn not_equal(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |lhs| replace_non_null(lhs.name(), &lhs.physical().chunks, true),
            |lhs| BooleanChunked::full_null(lhs.name(), lhs.len()),
            |s1, s2| CategoricalChunked::not_equal(s1, s2),
            UInt32Chunked::not_equal,
            StringChunked::not_equal,
        )
    }
    fn not_equal_missing(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |lhs| BooleanChunked::full(lhs.name(), true, lhs.len()),
            |lhs| !lhs.physical().is_null(),
            |s1, s2| CategoricalChunked::not_equal_missing(s1, s2),
            UInt32Chunked::not_equal_missing,
            StringChunked::not_equal_missing,
        )
    }

    fn gt(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::gt(s1, s2),
            UInt32Chunked::gt,
            Utf8ViewArray::tot_gt_kernel_broadcast,
            StringChunked::gt,
        )
    }

    fn gt_eq(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::gt_eq(s1, s2),
            UInt32Chunked::gt_eq,
            Utf8ViewArray::tot_ge_kernel_broadcast,
            StringChunked::gt_eq,
        )
    }

    fn lt(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::lt(s1, s2),
            UInt32Chunked::lt,
            Utf8ViewArray::tot_lt_kernel_broadcast,
            StringChunked::lt,
        )
    }

    fn lt_eq(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::lt_eq(s1, s2),
            UInt32Chunked::lt_eq,
            Utf8ViewArray::tot_le_kernel_broadcast,
            StringChunked::lt_eq,
        )
    }
}

fn cat_single_str_equality_helper<'a, ComparePhys, Missing>(
    lhs: &'a CategoricalChunked,
    rhs: &'a str,
    missing_function: Missing,
    phys_compare_function: ComparePhys,
) -> PolarsResult<BooleanChunked>
where
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
    Missing: Fn(&CategoricalChunked) -> BooleanChunked,
{
    let rev_map = lhs.get_rev_map();
    let idx = rev_map.find(rhs);
    if lhs.is_enum() {
        let Some(idx) = idx else {
            polars_bail!(
                not_in_enum,
                value = rhs,
                categories = rev_map.get_categories()
            )
        };
        Ok(phys_compare_function(lhs.physical(), idx))
    } else {
        match rev_map.find(rhs) {
            None => Ok(missing_function(lhs)),
            Some(idx) => Ok(phys_compare_function(lhs.physical(), idx)),
        }
    }
}

fn cat_single_str_compare_helper<'a, ComparePhys, CompareStringSingle>(
    lhs: &'a CategoricalChunked,
    rhs: &'a str,
    phys_compare_function: ComparePhys,
    str_single_compare_function: CompareStringSingle,
) -> PolarsResult<BooleanChunked>
where
    CompareStringSingle: Fn(&Utf8ViewArray, &str) -> Bitmap,
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
{
    let rev_map = lhs.get_rev_map();
    if lhs.is_enum() {
        match rev_map.find(rhs) {
            None => {
                polars_bail!(
                    not_in_enum,
                    value = rhs,
                    categories = rev_map.get_categories()
                )
            },
            Some(idx) => Ok(phys_compare_function(lhs.physical(), idx)),
        }
    } else {
        // Apply comparison on categories map and then do a lookup
        let bitmap = str_single_compare_function(lhs.get_rev_map().get_categories(), rhs);

        Ok(
            BooleanChunked::from_iter_trusted_length(lhs.physical().into_iter().map(|opt_idx| {
                // SAFETY: indexing into bitmap with same length as original array
                opt_idx.map(|idx| unsafe { bitmap.get_bit_unchecked(idx as usize) })
            }))
            .with_name(lhs.name()),
        )
    }
}

impl ChunkCompare<&str> for CategoricalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &str) -> Self::Item {
        cat_single_str_equality_helper(
            self,
            rhs,
            |lhs| replace_non_null(lhs.name(), &lhs.physical().chunks, false),
            UInt32Chunked::equal,
        )
    }

    fn equal_missing(&self, rhs: &str) -> Self::Item {
        cat_single_str_equality_helper(
            self,
            rhs,
            |lhs| BooleanChunked::full(lhs.name(), false, lhs.len()),
            UInt32Chunked::equal_missing,
        )
    }

    fn not_equal(&self, rhs: &str) -> Self::Item {
        cat_single_str_equality_helper(
            self,
            rhs,
            |lhs| replace_non_null(lhs.name(), &lhs.physical().chunks, true),
            UInt32Chunked::not_equal,
        )
    }

    fn not_equal_missing(&self, rhs: &str) -> Self::Item {
        cat_single_str_equality_helper(
            self,
            rhs,
            |lhs| BooleanChunked::full(lhs.name(), true, lhs.len()),
            UInt32Chunked::equal_missing,
        )
    }

    fn gt(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::gt,
            Utf8ViewArray::tot_gt_kernel_broadcast,
        )
    }

    fn gt_eq(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::gt_eq,
            Utf8ViewArray::tot_ge_kernel_broadcast,
        )
    }

    fn lt(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::lt,
            Utf8ViewArray::tot_lt_kernel_broadcast,
        )
    }

    fn lt_eq(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::lt_eq,
            Utf8ViewArray::tot_le_kernel_broadcast,
        )
    }
}
