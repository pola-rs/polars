use arrow::bitmap::Bitmap;
use arrow::legacy::utils::FromTrustedLenIterator;
use polars_compute::comparisons::TotalOrdKernel;

use crate::prelude::*;

#[cfg(feature = "dtype-categorical")]
fn cat_equality_helper<'a, Compare>(
    lhs: &'a CategoricalChunked,
    rhs: &'a CategoricalChunked,
    fill_value: bool,
    compare_function: Compare,
) -> PolarsResult<BooleanChunked>
where
    Compare: Fn(&'a UInt32Chunked, &'a UInt32Chunked) -> BooleanChunked,
{
    let rev_map_l = lhs.get_rev_map();
    polars_ensure!(rev_map_l.same_src(rhs.get_rev_map()), string_cache_mismatch);
    let rhs = rhs.physical();

    // Fast path for globals
    if rhs.len() == 1 && rhs.null_count() == 0 {
        let rhs = rhs.get(0).unwrap();
        if rev_map_l.get_optional(rhs).is_none() {
            return Ok(BooleanChunked::full(lhs.name(), fill_value, lhs.len()));
        }
    }
    Ok(compare_function(lhs.physical(), rhs))
}

fn cat_compare_helper<'a, Compare>(
    lhs: &'a CategoricalChunked,
    rhs: &'a CategoricalChunked,
    compare_function: Compare,
) -> PolarsResult<BooleanChunked>
where
    Compare: Fn(&'a UInt32Chunked, &'a UInt32Chunked) -> BooleanChunked,
{
    let rev_map_l = lhs.get_rev_map();
    let rev_map_r = rhs.get_rev_map();
    polars_ensure!(rev_map_l.is_enum() && rev_map_r.is_enum(), ComputeError: "can not compare (<, <=, >, >=) two categoricals, unless they are of Enum type");
    polars_ensure!(rev_map_l.same_src(rev_map_r), ComputeError: "can only compare Enum types with the same categories {:?} vs {:?}", rev_map_l.get_categories(), rev_map_r.get_categories());

    Ok(compare_function(lhs.physical(), rhs.physical()))
}

impl ChunkCompare<&CategoricalChunked> for CategoricalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(self, rhs, false, UInt32Chunked::equal)
    }

    fn equal_missing(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(self, rhs, false, UInt32Chunked::equal_missing)
    }

    fn not_equal(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(self, rhs, true, UInt32Chunked::not_equal)
    }

    fn not_equal_missing(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_equality_helper(self, rhs, true, UInt32Chunked::not_equal_missing)
    }

    fn gt(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::gt)
    }

    fn gt_eq(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::gt_eq)
    }

    fn lt(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::lt)
    }

    fn lt_eq(&self, rhs: &CategoricalChunked) -> Self::Item {
        cat_compare_helper(self, rhs, UInt32Chunked::lt_eq)
    }
}

fn cat_str_equality_helper<'a, Missing, CompareCat, ComparePhys, CompareString>(
    lhs: &'a CategoricalChunked,
    rhs: &'a Utf8Chunked,
    fill_value: bool,
    missing_compare_function: Missing,
    cat_compare_function: CompareCat,
    phys_compare_function: ComparePhys,
    str_compare_function: CompareString,
) -> PolarsResult<BooleanChunked>
where
    Missing: Fn(&CategoricalChunked) -> BooleanChunked,
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
    CompareCat: Fn(&CategoricalChunked, &CategoricalChunked) -> PolarsResult<BooleanChunked>,
    CompareString: Fn(&Utf8Chunked, &'a Utf8Chunked) -> BooleanChunked,
{
    let rev_map = lhs.get_rev_map();
    if rev_map.is_enum() {
        let rhs_cat = rhs.cast(lhs.dtype())?;
        cat_compare_function(lhs, rhs_cat.categorical().unwrap())
    } else if rhs.len() == 1 {
        match rhs.get(0) {
            None => Ok(missing_compare_function(lhs)),
            Some(s) => cat_single_str_equality_helper(lhs, s, fill_value, phys_compare_function),
        }
    } else {
        let lhs_string = lhs.cast(&DataType::Utf8)?;
        Ok(str_compare_function(lhs_string.utf8().unwrap(), rhs))
    }
}

fn cat_str_compare_helper<'a, CompareCat, ComparePhys, CompareStringSingle, CompareString>(
    lhs: &'a CategoricalChunked,
    rhs: &'a Utf8Chunked,
    cat_compare_function: CompareCat,
    phys_compare_function: ComparePhys,
    str_single_compare_function: CompareStringSingle,
    str_compare_function: CompareString,
) -> PolarsResult<BooleanChunked>
where
    CompareStringSingle: Fn(&Utf8Array<i64>, &str) -> Bitmap,
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
    CompareCat: Fn(&CategoricalChunked, &CategoricalChunked) -> PolarsResult<BooleanChunked>,
    CompareString: Fn(&Utf8Chunked, &'a Utf8Chunked) -> BooleanChunked,
{
    let rev_map = lhs.get_rev_map();
    if rev_map.is_enum() {
        let rhs_cat = rhs.cast(lhs.dtype())?;
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
        let lhs_string = lhs.cast(&DataType::Utf8)?;
        Ok(str_compare_function(lhs_string.utf8().unwrap(), rhs))
    }
}

impl ChunkCompare<&Utf8Chunked> for CategoricalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            false,
            |lhs| BooleanChunked::full_null(lhs.name(), lhs.len()),
            |s1, s2| CategoricalChunked::equal(s1, s2),
            UInt32Chunked::equal,
            Utf8Chunked::equal,
        )
    }
    fn equal_missing(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            false,
            |lhs| lhs.physical().is_null(),
            |s1, s2| CategoricalChunked::equal_missing(s1, s2),
            UInt32Chunked::equal_missing,
            Utf8Chunked::equal_missing,
        )
    }

    fn not_equal(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            true,
            |lhs| BooleanChunked::full_null(lhs.name(), lhs.len()),
            |s1, s2| CategoricalChunked::not_equal(s1, s2),
            UInt32Chunked::not_equal,
            Utf8Chunked::not_equal,
        )
    }
    fn not_equal_missing(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            true,
            |lhs| !lhs.physical().is_null(),
            |s1, s2| CategoricalChunked::not_equal_missing(s1, s2),
            UInt32Chunked::not_equal_missing,
            Utf8Chunked::not_equal_missing,
        )
    }

    fn gt(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::gt(s1, s2),
            UInt32Chunked::gt,
            Utf8Array::tot_gt_kernel_broadcast,
            Utf8Chunked::gt,
        )
    }

    fn gt_eq(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::gt_eq(s1, s2),
            UInt32Chunked::gt_eq,
            Utf8Array::tot_ge_kernel_broadcast,
            Utf8Chunked::gt_eq,
        )
    }

    fn lt(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::lt(s1, s2),
            UInt32Chunked::lt,
            Utf8Array::tot_lt_kernel_broadcast,
            Utf8Chunked::lt,
        )
    }

    fn lt_eq(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_compare_helper(
            self,
            rhs,
            |s1, s2| CategoricalChunked::lt_eq(s1, s2),
            UInt32Chunked::lt_eq,
            Utf8Array::tot_le_kernel_broadcast,
            Utf8Chunked::lt_eq,
        )
    }
}

fn cat_single_str_equality_helper<'a, ComparePhys>(
    lhs: &'a CategoricalChunked,
    rhs: &'a str,
    fill_value: bool,
    phys_compare_function: ComparePhys,
) -> PolarsResult<BooleanChunked>
where
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
{
    let rev_map = lhs.get_rev_map();
    if rev_map.is_enum() {
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
        match rev_map.find(rhs) {
            None => Ok(BooleanChunked::full(lhs.name(), fill_value, lhs.len())),
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
    CompareStringSingle: Fn(&Utf8Array<i64>, &str) -> Bitmap,
    ComparePhys: Fn(&UInt32Chunked, u32) -> BooleanChunked,
{
    let rev_map = lhs.get_rev_map();
    if rev_map.is_enum() {
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

        Ok(BooleanChunked::from_iter_trusted_length(
            lhs.physical().into_iter().map(|opt_idx| {
                // Safety: indexing into bitmap with same length as original array
                opt_idx.map(|idx| unsafe { bitmap.get_bit_unchecked(idx as usize) })
            }),
        ))
    }
}

impl ChunkCompare<&str> for CategoricalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &str) -> Self::Item {
        cat_single_str_equality_helper(self, rhs, false, UInt32Chunked::equal)
    }

    fn equal_missing(&self, rhs: &str) -> Self::Item {
        self.equal(rhs)
    }

    fn not_equal(&self, rhs: &str) -> Self::Item {
        cat_single_str_equality_helper(self, rhs, true, UInt32Chunked::not_equal)
    }

    fn not_equal_missing(&self, rhs: &str) -> Self::Item {
        self.not_equal(rhs)
    }

    fn gt(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::gt,
            Utf8Array::tot_gt_kernel_broadcast,
        )
    }

    fn gt_eq(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::gt_eq,
            Utf8Array::tot_ge_kernel_broadcast,
        )
    }

    fn lt(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::lt,
            Utf8Array::tot_lt_kernel_broadcast,
        )
    }

    fn lt_eq(&self, rhs: &str) -> Self::Item {
        cat_single_str_compare_helper(
            self,
            rhs,
            UInt32Chunked::lt_eq,
            Utf8Array::tot_le_kernel_broadcast,
        )
    }
}
