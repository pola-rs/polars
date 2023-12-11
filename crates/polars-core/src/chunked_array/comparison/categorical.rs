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

// Huidige gedrag
// Equality comparisons
// If len == 1 => Convert it to phys and then apply on phys
// If len > 1 => Convert cat to str and then apply on string
// Ordering comparisons
// Convert to str and then apply on string

fn cat_str_equality_helper<'a, ComparePhys, CompareString>(
    lhs: &'a CategoricalChunked,
    rhs: &'a Utf8Chunked,
    fill_value: bool,
    phys_compare_function: ComparePhys,
    str_compare_function: CompareString,
) -> PolarsResult<BooleanChunked>
where
    ComparePhys: Fn(&'a UInt32Chunked, u32) -> BooleanChunked,
    CompareString: Fn(&Utf8Chunked, &'a Utf8Chunked) -> BooleanChunked,
{
    let rev_map = lhs.get_rev_map();

    if rhs.len() == 1 {
        if let Some(Some(idx)) = rhs.get(0).map(|s| rev_map.find(s)) {
            Ok(phys_compare_function(lhs.physical(), idx))
        } else {
            Ok(BooleanChunked::full(lhs.name(), fill_value, lhs.len()))
        }
    } else {
        let lhs_string = lhs.cast(&DataType::Utf8)?;
        Ok(str_compare_function(lhs_string.utf8().unwrap(), rhs))
    }
}

impl ChunkCompare<&Utf8Chunked> for CategoricalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(self, rhs, false, UInt32Chunked::equal, Utf8Chunked::equal)
    }

    fn equal_missing(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            false,
            UInt32Chunked::equal_missing,
            Utf8Chunked::equal_missing,
        )
    }

    fn not_equal(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            true,
            UInt32Chunked::not_equal,
            Utf8Chunked::not_equal,
        )
    }

    fn not_equal_missing(&self, rhs: &Utf8Chunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            true,
            UInt32Chunked::not_equal_missing,
            Utf8Chunked::not_equal_missing,
        )
    }

    fn gt(&self, rhs: &Utf8Chunked) -> Self::Item {
        todo!()
    }

    fn gt_eq(&self, rhs: &Utf8Chunked) -> Self::Item {
        todo!()
    }

    fn lt(&self, rhs: &Utf8Chunked) -> Self::Item {
        todo!()
    }

    fn lt_eq(&self, rhs: &Utf8Chunked) -> Self::Item {
        todo!()
    }
}
