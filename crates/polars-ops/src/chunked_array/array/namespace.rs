use super::min_max::AggType;
use super::*;
use crate::chunked_array::array::sum_mean::sum_with_nulls;
#[cfg(feature = "array_any_all")]
use crate::prelude::array::any_all::{array_all, array_any};
use crate::prelude::array::get::array_get;
use crate::prelude::array::join::array_join;
use crate::prelude::array::sum_mean::sum_array_numerical;
use crate::series::ArgAgg;

pub fn has_inner_nulls(ca: &ArrayChunked) -> bool {
    for arr in ca.downcast_iter() {
        if arr.values().null_count() > 0 {
            return true;
        }
    }
    false
}

fn get_agg(ca: &ArrayChunked, agg_type: AggType) -> Series {
    let values = ca.get_inner();
    let width = ca.width();
    min_max::array_dispatch(ca.name(), &values, width, agg_type)
}

pub trait ArrayNameSpace: AsArray {
    fn array_max(&self) -> Series {
        let ca = self.as_array();
        get_agg(ca, AggType::Max)
    }

    fn array_min(&self) -> Series {
        let ca = self.as_array();
        get_agg(ca, AggType::Min)
    }

    fn array_sum(&self) -> PolarsResult<Series> {
        let ca = self.as_array();

        if has_inner_nulls(ca) {
            return sum_with_nulls(ca, &ca.inner_dtype());
        };

        match ca.inner_dtype() {
            dt if dt.is_numeric() => Ok(sum_array_numerical(ca, &dt)),
            dt => sum_with_nulls(ca, &dt),
        }
    }

    fn array_unique(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_array();
        ca.try_apply_amortized_to_list(|s| s.as_ref().unique())
    }

    fn array_unique_stable(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_array();
        ca.try_apply_amortized_to_list(|s| s.as_ref().unique_stable())
    }

    #[cfg(feature = "array_any_all")]
    fn array_any(&self) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_any(ca)
    }

    #[cfg(feature = "array_any_all")]
    fn array_all(&self) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_all(ca)
    }

    fn array_sort(&self, options: SortOptions) -> ArrayChunked {
        let ca = self.as_array();
        // SAFETY: Sort only changes the order of the elements in each subarray.
        unsafe { ca.apply_amortized_same_type(|s| s.as_ref().sort_with(options)) }
    }

    fn array_reverse(&self) -> ArrayChunked {
        let ca = self.as_array();
        // SAFETY: Reverse only changes the order of the elements in each subarray
        unsafe { ca.apply_amortized_same_type(|s| s.as_ref().reverse()) }
    }

    fn array_arg_min(&self) -> IdxCa {
        let ca = self.as_array();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_min().map(|idx| idx as IdxSize))
        })
    }

    fn array_arg_max(&self) -> IdxCa {
        let ca = self.as_array();
        ca.apply_amortized_generic(|opt_s| {
            opt_s.and_then(|s| s.as_ref().arg_max().map(|idx| idx as IdxSize))
        })
    }

    fn array_get(&self, index: &Int64Chunked) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_get(ca, index)
    }

    fn array_join(&self, separator: &StringChunked) -> PolarsResult<Series> {
        let ca = self.as_array();
        array_join(ca, separator).map(|ok| ok.into_series())
    }
}

impl ArrayNameSpace for ArrayChunked {}
