use super::min_max::AggType;
use super::*;
use crate::chunked_array::array::sum_mean::sum_with_nulls;
use crate::prelude::array::sum_mean::sum_array_numerical;

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

    fn array_median(&self) -> PolarsResult<Series> {
        let ca = self.as_array();
        dispersion::median_with_nulls(ca)
    }

    fn array_std(&self, ddof: u8) -> PolarsResult<Series> {
        let ca = self.as_array();
        dispersion::std_with_nulls(ca, ddof)
    }

    fn array_var(&self, ddof: u8) -> PolarsResult<Series> {
        let ca = self.as_array();
        dispersion::var_with_nulls(ca, ddof)
    }

    fn array_unique(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_array();
        ca.try_apply_amortized(|s| s.as_ref().unique())
    }

    fn array_unique_stable(&self) -> PolarsResult<ListChunked> {
        let ca = self.as_array();
        ca.try_apply_amortized(|s| s.as_ref().unique_stable())
    }
}

impl ArrayNameSpace for ArrayChunked {}
