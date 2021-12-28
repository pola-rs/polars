use super::private;
use super::IntoSeries;
use super::SeriesTrait;
use super::SeriesWrap;
use crate::chunked_array::{
    comparison::*,
    ops::{explode::ExplodeByOffsets, ToBitRepr},
    AsSinglePtr, ChunkIdIter,
};
use crate::fmt::FmtList;
#[cfg(feature = "pivot")]
use crate::frame::groupby::pivot::*;
use crate::frame::{groupby::*, hash_join::*};
use crate::prelude::*;
use ahash::RandomState;
#[cfg(feature = "object")]
use std::any::Any;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

impl IntoSeries for DatetimeChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeries for SeriesWrap<DatetimeChunked> {
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(self.0.field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0
            .explode_by_offsets(offsets)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummax(&self, reverse: bool) -> Series {
        self.0.cummax(reverse).into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummin(&self, reverse: bool) -> Series {
        self.0.cummin(reverse).into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cumsum(&self, _reverse: bool) -> Series {
        panic!("cannot sum logical")
    }

    #[cfg(feature = "asof_join")]
    fn join_asof(&self, other: &Series) -> Result<Vec<Option<u32>>> {
        let other = other.to_physical_repr();
        self.0.deref().join_asof(&other)
    }

    fn set_sorted(&mut self, reverse: bool) {
        self.0.deref_mut().set_sorted(reverse)
    }

    unsafe fn equal_element(
        &self,
        idx_self: usize,
        idx_other: usize,
        other: &Series,
    ) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
        let other = other.to_physical_repr().into_owned();
        self.0
            .zip_with(mask, &other.as_ref().as_ref())
            .map(|ca| ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series())
    }

    fn vec_hash(&self, random_state: RandomState) -> Vec<u64> {
        self.0.vec_hash(random_state)
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
        self.0.vec_hash_combine(build_hasher, hashes)
    }

    fn agg_mean(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_min(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.0
            .agg_min(groups)
            .map(|ca| ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series())
    }

    fn agg_max(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.0
            .agg_max(groups)
            .map(|ca| ca.into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series())
    }

    fn agg_sum(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        self.0.agg_first(groups).into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series()
    }

    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        self.0.agg_last(groups).into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series()
    }

    fn agg_std(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_var(&self, _groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        // does not make sense on logical
        None
    }

    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        self.0.agg_n_unique(groups)
    }

    fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        // we cannot cast and dispatch as the inner type of the list would be incorrect
        self.0.agg_list(groups).map(|s| {
            s.cast(&DataType::List(Box::new(self.dtype().clone())))
                .unwrap()
        })
    }

    fn agg_quantile(
        &self,
        groups: &[(u32, Vec<u32>)],
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Option<Series> {
        self.0
            .agg_quantile(groups, quantile, interpol)
            .map(|s| s.into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series())
    }

    fn agg_median(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.0
            .agg_median(groups)
            .map(|s| s.into_datetime(self.0.time_unit(), self.0.time_zone().clone()).into_series())
    }
    #[cfg(feature = "lazy")]
    fn agg_valid_count(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.0.agg_valid_count(groups)
    }

    #[cfg(feature = "pivot")]
    fn pivot<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
        agg_type: PivotAgg,
    ) -> Result<DataFrame> {
        self.0.pivot(pivot_series, keys, groups, agg_type)
    }

    #[cfg(feature = "pivot")]
    fn pivot_count<'a>(
        &self,
        pivot_series: &'a Series,
        keys: Vec<Series>,
        groups: &[(u32, Vec<u32>)],
    ) -> Result<DataFrame> {
        self.0.pivot_count(pivot_series, keys, groups)
    }
    fn hash_join_inner(&self, other: &Series) -> Vec<(u32, u32)> {
        let other = other.to_physical_repr().into_owned();
        self.0.hash_join_inner(&other.as_ref().as_ref())
    }
    fn hash_join_left(&self, other: &Series) -> Vec<(u32, Option<u32>)> {
        let other = other.to_physical_repr().into_owned();
        self.0.hash_join_left(&other.as_ref().as_ref())
    }
    fn hash_join_outer(&self, other: &Series) -> Vec<(Option<u32>, Option<u32>)> {
        let other = other.to_physical_repr().into_owned();
        self.0.hash_join_outer(&other.as_ref().as_ref())
    }
    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<u32>, Option<u32>)],
    ) -> Series {
        let right_column = right_column.to_physical_repr().into_owned();
        self.0
            .zip_outer_join_column(&right_column, opt_join_tuples)
            .into_datetime(self.0.time_unit(), self.0.time_zone().clone())
            .into_series()
    }
    fn subtract(&self, rhs: &Series) -> Result<Series> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Datetime(tu, tz), DataType::Datetime(tur, tzr)) => {
                assert_eq!(tu, tur);
                assert_eq!(tz, tzr);
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs.subtract(&rhs)?.into_datetime(tu, tz.clone()).into_series())
            }
            (dtl, dtr) => Err(PolarsError::ComputeError(
                format!(
                    "cannot do subtraction on these date types: {:?}, {:?}",
                    dtl, dtr
                )
                    .into(),
            )),
        }
    }
    fn add_to(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::ComputeError(
            "cannot do addition on logical".into(),
        ))
    }
    fn multiply(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::ComputeError(
            "cannot do multiplication on logical".into(),
        ))
    }
    fn divide(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::ComputeError(
            "cannot do division on logical".into(),
        ))
    }
    fn remainder(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::ComputeError(
            "cannot do remainder operation on logical".into(),
        ))
    }
    fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
        self.0.group_tuples(multithreaded)
    }
    #[cfg(feature = "sort_multiple")]
    fn argsort_multiple(&self, by: &[Series], reverse: &[bool]) -> Result<UInt32Chunked> {
        self.0.deref().argsort_multiple(by, reverse)
    }

    fn str_value(&self, index: usize) -> Cow<str> {
        // get AnyValue
        Cow::Owned(format!("{}", self.get(index)))
    }

}
