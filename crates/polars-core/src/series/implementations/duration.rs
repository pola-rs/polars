use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

use ahash::RandomState;

use super::{private, IntoSeries, SeriesTrait, SeriesWrap, *};
use crate::chunked_array::comparison::*;
use crate::chunked_array::ops::explode::ExplodeByOffsets;
use crate::chunked_array::AsSinglePtr;
#[cfg(feature = "algorithm_group_by")]
use crate::frame::group_by::*;
use crate::prelude::*;

unsafe impl IntoSeries for DurationChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DurationChunked> {
    fn bit_repr_is_large(&self) -> bool {
        true
    }
    fn bit_repr_large(&self) -> UInt64Chunked {
        self.0.bit_repr_large()
    }
}

impl private::PrivateSeries for SeriesWrap<DurationChunked> {
    fn compute_len(&mut self) {
        self.0.compute_len()
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Owned(self.0.field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0
            .explode_by_offsets(offsets)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummax(&self, reverse: bool) -> Series {
        self.0
            .cummax(reverse)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "cum_agg")]
    fn _cummin(&self, reverse: bool) -> Series {
        self.0
            .cummin(reverse)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn _set_flags(&mut self, flags: Settings) {
        self.0.deref_mut().set_flags(flags)
    }
    fn _get_flags(&self) -> Settings {
        self.0.deref().get_flags()
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let other = other.to_physical_repr().into_owned();
        self.0
            .zip_with(mask, other.as_ref().as_ref())
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    fn vec_hash(&self, random_state: RandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.vec_hash(random_state, buf)?;
        Ok(())
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) -> PolarsResult<()> {
        self.0.vec_hash_combine(build_hasher, hashes)?;
        Ok(())
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        self.0
            .agg_min(groups)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        self.0
            .agg_max(groups)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
        self.0
            .agg_sum(groups)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_std(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        self.0
            .agg_std(groups, ddof)
            // cast f64 back to physical type
            .cast(&DataType::Int64)
            .unwrap()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_var(&self, groups: &GroupsProxy, ddof: u8) -> Series {
        self.0
            .agg_var(groups, ddof)
            // cast f64 back to physical type
            .cast(&DataType::Int64)
            .unwrap()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        // we cannot cast and dispatch as the inner type of the list would be incorrect
        self.0
            .agg_list(groups)
            .cast(&DataType::List(Box::new(self.dtype().clone())))
            .unwrap()
    }

    fn subtract(&self, rhs: &Series) -> PolarsResult<Series> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Duration(tu), DataType::Duration(tur)) => {
                polars_ensure!(tu == tur, InvalidOperation: "units are different");
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs.subtract(&rhs)?.into_duration(*tu).into_series())
            },
            (dtl, dtr) => polars_bail!(opq = sub, dtl, dtr),
        }
    }
    fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
        match (self.dtype(), rhs.dtype()) {
            (DataType::Duration(tu), DataType::Duration(tur)) => {
                polars_ensure!(tu == tur, InvalidOperation: "units are different");
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs.add_to(&rhs)?.into_duration(*tu).into_series())
            },
            (DataType::Duration(tu), DataType::Date) => {
                let one_day_in_tu: i64 = match tu {
                    TimeUnit::Milliseconds => 86_400_000,
                    TimeUnit::Microseconds => 86_400_000_000,
                    TimeUnit::Nanoseconds => 86_400_000_000_000,
                };
                let lhs = self.cast(&DataType::Int64).unwrap() / one_day_in_tu;
                let rhs = rhs
                    .cast(&DataType::Int32)
                    .unwrap()
                    .cast(&DataType::Int64)
                    .unwrap();
                Ok(lhs
                    .add_to(&rhs)?
                    .cast(&DataType::Int32)?
                    .into_date()
                    .into_series())
            },
            (DataType::Duration(tu), DataType::Datetime(tur, tz)) => {
                polars_ensure!(tu == tur, InvalidOperation: "units are different");
                let lhs = self.cast(&DataType::Int64).unwrap();
                let rhs = rhs.cast(&DataType::Int64).unwrap();
                Ok(lhs
                    .add_to(&rhs)?
                    .into_datetime(*tu, tz.clone())
                    .into_series())
            },
            (dtl, dtr) => polars_bail!(opq = add, dtl, dtr),
        }
    }
    fn multiply(&self, rhs: &Series) -> PolarsResult<Series> {
        // dependent on units, doesn't make sense
        polars_bail!(opq = mul, self.dtype(), rhs.dtype());
    }
    fn divide(&self, rhs: &Series) -> PolarsResult<Series> {
        // dependent on units, doesn't make sense
        polars_bail!(opq = div, self.dtype(), rhs.dtype());
    }
    fn remainder(&self, rhs: &Series) -> PolarsResult<Series> {
        polars_ensure!(self.dtype() == rhs.dtype(), InvalidOperation: "dtypes and units must be equal in duration arithmetic");
        let lhs = self.cast(&DataType::Int64).unwrap();
        let rhs = rhs.cast(&DataType::Int64).unwrap();
        Ok(lhs
            .remainder(&rhs)?
            .into_duration(self.0.time_unit())
            .into_series())
    }
    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        self.0.group_tuples(multithreaded, sorted)
    }

    fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
        self.0.deref().arg_sort_multiple(options)
    }
}

impl SeriesTrait for SeriesWrap<DurationChunked> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name);
    }

    fn chunk_lengths(&self) -> ChunkIdIter {
        self.0.chunk_id()
    }
    fn name(&self) -> &str {
        self.0.name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.chunks()
    }
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0
            .slice(offset, length)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn mean(&self) -> Option<f64> {
        self.0.mean()
    }

    fn median(&self) -> Option<f64> {
        self.0.median()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        let other = other.to_physical_repr().into_owned();
        self.0.append(other.as_ref().as_ref());
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        let other = other.to_physical_repr();
        self.0.extend(other.as_ref().as_ref().as_ref());
        Ok(())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        self.0
            .filter(filter)
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        let ca = self.0.deref().take_chunked_unchecked(by, sorted);
        ca.into_duration(self.0.time_unit()).into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        let ca = self.0.deref().take_opt_chunked_unchecked(by);
        ca.into_duration(self.0.time_unit()).into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(self
            .0
            .take(indices)?
            .into_duration(self.0.time_unit())
            .into_series())
    }

    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
        self.0
            .take_unchecked(indices)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        Ok(self
            .0
            .take(indices)?
            .into_duration(self.0.time_unit())
            .into_series())
    }

    unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
        self.0
            .take_unchecked(indices)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0
            .rechunk()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.0
            .new_from_index(index, length)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.0.cast(data_type)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn sort_with(&self, options: SortOptions) -> Series {
        self.0
            .sort_with(options)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn unique(&self) -> PolarsResult<Series> {
        self.0
            .unique()
            .map(|ca| ca.into_duration(self.0.time_unit()).into_series())
    }

    #[cfg(feature = "algorithm_group_by")]
    fn n_unique(&self) -> PolarsResult<usize> {
        self.0.n_unique()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.0.arg_unique()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn reverse(&self) -> Series {
        self.0
            .reverse()
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        self.0
            .shift(periods)
            .into_duration(self.0.time_unit())
            .into_series()
    }

    fn _sum_as_series(&self) -> Series {
        self.0.sum_as_series().into_duration(self.0.time_unit())
    }

    fn max_as_series(&self) -> Series {
        self.0.max_as_series().into_duration(self.0.time_unit())
    }
    fn min_as_series(&self) -> Series {
        self.0.min_as_series().into_duration(self.0.time_unit())
    }
    fn median_as_series(&self) -> Series {
        self.0
            .median_as_series()
            .cast(&self.dtype().to_physical())
            .unwrap()
            .cast(self.dtype())
            .unwrap()
    }
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        self.0
            .quantile_as_series(quantile, interpol)?
            .cast(&self.dtype().to_physical())
            .unwrap()
            .cast(self.dtype())
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }
}
