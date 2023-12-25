use std::borrow::Cow;

use ahash::RandomState;
use arrow::legacy::prelude::QuantileInterpolOptions;

use super::{private, IntoSeries, SeriesTrait, SeriesWrap, *};
use crate::chunked_array::comparison::*;
use crate::chunked_array::ops::aggregate::{ChunkAggSeries, QuantileAggSeries, VarAggSeries};
use crate::chunked_array::ops::compare_inner::{
    IntoTotalEqInner, IntoTotalOrdInner, TotalEqInner, TotalOrdInner,
};
use crate::chunked_array::ops::explode::ExplodeByOffsets;
use crate::chunked_array::AsSinglePtr;
#[cfg(feature = "algorithm_group_by")]
use crate::frame::group_by::*;
use crate::prelude::*;
#[cfg(feature = "checked_arithmetic")]
use crate::series::arithmetic::checked::NumOpsDispatchChecked;

macro_rules! impl_dyn_series {
    ($ca: ident) => {
        impl private::PrivateSeries for SeriesWrap<$ca> {
            fn compute_len(&mut self) {
                self.0.compute_len()
            }
            fn _field(&self) -> Cow<Field> {
                Cow::Borrowed(self.0.ref_field())
            }
            fn _dtype(&self) -> &DataType {
                self.0.ref_field().data_type()
            }

            fn _set_flags(&mut self, flags: Settings) {
                self.0.set_flags(flags)
            }
            fn _get_flags(&self) -> Settings {
                self.0.get_flags()
            }
            fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
                self.0.explode_by_offsets(offsets)
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
            fn zip_with_same_type(
                &self,
                mask: &BooleanChunked,
                other: &Series,
            ) -> PolarsResult<Series> {
                ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref())
                    .map(|ca| ca.into_series())
            }
            fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
                (&self.0).into_total_eq_inner()
            }
            fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
                (&self.0).into_total_ord_inner()
            }

            fn vec_hash(&self, random_state: RandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
                self.0.vec_hash(random_state, buf)?;
                Ok(())
            }

            fn vec_hash_combine(
                &self,
                build_hasher: RandomState,
                hashes: &mut [u64],
            ) -> PolarsResult<()> {
                self.0.vec_hash_combine(build_hasher, hashes)?;
                Ok(())
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
                self.0.agg_min(groups)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
                self.0.agg_max(groups)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Series {
                self.0.agg_sum(groups)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_std(&self, groups: &GroupsProxy, ddof: u8) -> Series {
                self.agg_std(groups, ddof)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_var(&self, groups: &GroupsProxy, ddof: u8) -> Series {
                self.agg_var(groups, ddof)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
                self.0.agg_list(groups)
            }

            fn subtract(&self, rhs: &Series) -> PolarsResult<Series> {
                NumOpsDispatch::subtract(&self.0, rhs)
            }
            fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
                NumOpsDispatch::add_to(&self.0, rhs)
            }
            fn multiply(&self, rhs: &Series) -> PolarsResult<Series> {
                NumOpsDispatch::multiply(&self.0, rhs)
            }
            fn divide(&self, rhs: &Series) -> PolarsResult<Series> {
                NumOpsDispatch::divide(&self.0, rhs)
            }
            fn remainder(&self, rhs: &Series) -> PolarsResult<Series> {
                NumOpsDispatch::remainder(&self.0, rhs)
            }
            #[cfg(feature = "algorithm_group_by")]
            fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
                IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
            }

            fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
                self.0.arg_sort_multiple(options)
            }
        }

        impl SeriesTrait for SeriesWrap<$ca> {
            #[cfg(feature = "rolling_window")]
            fn rolling_map(
                &self,
                _f: &dyn Fn(&Series) -> Series,
                _options: RollingOptionsFixedWindow,
            ) -> PolarsResult<Series> {
                ChunkRollApply::rolling_map(&self.0, _f, _options).map(|ca| ca.into_series())
            }

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
                return self.0.slice(offset, length).into_series();
            }

            fn append(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), append);
                self.0.append(other.as_ref().as_ref());
                Ok(())
            }

            fn extend(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), extend);
                self.0.extend(other.as_ref().as_ref());
                Ok(())
            }

            fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
                ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
            }

            fn mean(&self) -> Option<f64> {
                self.0.mean()
            }

            fn median(&self) -> Option<f64> {
                self.0.median().map(|v| v as f64)
            }

            fn std(&self, ddof: u8) -> Option<f64> {
                self.0.std(ddof)
            }

            fn var(&self, ddof: u8) -> Option<f64> {
                self.0.var(ddof)
            }

            #[cfg(feature = "chunked_ids")]
            unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
                self.0.take_chunked_unchecked(by, sorted).into_series()
            }

            #[cfg(feature = "chunked_ids")]
            unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
                self.0.take_opt_chunked_unchecked(by).into_series()
            }

            fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
                Ok(self.0.take(indices)?.into_series())
            }

            unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
                self.0.take_unchecked(indices).into_series()
            }

            fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
                Ok(self.0.take(indices)?.into_series())
            }

            unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
                self.0.take_unchecked(indices).into_series()
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn rechunk(&self) -> Series {
                self.0.rechunk().into_series()
            }

            fn new_from_index(&self, index: usize, length: usize) -> Series {
                ChunkExpandAtIndex::new_from_index(&self.0, index, length).into_series()
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
                ChunkSort::sort_with(&self.0, options).into_series()
            }

            fn arg_sort(&self, options: SortOptions) -> IdxCa {
                ChunkSort::arg_sort(&self.0, options)
            }

            fn null_count(&self) -> usize {
                self.0.null_count()
            }

            fn has_validity(&self) -> bool {
                self.0.has_validity()
            }

            #[cfg(feature = "algorithm_group_by")]
            fn unique(&self) -> PolarsResult<Series> {
                ChunkUnique::unique(&self.0).map(|ca| ca.into_series())
            }

            #[cfg(feature = "algorithm_group_by")]
            fn n_unique(&self) -> PolarsResult<usize> {
                ChunkUnique::n_unique(&self.0)
            }

            #[cfg(feature = "algorithm_group_by")]
            fn arg_unique(&self) -> PolarsResult<IdxCa> {
                ChunkUnique::arg_unique(&self.0)
            }

            fn is_null(&self) -> BooleanChunked {
                self.0.is_null()
            }

            fn is_not_null(&self) -> BooleanChunked {
                self.0.is_not_null()
            }

            fn reverse(&self) -> Series {
                ChunkReverse::reverse(&self.0).into_series()
            }

            fn as_single_ptr(&mut self) -> PolarsResult<usize> {
                self.0.as_single_ptr()
            }

            fn shift(&self, periods: i64) -> Series {
                ChunkShift::shift(&self.0, periods).into_series()
            }

            fn _sum_as_series(&self) -> PolarsResult<Series> {
                Ok(ChunkAggSeries::sum_as_series(&self.0))
            }
            fn max_as_series(&self) -> PolarsResult<Series> {
                Ok(ChunkAggSeries::max_as_series(&self.0))
            }
            fn min_as_series(&self) -> PolarsResult<Series> {
                Ok(ChunkAggSeries::min_as_series(&self.0))
            }
            fn median_as_series(&self) -> PolarsResult<Series> {
                Ok(QuantileAggSeries::median_as_series(&self.0))
            }
            fn var_as_series(&self, ddof: u8) -> PolarsResult<Series> {
                Ok(VarAggSeries::var_as_series(&self.0, ddof))
            }
            fn std_as_series(&self, ddof: u8) -> PolarsResult<Series> {
                Ok(VarAggSeries::std_as_series(&self.0, ddof))
            }
            fn quantile_as_series(
                &self,
                quantile: f64,
                interpol: QuantileInterpolOptions,
            ) -> PolarsResult<Series> {
                QuantileAggSeries::quantile_as_series(&self.0, quantile, interpol)
            }

            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(SeriesWrap(Clone::clone(&self.0)))
            }

            #[cfg(feature = "checked_arithmetic")]
            fn checked_div(&self, rhs: &Series) -> PolarsResult<Series> {
                self.0.checked_div(rhs)
            }
        }
    };
}

impl_dyn_series!(Float32Chunked);
impl_dyn_series!(Float64Chunked);
