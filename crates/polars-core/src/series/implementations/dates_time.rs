//! This module exists to reduce compilation times.
//!
//! All the data types are backed by a physical type in memory e.g. Date -> i32, Datetime-> i64.
//!
//! Series lead to code implementations of all traits. Whereas there are a lot of duplicates due to
//! data types being backed by the same physical type. In this module we reduce compile times by
//! opting for a little more run time cost. We cast to the physical type -> apply the operation and
//! (depending on the result) cast back to the original type
//!
use std::borrow::Cow;
use std::ops::Deref;

use ahash::RandomState;
use arrow::legacy::prelude::QuantileInterpolOptions;

use super::{private, IntoSeries, SeriesTrait, SeriesWrap, *};
use crate::chunked_array::ops::explode::ExplodeByOffsets;
use crate::chunked_array::ops::ToBitRepr;
use crate::chunked_array::AsSinglePtr;
#[cfg(feature = "algorithm_group_by")]
use crate::frame::group_by::*;
use crate::prelude::*;

macro_rules! impl_dyn_series {
    ($ca: ident, $into_logical: ident) => {
        unsafe impl IntoSeries for $ca {
            fn into_series(self) -> Series {
                Series(Arc::new(SeriesWrap(self)))
            }
        }

        impl private::PrivateSeries for SeriesWrap<$ca> {
            fn compute_len(&mut self) {
                self.0.compute_len()
            }
            fn _field(&self) -> Cow<Field> {
                Cow::Owned(self.0.field())
            }
            fn _dtype(&self) -> &DataType {
                self.0.dtype()
            }
            fn _get_flags(&self) -> Settings{
                self.0.get_flags()
            }
            fn _set_flags(&mut self, flags: Settings){
                self.0.set_flags(flags)
            }

            fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
                self.0
                    .explode_by_offsets(offsets)
                    .$into_logical()
                    .into_series()
            }

            #[cfg(feature = "zip_with")]
            fn zip_with_same_type(
                &self,
                mask: &BooleanChunked,
                other: &Series,
            ) -> PolarsResult<Series> {
                let other = other.to_physical_repr().into_owned();
                self.0
                    .zip_with(mask, &other.as_ref().as_ref())
                    .map(|ca| ca.$into_logical().into_series())
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
                self.0.agg_min(groups).$into_logical().into_series()
            }

        #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
                self.0.agg_max(groups).$into_logical().into_series()
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
                    (DataType::Date, DataType::Date) => {
                        let dt = DataType::Datetime(TimeUnit::Milliseconds, None);
                        let lhs = self.cast(&dt)?;
                        let rhs = rhs.cast(&dt)?;
                        lhs.subtract(&rhs)
                    }
                    (DataType::Date, DataType::Duration(_)) => ((&self
                        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                        .unwrap())
                        - rhs)
                        .cast(&DataType::Date),
                    (dtl, dtr) => polars_bail!(opq = sub, dtl, dtr),
                }
            }
            fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
                match (self.dtype(), rhs.dtype()) {
                    (DataType::Date, DataType::Duration(_)) => ((&self
                        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                        .unwrap())
                        + rhs)
                        .cast(&DataType::Date),
                    (dtl, dtr) => polars_bail!(opq = add, dtl, dtr),
                }
            }
            fn multiply(&self, rhs: &Series) -> PolarsResult<Series> {
                polars_bail!(opq = mul, self.0.dtype(), rhs.dtype());
            }
            fn divide(&self, rhs: &Series) -> PolarsResult<Series> {
                polars_bail!(opq = div, self.0.dtype(), rhs.dtype());
            }
            fn remainder(&self, rhs: &Series) -> PolarsResult<Series> {
                polars_bail!(opq = rem, self.0.dtype(), rhs.dtype());
            }
    #[cfg(feature = "algorithm_group_by")]
            fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
                self.0.group_tuples(multithreaded, sorted)
            }

            fn arg_sort_multiple(&self, options: &SortMultipleOptions) -> PolarsResult<IdxCa> {
                self.0.deref().arg_sort_multiple(options)
            }
        }

        impl SeriesTrait for SeriesWrap<$ca> {

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
                self.0.slice(offset, length).$into_logical().into_series()
            }

            fn mean(&self) -> Option<f64> {
                self.0.mean()
            }

            fn median(&self) -> Option<f64> {
                self.0.median()
            }

            fn append(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), append);
                let other = other.to_physical_repr();
                // 3 refs
                // ref Cow
                // ref SeriesTrait
                // ref ChunkedArray
                self.0.append(other.as_ref().as_ref().as_ref());
                Ok(())
            }
            fn extend(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), extend);
                // 3 refs
                // ref Cow
                // ref SeriesTrait
                // ref ChunkedArray
                let other = other.to_physical_repr();
                self.0.extend(other.as_ref().as_ref().as_ref());
                Ok(())
            }

            fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
                self.0
                    .filter(filter)
                    .map(|ca| ca.$into_logical().into_series())
            }

            #[cfg(feature = "chunked_ids")]
            unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
                let ca = self.0.deref().take_chunked_unchecked(by, sorted);
                ca.$into_logical().into_series()
            }

            #[cfg(feature = "chunked_ids")]
            unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
                let ca = self.0.deref().take_opt_chunked_unchecked(by);
                ca.$into_logical().into_series()
            }

            fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
                Ok(self.0.take(indices)?.$into_logical().into_series())
            }

            unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
                self.0.take_unchecked(indices).$into_logical().into_series()
            }

            fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
                Ok(self.0.take(indices)?.$into_logical().into_series())
            }

            unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
                self.0.take_unchecked(indices).$into_logical().into_series()
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn rechunk(&self) -> Series {
                self.0.rechunk().$into_logical().into_series()
            }

            fn new_from_index(&self, index: usize, length: usize) -> Series {
                self.0
                    .new_from_index(index, length)
                    .$into_logical()
                    .into_series()
            }

            fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
                match (self.dtype(), data_type) {
                    #[cfg(feature="dtype-date")]
                    (DataType::Date, DataType::Utf8) => Ok(self
                        .0
                        .clone()
                        .into_series()
                        .date()
                        .unwrap()
                        .to_string("%Y-%m-%d")
                        .into_series()),
                    #[cfg(feature="dtype-time")]
                    (DataType::Time, DataType::Utf8) => Ok(self
                        .0
                        .clone()
                        .into_series()
                        .time()
                        .unwrap()
                        .to_string("%T")
                        .into_series()),
                    #[cfg(feature = "dtype-datetime")]
                    (DataType::Time, DataType::Datetime(_, _)) => {
                        polars_bail!(
                            ComputeError:
                            "cannot cast `Time` to `Datetime`; consider using 'dt.combine'"
                        );
                    }
                    #[cfg(feature = "dtype-datetime")]
                    (DataType::Date, DataType::Datetime(_, _)) => {
                        let mut out = self.0.cast(data_type)?;
                        out.set_sorted_flag(self.0.is_sorted_flag());
                        Ok(out)
                    }
                    _ => self.0.cast(data_type),
                }
            }

            fn get(&self, index: usize) -> PolarsResult<AnyValue> {
                self.0.get_any_value(index)
            }

            #[inline]
            unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
                self.0.get_any_value_unchecked(index)
            }

            fn sort_with(&self, options: SortOptions) -> Series {
                self.0.sort_with(options).$into_logical().into_series()
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
                self.0.unique().map(|ca| ca.$into_logical().into_series())
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
                self.0.reverse().$into_logical().into_series()
            }

            fn as_single_ptr(&mut self) -> PolarsResult<usize> {
                self.0.as_single_ptr()
            }

            fn shift(&self, periods: i64) -> Series {
                self.0.shift(periods).$into_logical().into_series()
            }

            fn _sum_as_series(&self) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
            }
            fn max_as_series(&self) -> Series {
                self.0.max_as_series().$into_logical()
            }
            fn min_as_series(&self) -> Series {
                self.0.min_as_series().$into_logical()
            }
            fn median_as_series(&self) -> Series {
                Series::new(
                    self.name(),
                    &[self.median().map(|v| (v * 86_400_000_000f64) as i64)],
                )
            }
            fn var_as_series(&self, _ddof: u8) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
            }
            fn std_as_series(&self, _ddof: u8) -> Series {
                Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into()
            }
            fn quantile_as_series(
                &self,
                _quantile: f64,
                _interpol: QuantileInterpolOptions,
            ) -> PolarsResult<Series> {
                Ok(Int32Chunked::full_null(self.name(), 1)
                    .cast(self.dtype())
                    .unwrap()
                    .into())
            }

            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(SeriesWrap(Clone::clone(&self.0)))
            }
        }
    };
}

#[cfg(feature = "dtype-date")]
impl_dyn_series!(DateChunked, into_date);
#[cfg(feature = "dtype-time")]
impl_dyn_series!(TimeChunked, into_time);

macro_rules! impl_dyn_series_numeric {
    ($ca: ident) => {
        impl private::PrivateSeriesNumeric for SeriesWrap<$ca> {
            fn bit_repr_is_large(&self) -> bool {
                if let DataType::Time = self.dtype() {
                    true
                } else {
                    false
                }
            }
            fn bit_repr_large(&self) -> UInt64Chunked {
                self.0.bit_repr_large()
            }
            fn bit_repr_small(&self) -> UInt32Chunked {
                self.0.bit_repr_small()
            }
        }
    };
}

#[cfg(feature = "dtype-date")]
impl_dyn_series_numeric!(DateChunked);
#[cfg(feature = "dtype-time")]
impl_dyn_series_numeric!(TimeChunked);
