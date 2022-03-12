use super::private;
use super::IntoSeries;
use super::SeriesTrait;
use super::SeriesWrap;
use super::*;
use crate::chunked_array::comparison::*;
#[cfg(feature = "rolling_window")]
use crate::chunked_array::ops::rolling_window::RollingOptions;
use crate::chunked_array::{
    ops::{
        aggregate::{ChunkAggSeries, QuantileAggSeries, VarAggSeries},
        compare_inner::{IntoPartialEqInner, IntoPartialOrdInner, PartialEqInner, PartialOrdInner},
        explode::ExplodeByOffsets,
    },
    AsSinglePtr, ChunkIdIter,
};
use crate::fmt::FmtList;
use crate::frame::groupby::*;
use crate::frame::hash_join::ZipOuterJoinColumn;
use crate::prelude::*;
#[cfg(feature = "checked_arithmetic")]
use crate::series::arithmetic::checked::NumOpsDispatchChecked;
use ahash::RandomState;
use arrow::array::ArrayRef;
use polars_arrow::prelude::QuantileInterpolOptions;
use std::borrow::Cow;
use std::ops::{BitAnd, BitOr, BitXor};

macro_rules! impl_dyn_series {
    ($ca: ident) => {
        impl IntoSeries for $ca {
            fn into_series(self) -> Series {
                Series(Arc::new(SeriesWrap(self)))
            }
        }

        impl private::PrivateSeries for SeriesWrap<$ca> {
            fn _field(&self) -> Cow<Field> {
                Cow::Borrowed(self.0.ref_field())
            }
            fn _dtype(&self) -> &DataType {
                self.0.ref_field().data_type()
            }

            fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
                self.0.explode_by_offsets(offsets)
            }

            #[cfg(feature = "rolling_window")]
            fn _rolling_mean(&self, options: RollingOptions) -> Result<Series> {
                self.0.rolling_mean(options)
            }
            #[cfg(feature = "rolling_window")]
            fn _rolling_median(&self, options: RollingOptions) -> Result<Series> {
                self.0.rolling_median(options)
            }
            #[cfg(feature = "rolling_window")]
            fn _rolling_quantile(
                &self,
                quantile: f64,
                interpolation: QuantileInterpolOptions,
                options: RollingOptions,
            ) -> Result<Series> {
                self.0.rolling_quantile(quantile, interpolation, options)
            }
            #[cfg(feature = "rolling_window")]
            fn _rolling_sum(&self, options: RollingOptions) -> Result<Series> {
                self.0.rolling_sum(options).map(|ca| ca.into())
            }
            #[cfg(feature = "rolling_window")]
            fn _rolling_min(&self, options: RollingOptions) -> Result<Series> {
                self.0.rolling_min(options).map(|ca| ca.into())
            }
            #[cfg(feature = "rolling_window")]
            fn _rolling_max(&self, options: RollingOptions) -> Result<Series> {
                self.0.rolling_max(options).map(|ca| ca.into())
            }
            #[cfg(feature = "rolling_window")]
            fn _rolling_std(&self, options: RollingOptions) -> Result<Series> {
                self.0.rolling_std(options)
            }

            #[cfg(feature = "rolling_window")]
            fn _rolling_var(&self, options: RollingOptions) -> Result<Series> {
                self.0.rolling_var(options)
            }

            #[cfg(feature = "cum_agg")]
            fn _cummax(&self, reverse: bool) -> Series {
                self.0.cummax(reverse).into_series()
            }

            #[cfg(feature = "cum_agg")]
            fn _cummin(&self, reverse: bool) -> Series {
                self.0.cummin(reverse).into_series()
            }

            fn set_sorted(&mut self, reverse: bool) {
                self.0.set_sorted(reverse)
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
                ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref())
                    .map(|ca| ca.into_series())
            }
            fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
                (&self.0).into_partial_eq_inner()
            }
            fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
                (&self.0).into_partial_ord_inner()
            }

            fn vec_hash(&self, random_state: RandomState) -> Vec<u64> {
                self.0.vec_hash(random_state)
            }

            fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
                self.0.vec_hash_combine(build_hasher, hashes)
            }

            fn agg_mean(&self, groups: &GroupsProxy) -> Option<Series> {
                self.agg_mean(groups)
            }

            fn agg_min(&self, groups: &GroupsProxy) -> Option<Series> {
                self.0.agg_min(groups)
            }

            fn agg_max(&self, groups: &GroupsProxy) -> Option<Series> {
                self.0.agg_max(groups)
            }

            fn agg_sum(&self, groups: &GroupsProxy) -> Option<Series> {
                self.0.agg_sum(groups)
            }

            fn agg_std(&self, groups: &GroupsProxy) -> Option<Series> {
                self.agg_std(groups)
            }

            fn agg_var(&self, groups: &GroupsProxy) -> Option<Series> {
                self.agg_var(groups)
            }

            fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
                self.0.agg_list(groups)
            }

            fn agg_quantile(
                &self,
                groups: &GroupsProxy,
                quantile: f64,
                interpol: QuantileInterpolOptions,
            ) -> Option<Series> {
                self.agg_quantile(groups, quantile, interpol)
            }

            fn agg_median(&self, groups: &GroupsProxy) -> Option<Series> {
                self.agg_median(groups)
            }
            fn zip_outer_join_column(
                &self,
                right_column: &Series,
                opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
            ) -> Series {
                ZipOuterJoinColumn::zip_outer_join_column(&self.0, right_column, opt_join_tuples)
            }
            fn subtract(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::subtract(&self.0, rhs)
            }
            fn add_to(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::add_to(&self.0, rhs)
            }
            fn multiply(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::multiply(&self.0, rhs)
            }
            fn divide(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::divide(&self.0, rhs)
            }
            fn remainder(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::remainder(&self.0, rhs)
            }
            fn group_tuples(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
                IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
            }

            #[cfg(feature = "sort_multiple")]
            fn argsort_multiple(&self, by: &[Series], reverse: &[bool]) -> Result<IdxCa> {
                self.0.argsort_multiple(by, reverse)
            }
        }

        impl SeriesTrait for SeriesWrap<$ca> {
            fn is_sorted(&self) -> IsSorted {
                if self.0.is_sorted() {
                    IsSorted::Ascending
                } else if self.0.is_sorted_reverse() {
                    IsSorted::Descending
                } else {
                    IsSorted::Not
                }
            }

            #[cfg(feature = "rolling_window")]
            fn rolling_apply(
                &self,
                _f: &dyn Fn(&Series) -> Series,
                _options: RollingOptions,
            ) -> Result<Series> {
                ChunkRollApply::rolling_apply(&self.0, _f, _options).map(|ca| ca.into_series())
            }

            #[cfg(feature = "interpolate")]
            fn interpolate(&self) -> Series {
                self.0.interpolate().into_series()
            }

            fn bitand(&self, other: &Series) -> Result<Series> {
                let other = if other.len() == 1 {
                    Cow::Owned(other.cast(self.dtype())?)
                } else {
                    Cow::Borrowed(other)
                };
                let other = self.0.unpack_series_matching_type(&other)?;
                Ok(self.0.bitand(&other).into_series())
            }

            fn bitor(&self, other: &Series) -> Result<Series> {
                let other = if other.len() == 1 {
                    Cow::Owned(other.cast(self.dtype())?)
                } else {
                    Cow::Borrowed(other)
                };
                let other = self.0.unpack_series_matching_type(&other)?;
                Ok(self.0.bitor(&other).into_series())
            }

            fn bitxor(&self, other: &Series) -> Result<Series> {
                let other = if other.len() == 1 {
                    Cow::Owned(other.cast(self.dtype())?)
                } else {
                    Cow::Borrowed(other)
                };
                let other = self.0.unpack_series_matching_type(&other)?;
                Ok(self.0.bitxor(&other).into_series())
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
            fn shrink_to_fit(&mut self) {
                self.0.shrink_to_fit()
            }

            fn f32(&self) -> Result<&Float32Chunked> {
                if matches!(self.0.dtype(), DataType::Float32) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Float32Chunked)) }
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into f32",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn f64(&self) -> Result<&Float64Chunked> {
                if matches!(self.0.dtype(), DataType::Float64) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Float64Chunked)) }
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into f64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn append_array(&mut self, other: ArrayRef) -> Result<()> {
                self.0.append_array(other)
            }

            fn slice(&self, offset: i64, length: usize) -> Series {
                return self.0.slice(offset, length).into_series();
            }

            fn append(&mut self, other: &Series) -> Result<()> {
                if self.0.dtype() == other.dtype() {
                    self.0.append(other.as_ref().as_ref());
                    Ok(())
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        "cannot append Series; data types don't match".into(),
                    ))
                }
            }

            fn extend(&mut self, other: &Series) -> Result<()> {
                if self.0.dtype() == other.dtype() {
                    self.0.extend(other.as_ref().as_ref());
                    Ok(())
                } else {
                    Err(PolarsError::SchemaMisMatch(
                        "cannot extend Series; data types don't match".into(),
                    ))
                }
            }

            fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
                ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
            }

            fn mean(&self) -> Option<f64> {
                self.0.mean()
            }

            fn median(&self) -> Option<f64> {
                self.0.median().map(|v| v as f64)
            }

            fn take(&self, indices: &IdxCa) -> Result<Series> {
                let indices = if indices.chunks.len() > 1 {
                    Cow::Owned(indices.rechunk())
                } else {
                    Cow::Borrowed(indices)
                };
                Ok(ChunkTake::take(&self.0, (&*indices).into())?.into_series())
            }

            fn take_iter(&self, iter: &mut dyn TakeIterator) -> Result<Series> {
                Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
            }

            fn take_every(&self, n: usize) -> Series {
                self.0.take_every(n).into_series()
            }

            unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
                ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
            }

            unsafe fn take_unchecked(&self, idx: &IdxCa) -> Result<Series> {
                let idx = if idx.chunks.len() > 1 {
                    Cow::Owned(idx.rechunk())
                } else {
                    Cow::Borrowed(idx)
                };
                Ok(ChunkTake::take_unchecked(&self.0, (&*idx).into()).into_series())
            }

            unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
                ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
            }

            #[cfg(feature = "take_opt_iter")]
            fn take_opt_iter(&self, iter: &mut dyn TakeIteratorNulls) -> Result<Series> {
                Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn rechunk(&self) -> Series {
                ChunkOps::rechunk(&self.0).into_series()
            }

            fn expand_at_index(&self, index: usize, length: usize) -> Series {
                ChunkExpandAtIndex::expand_at_index(&self.0, index, length).into_series()
            }

            fn cast(&self, data_type: &DataType) -> Result<Series> {
                self.0.cast(data_type)
            }

            fn to_dummies(&self) -> Result<DataFrame> {
                ToDummies::to_dummies(&self.0)
            }

            fn get(&self, index: usize) -> AnyValue {
                self.0.get_any_value(index)
            }

            #[inline]
            #[cfg(feature = "private")]
            unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
                self.0.get_any_value_unchecked(index)
            }

            fn sort_with(&self, options: SortOptions) -> Series {
                ChunkSort::sort_with(&self.0, options).into_series()
            }

            fn argsort(&self, options: SortOptions) -> IdxCa {
                ChunkSort::argsort(&self.0, options)
            }

            fn null_count(&self) -> usize {
                self.0.null_count()
            }

            fn has_validity(&self) -> bool {
                self.0.has_validity()
            }

            fn unique(&self) -> Result<Series> {
                ChunkUnique::unique(&self.0).map(|ca| ca.into_series())
            }

            fn n_unique(&self) -> Result<usize> {
                ChunkUnique::n_unique(&self.0)
            }

            fn arg_unique(&self) -> Result<IdxCa> {
                ChunkUnique::arg_unique(&self.0)
            }

            fn arg_min(&self) -> Option<usize> {
                ArgAgg::arg_min(&self.0)
            }

            fn arg_max(&self) -> Option<usize> {
                ArgAgg::arg_max(&self.0)
            }

            fn is_null(&self) -> BooleanChunked {
                self.0.is_null()
            }

            fn is_not_null(&self) -> BooleanChunked {
                self.0.is_not_null()
            }

            fn is_unique(&self) -> Result<BooleanChunked> {
                ChunkUnique::is_unique(&self.0)
            }

            fn is_duplicated(&self) -> Result<BooleanChunked> {
                ChunkUnique::is_duplicated(&self.0)
            }

            fn reverse(&self) -> Series {
                ChunkReverse::reverse(&self.0).into_series()
            }

            fn as_single_ptr(&mut self) -> Result<usize> {
                self.0.as_single_ptr()
            }

            fn shift(&self, periods: i64) -> Series {
                ChunkShift::shift(&self.0, periods).into_series()
            }

            fn fill_null(&self, strategy: FillNullStrategy) -> Result<Series> {
                ChunkFillNull::fill_null(&self.0, strategy).map(|ca| ca.into_series())
            }

            fn _sum_as_series(&self) -> Series {
                ChunkAggSeries::sum_as_series(&self.0)
            }
            fn max_as_series(&self) -> Series {
                ChunkAggSeries::max_as_series(&self.0)
            }
            fn min_as_series(&self) -> Series {
                ChunkAggSeries::min_as_series(&self.0)
            }
            fn median_as_series(&self) -> Series {
                QuantileAggSeries::median_as_series(&self.0)
            }
            fn var_as_series(&self) -> Series {
                VarAggSeries::var_as_series(&self.0)
            }
            fn std_as_series(&self) -> Series {
                VarAggSeries::std_as_series(&self.0)
            }
            fn quantile_as_series(
                &self,
                quantile: f64,
                interpol: QuantileInterpolOptions,
            ) -> Result<Series> {
                QuantileAggSeries::quantile_as_series(&self.0, quantile, interpol)
            }

            fn fmt_list(&self) -> String {
                FmtList::fmt_list(&self.0)
            }
            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(SeriesWrap(Clone::clone(&self.0)))
            }

            fn pow(&self, exponent: f64) -> Result<Series> {
                let f_err = || {
                    Err(PolarsError::InvalidOperation(
                        format!("power operation not supported on dtype {:?}", self.dtype()).into(),
                    ))
                };

                match self.dtype() {
                    DataType::Utf8 | DataType::List(_) | DataType::Boolean => f_err(),
                    DataType::Float32 => Ok(self.0.pow_f32(exponent as f32).into_series()),
                    _ => Ok(self.0.pow_f64(exponent).into_series()),
                }
            }

            fn peak_max(&self) -> BooleanChunked {
                self.0.peak_max()
            }

            fn peak_min(&self) -> BooleanChunked {
                self.0.peak_min()
            }

            #[cfg(feature = "is_in")]
            fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
                IsIn::is_in(&self.0, other)
            }
            #[cfg(feature = "repeat_by")]
            fn repeat_by(&self, by: &IdxCa) -> ListChunked {
                RepeatBy::repeat_by(&self.0, by)
            }

            #[cfg(feature = "checked_arithmetic")]
            fn checked_div(&self, rhs: &Series) -> Result<Series> {
                self.0.checked_div(rhs)
            }

            #[cfg(feature = "is_first")]
            fn is_first(&self) -> Result<BooleanChunked> {
                self.0.is_first()
            }

            #[cfg(feature = "mode")]
            fn mode(&self) -> Result<Series> {
                Ok(self.0.mode()?.into_series())
            }
        }
    };
}

impl_dyn_series!(Float32Chunked);
impl_dyn_series!(Float64Chunked);
