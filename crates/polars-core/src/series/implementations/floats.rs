use super::*;
use crate::chunked_array::comparison::*;
#[cfg(feature = "algorithm_group_by")]
use crate::frame::group_by::*;
use crate::prelude::*;

macro_rules! impl_dyn_series {
    ($ca: ident, $pdt:ident) => {
        impl private::PrivateSeries for SeriesWrap<$ca> {
            fn compute_len(&mut self) {
                self.0.compute_len()
            }
            fn _field(&self) -> Cow<Field> {
                Cow::Borrowed(self.0.ref_field())
            }
            fn _dtype(&self) -> &DataType {
                self.0.ref_field().dtype()
            }

            fn _set_flags(&mut self, flags: StatisticsFlags) {
                self.0.set_flags(flags)
            }
            fn _get_flags(&self) -> StatisticsFlags {
                self.0.get_flags()
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

            fn vec_hash(
                &self,
                random_state: PlRandomState,
                buf: &mut Vec<u64>,
            ) -> PolarsResult<()> {
                self.0.vec_hash(random_state, buf)?;
                Ok(())
            }

            fn vec_hash_combine(
                &self,
                build_hasher: PlRandomState,
                hashes: &mut [u64],
            ) -> PolarsResult<()> {
                self.0.vec_hash_combine(build_hasher, hashes)?;
                Ok(())
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_min(&self, groups: &GroupsType) -> Series {
                self.0.agg_min(groups)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_max(&self, groups: &GroupsType) -> Series {
                self.0.agg_max(groups)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_sum(&self, groups: &GroupsType) -> Series {
                self.0.agg_sum(groups)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_std(&self, groups: &GroupsType, ddof: u8) -> Series {
                self.agg_std(groups, ddof)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_var(&self, groups: &GroupsType, ddof: u8) -> Series {
                self.agg_var(groups, ddof)
            }

            #[cfg(feature = "algorithm_group_by")]
            unsafe fn agg_list(&self, groups: &GroupsType) -> Series {
                self.0.agg_list(groups)
            }

            #[cfg(feature = "bitwise")]
            unsafe fn agg_and(&self, groups: &GroupsType) -> Series {
                self.0.agg_and(groups)
            }
            #[cfg(feature = "bitwise")]
            unsafe fn agg_or(&self, groups: &GroupsType) -> Series {
                self.0.agg_or(groups)
            }
            #[cfg(feature = "bitwise")]
            unsafe fn agg_xor(&self, groups: &GroupsType) -> Series {
                self.0.agg_xor(groups)
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
            fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsType> {
                IntoGroupsType::group_tuples(&self.0, multithreaded, sorted)
            }

            fn arg_sort_multiple(
                &self,
                by: &[Column],
                options: &SortMultipleOptions,
            ) -> PolarsResult<IdxCa> {
                self.0.arg_sort_multiple(by, options)
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

            fn rename(&mut self, name: PlSmallStr) {
                self.0.rename(name);
            }

            fn chunk_lengths(&self) -> ChunkLenIter {
                self.0.chunk_lengths()
            }
            fn name(&self) -> &PlSmallStr {
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

            fn split_at(&self, offset: i64) -> (Series, Series) {
                let (a, b) = self.0.split_at(offset);
                (a.into_series(), b.into_series())
            }

            fn append(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), append);
                self.0.append(other.as_ref().as_ref())?;
                Ok(())
            }

            fn extend(&mut self, other: &Series) -> PolarsResult<()> {
                polars_ensure!(self.0.dtype() == other.dtype(), extend);
                self.0.extend(other.as_ref().as_ref())?;
                Ok(())
            }

            fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
                ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
            }

            fn _sum_as_f64(&self) -> f64 {
                self.0._sum_as_f64()
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

            fn cast(&self, dtype: &DataType, cast_options: CastOptions) -> PolarsResult<Series> {
                self.0.cast_with_options(dtype, cast_options)
            }

            #[inline]
            unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
                self.0.get_any_value_unchecked(index)
            }

            fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
                Ok(ChunkSort::sort_with(&self.0, options).into_series())
            }

            fn arg_sort(&self, options: SortOptions) -> IdxCa {
                ChunkSort::arg_sort(&self.0, options)
            }

            fn null_count(&self) -> usize {
                self.0.null_count()
            }

            fn has_nulls(&self) -> bool {
                self.0.has_nulls()
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

            fn sum_reduce(&self) -> PolarsResult<Scalar> {
                Ok(ChunkAggSeries::sum_reduce(&self.0))
            }
            fn max_reduce(&self) -> PolarsResult<Scalar> {
                Ok(ChunkAggSeries::max_reduce(&self.0))
            }
            fn min_reduce(&self) -> PolarsResult<Scalar> {
                Ok(ChunkAggSeries::min_reduce(&self.0))
            }
            fn median_reduce(&self) -> PolarsResult<Scalar> {
                Ok(QuantileAggSeries::median_reduce(&self.0))
            }
            fn var_reduce(&self, ddof: u8) -> PolarsResult<Scalar> {
                Ok(VarAggSeries::var_reduce(&self.0, ddof))
            }
            fn std_reduce(&self, ddof: u8) -> PolarsResult<Scalar> {
                Ok(VarAggSeries::std_reduce(&self.0, ddof))
            }
            fn quantile_reduce(
                &self,
                quantile: f64,
                method: QuantileMethod,
            ) -> PolarsResult<Scalar> {
                QuantileAggSeries::quantile_reduce(&self.0, quantile, method)
            }
            #[cfg(feature = "bitwise")]
            fn and_reduce(&self) -> PolarsResult<Scalar> {
                let dt = <$pdt as PolarsDataType>::get_dtype();
                let av = self.0.and_reduce().map_or(AnyValue::Null, Into::into);

                Ok(Scalar::new(dt, av))
            }
            #[cfg(feature = "bitwise")]
            fn or_reduce(&self) -> PolarsResult<Scalar> {
                let dt = <$pdt as PolarsDataType>::get_dtype();
                let av = self.0.or_reduce().map_or(AnyValue::Null, Into::into);

                Ok(Scalar::new(dt, av))
            }
            #[cfg(feature = "bitwise")]
            fn xor_reduce(&self) -> PolarsResult<Scalar> {
                let dt = <$pdt as PolarsDataType>::get_dtype();
                let av = self.0.xor_reduce().map_or(AnyValue::Null, Into::into);

                Ok(Scalar::new(dt, av))
            }

            #[cfg(feature = "approx_unique")]
            fn approx_n_unique(&self) -> PolarsResult<IdxSize> {
                Ok(ChunkApproxNUnique::approx_n_unique(&self.0))
            }

            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(SeriesWrap(Clone::clone(&self.0)))
            }

            #[cfg(feature = "checked_arithmetic")]
            fn checked_div(&self, rhs: &Series) -> PolarsResult<Series> {
                self.0.checked_div(rhs)
            }

            fn as_any(&self) -> &dyn Any {
                &self.0
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                &mut self.0
            }
        }
    };
}

impl_dyn_series!(Float32Chunked, Float32Type);
impl_dyn_series!(Float64Chunked, Float64Type);
