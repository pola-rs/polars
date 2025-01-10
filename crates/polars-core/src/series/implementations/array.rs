use std::any::Any;
use std::borrow::Cow;

use self::compare_inner::{TotalEqInner, TotalOrdInner};
use self::sort::arg_sort_row_fmt;
use super::{private, StatisticsFlags};
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::comparison::*;
use crate::chunked_array::AsSinglePtr;
#[cfg(feature = "algorithm_group_by")]
use crate::frame::group_by::*;
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;

impl private::PrivateSeries for SeriesWrap<ArrayChunked> {
    fn compute_len(&mut self) {
        self.0.compute_len()
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.ref_field().dtype()
    }

    fn _get_flags(&self) -> StatisticsFlags {
        self.0.get_flags()
    }

    fn _set_flags(&mut self, flags: StatisticsFlags) {
        self.0.set_flags(flags)
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref()).map(|ca| ca.into_series())
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
    }

    fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
        self.0.add_to(rhs)
    }

    fn subtract(&self, rhs: &Series) -> PolarsResult<Series> {
        self.0.subtract(rhs)
    }

    fn multiply(&self, rhs: &Series) -> PolarsResult<Series> {
        self.0.multiply(rhs)
    }
    fn divide(&self, rhs: &Series) -> PolarsResult<Series> {
        self.0.divide(rhs)
    }
    fn remainder(&self, rhs: &Series) -> PolarsResult<Series> {
        self.0.remainder(rhs)
    }

    fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
        invalid_operation_panic!(into_total_eq_inner, self)
    }
    fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
        invalid_operation_panic!(into_total_ord_inner, self)
    }
}

impl SeriesTrait for SeriesWrap<ArrayChunked> {
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

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        let slf = (*self).clone();
        let slf = slf.into_column();
        arg_sort_row_fmt(
            &[slf],
            options.descending,
            options.nulls_last,
            options.multithreaded,
        )
        .unwrap()
    }

    fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
        let idxs = self.arg_sort(options);
        Ok(unsafe { self.take_unchecked(&idxs) })
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (a, b) = self.0.split_at(offset);
        (a.into_series(), b.into_series())
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        let other = other.array()?;
        self.0.append(other)
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        self.0.extend(other.as_ref().as_ref())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
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

    fn cast(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(dtype, options)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_nulls(&self) -> bool {
        self.0.has_nulls()
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

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }
}
