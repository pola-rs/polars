use super::private;
use super::IntoSeries;
use super::SeriesTrait;
use crate::chunked_array::comparison::*;
use crate::chunked_array::{
    ops::{
        compare_inner::{IntoPartialEqInner, IntoPartialOrdInner, PartialEqInner, PartialOrdInner},
        explode::ExplodeByOffsets,
    },
    AsSinglePtr, ChunkIdIter,
};
use crate::fmt::FmtList;
#[cfg(feature = "pivot")]
use crate::frame::groupby::pivot::*;
use crate::frame::groupby::*;
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use ahash::RandomState;
use arrow::array::ArrayRef;
#[cfg(feature = "object")]
use std::any::Any;
use std::borrow::Cow;

impl IntoSeries for ListChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeries for SeriesWrap<ListChunked> {
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.ref_field().data_type()
    }
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0.explode_by_offsets(offsets)
    }

    fn set_sorted(&mut self, reverse: bool) {
        self.0.set_sorted(reverse)
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
        ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref()).map(|ca| ca.into_series())
    }
    fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
        (&self.0).into_partial_eq_inner()
    }
    fn into_partial_ord_inner<'a>(&'a self) -> Box<dyn PartialOrdInner + 'a> {
        (&self.0).into_partial_ord_inner()
    }

    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        self.0.vec_hash(random_state)
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
        self.0.vec_hash_combine(build_hasher, hashes)
    }

    fn agg_first(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        self.0.agg_first(groups)
    }

    fn agg_last(&self, groups: &[(u32, Vec<u32>)]) -> Series {
        self.0.agg_last(groups)
    }

    fn agg_n_unique(&self, groups: &[(u32, Vec<u32>)]) -> Option<UInt32Chunked> {
        self.0.agg_n_unique(groups)
    }

    fn agg_list(&self, groups: &[(u32, Vec<u32>)]) -> Option<Series> {
        self.0.agg_list(groups)
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
    fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
        IntoGroupTuples::group_tuples(&self.0, multithreaded)
    }

    fn str_value(&self, index: usize) -> Cow<str> {
        // get AnyValue
        Cow::Owned(format!("{}", self.get(index)))
    }
}

impl SeriesTrait for SeriesWrap<ListChunked> {
    #[cfg(feature = "interpolate")]
    fn interpolate(&self) -> Series {
        self.0.clone().into_series()
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

    fn list(&self) -> Result<&ListChunked> {
        unsafe { Ok(&*(self as *const dyn SeriesTrait as *const ListChunked)) }
    }

    fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        self.0.append_array(other)
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn append(&mut self, other: &Series) -> Result<()> {
        if self.0.dtype() == other.dtype() {
            // todo! add object
            self.0.append(other.as_ref().as_ref());
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }

    fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
    }

    fn take(&self, indices: &UInt32Chunked) -> Result<Series> {
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

    unsafe fn take_unchecked(&self, idx: &UInt32Chunked) -> Result<Series> {
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

    fn head(&self, length: Option<usize>) -> Series {
        self.0.head(length).into_series()
    }

    fn tail(&self, length: Option<usize>) -> Series {
        self.0.tail(length).into_series()
    }

    fn expand_at_index(&self, index: usize, length: usize) -> Series {
        ChunkExpandAtIndex::expand_at_index(&self.0, index, length).into_series()
    }

    fn cast(&self, data_type: &DataType) -> Result<Series> {
        self.0.cast(data_type)
    }

    fn get(&self, index: usize) -> AnyValue {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_validity(&self) -> bool {
        self.0.has_validity()
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
    fn mean_as_series(&self) -> Series {
        ChunkAggSeries::mean_as_series(&self.0)
    }
    fn median_as_series(&self) -> Series {
        ChunkAggSeries::median_as_series(&self.0)
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
        ChunkAggSeries::quantile_as_series(&self.0, quantile, interpol)
    }

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }
    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    #[cfg(feature = "object")]
    fn as_any(&self) -> &dyn Any {
        &self.0
    }
}
