use std::ops::Not;

use polars_arrow::bitmap::Bitmap;

use super::*;
use crate::frame::group_by::aggregations::AggList;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};
use crate::utils::align_chunks_binary_ca_series;

impl PrivateSeriesNumeric for SeriesWrap<UnionChunked> {
    fn bit_repr(&self) -> Option<BitRepr> {
        None
    }
}

impl PrivateSeries for SeriesWrap<UnionChunked> {
    fn _field(&self) -> Cow<'_, Field> {
        Cow::Borrowed(self.0.ref_field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn compute_len(&mut self) {
        self.0.compute_len()
    }

    fn _get_flags(&self) -> StatisticsFlags {
        self.0.get_flags()
    }

    fn _set_flags(&mut self, flags: StatisticsFlags) {
        self.0.set_flags(flags);
    }

    fn vec_hash(
        &self,
        _build_hasher: PlSeedableRandomStateQuality,
        _buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        polars_bail!(opq = vec_hash, self.dtype())
    }

    fn vec_hash_combine(
        &self,
        _build_hasher: PlSeedableRandomStateQuality,
        _hashes: &mut [u64],
    ) -> PolarsResult<()> {
        polars_bail!(opq = vec_hash_combine, self.dtype())
    }

    fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
        invalid_operation_panic!(into_total_ord_inner, self)
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsType) -> Series {
        self.0.agg_list(groups)
    }
}

impl SeriesTrait for SeriesWrap<UnionChunked> {
    fn rename(&mut self, name: PlSmallStr) {
        self.0.rename(name)
    }

    fn chunk_lengths(&self) -> ChunkLenIter<'_> {
        self.0.chunk_lengths()
    }

    fn name(&self) -> &PlSmallStr {
        self.0.name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        &self.0.chunks
    }

    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (l, r) = self.0.split_at(offset);
        (l.into_series(), r.into_series())
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append(other.as_ref().as_ref())
    }

    fn append_owned(&mut self, other: Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append_owned(other.take_inner())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        self.0.extend(other.as_ref().as_ref())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        self.0.take(indices).map(|ca| ca.into_series())
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> Series {
        self.0.take_unchecked(idx).into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        self.0.take(indices).map(|ca| ca.into_series())
    }

    unsafe fn take_slice_unchecked(&self, idx: &[IdxSize]) -> Series {
        self.0.take_unchecked(idx).into_series()
    }

    fn deposit(&self, validity: &Bitmap) -> Series {
        self.0.deposit(validity).into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0.rechunk().into_owned().into_series()
    }

    fn with_validity(&self, validity: Option<Bitmap>) -> Series {
        self.0.clone().with_validity(validity).into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.0.new_from_index(index, length).into_series()
    }

    fn cast(&self, dtype: &DataType, _cast_options: CastOptions) -> PolarsResult<Series> {
        polars_ensure!(
            dtype == self.dtype(),
            InvalidOperation: "casting from {} to {} is not supported", self.dtype(), dtype
        );
        Ok(self.0.clone().into_series())
    }

    unsafe fn get_unchecked(&self, index: usize) -> AnyValue<'_> {
        self.0.get_any_value_unchecked(index)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_nulls(&self) -> bool {
        self.0.has_nulls()
    }

    fn is_null(&self) -> BooleanChunked {
        let iter = self.downcast_iter().map(|arr| {
            let bitmap = match arr.validity() {
                Some(valid) => valid.not(),
                None => Bitmap::new_with_value(false, arr.len()),
            };
            BooleanArray::from_data_default(bitmap, None)
        });
        BooleanChunked::from_chunk_iter(self.name().clone(), iter)
    }

    fn is_not_null(&self) -> BooleanChunked {
        let iter = self.downcast_iter().map(|arr| {
            let bitmap = match arr.validity() {
                Some(valid) => valid.clone(),
                None => Bitmap::new_with_value(true, arr.len()),
            };
            BooleanArray::from_data_default(bitmap, None)
        });
        BooleanChunked::from_chunk_iter(self.name().clone(), iter)
    }

    fn reverse(&self) -> Series {
        let len = self.len() as IdxSize;
        let idx: Vec<IdxSize> = (0..len).rev().collect();
        // SAFETY: the indices are in bounds.
        unsafe { self.0.take_unchecked(idx.as_slice()).into_series() }
    }

    fn shift(&self, periods: i64) -> Series {
        self.0.shift(periods).into_series()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        let (slf, other) = align_chunks_binary_ca_series(&self.0, other);
        let mut offset: IdxSize = 0;
        for (l, r) in slf.downcast_iter().zip(other.chunks()) {
            let start_length = idxs.len();
            polars_compute::find_validity_mismatch::find_validity_mismatch_shallow(
                l.validity(),
                r.validity(),
                idxs,
            );
            for idx in idxs[start_length..].iter_mut() {
                *idx += offset;
            }
            offset += l.len() as IdxSize;
        }
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }

    fn as_phys_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self as _
    }
}
