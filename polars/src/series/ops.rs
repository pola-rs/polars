use crate::fmt::FmtList;
use crate::frame::group_by::{AggFirst, AggNUnique};
use crate::frame::hash_join::ZipOuterJoinColumn;
use crate::prelude::*;
use arrow::array::ArrayRef;
use arrow::datatypes::DataType;
use std::any::Any;
use std::fmt::Debug;

pub trait SeriesOps: Send + Sync + Debug + ZipOuterJoinColumn {
    fn ref_field(&self) -> &Field;
    fn fmt_list(&self) -> String;
    fn append(&mut self, other: &dyn SeriesOps);
    fn rename(&mut self, name: &str);
    fn chunk_id(&self) -> &Vec<usize>;
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Box<dyn SeriesOps>>;
    fn is_null(&self) -> BooleanChunked;
    fn is_not_null(&self) -> BooleanChunked;
    fn zip_with_series(&self, mask: &BooleanChunked, other: &Series) -> Result<Box<dyn SeriesOps>>;
    fn shift(&self, periods: i32, fill_value: &Option<u8>) -> Result<Box<dyn SeriesOps>>;
    fn name(&self) -> &str;
    fn get_any(&self, index: usize) -> AnyType;
    fn get_as_any(&self, index: usize) -> &dyn Any;
    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Box<dyn SeriesOps>>;
    fn reverse(&self) -> Box<dyn SeriesOps>;
    fn null_count(&self) -> usize;
    fn head(&self, length: Option<usize>) -> Box<dyn SeriesOps>;
    fn tail(&self, length: Option<usize>) -> Box<dyn SeriesOps>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn chunks(&self) -> &Vec<ArrayRef>;
    fn chunks_mut(&mut self) -> &mut Vec<ArrayRef>;
    fn dtype(&self) -> &ArrowDataType;
    fn agg_n_unique(&self, _groups: &[(usize, Vec<usize>)]) -> Option<UInt32Chunked>;
    fn agg_first(&self, _groups: &[(usize, Vec<usize>)]) -> Series;
    fn take(
        &self,
        indices: &mut dyn Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps>;
    unsafe fn take_unchecked(
        &self,
        indices: &mut dyn Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps>;
    fn take_opt(
        &self,
        indices: &mut dyn Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps>;
    unsafe fn take_opt_unchecked(
        &self,
        indices: &mut dyn Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps>;
    fn expand_at_index(&self, index: usize, length: usize) -> Box<dyn SeriesOps>;
    fn filter(&self, filter: &BooleanChunked) -> Result<Box<dyn SeriesOps>>;
    fn limit(&self, num_elements: usize) -> Result<Box<dyn SeriesOps>>;
    fn slice(&self, offset: usize, length: usize) -> Result<Box<dyn SeriesOps>>;
    fn clone(&self) -> Box<dyn SeriesOps>;

    #[cfg(feature = "random")]
    #[doc(cfg(feature = "random"))]
    fn sample_n(&self, n: usize) -> Result<Box<dyn SeriesOps>>;
}

fn to_object_chunked<T>(mut ca: Box<dyn SeriesOps>) -> ObjectChunked<T>
where
    T: Any + Debug + Clone + Send + Sync,
{
    let chunks = ca.chunks_mut();
    let chunks = std::mem::take(chunks);
    ObjectChunked::new_from_chunks(ca.name(), chunks)
}

impl<T> ObjectChunked<T>
where
    T: Any + Debug + Clone + Send + Sync + Default,
{
    fn as_series_ops(self) -> Box<dyn SeriesOps> {
        Box::new(self)
    }
}

impl<T> From<ObjectChunked<T>> for Box<dyn SeriesOps>
where
    T: Any + Debug + Clone + Send + Sync + Default,
{
    fn from(ca: ObjectChunked<T>) -> Self {
        Box::new(ca)
    }
}

impl<T> SeriesOps for ObjectChunked<T>
where
    T: 'static + Debug + Clone + Send + Sync + Default,
{
    fn ref_field(&self) -> &Field {
        ObjectChunked::ref_field(self)
    }
    fn agg_n_unique(&self, groups: &[(usize, Vec<usize>)]) -> Option<UInt32Chunked> {
        AggNUnique::agg_n_unique(self, groups)
    }

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(self)
    }

    fn append(&mut self, other: &dyn SeriesOps) {
        let other = to_object_chunked(other.clone());
        ObjectChunked::append(self, &other)
    }

    fn rename(&mut self, name: &str) {
        ObjectChunked::rename(self, name)
    }
    fn agg_first(&self, groups: &[(usize, Vec<usize>)]) -> Series {
        AggFirst::agg_first(self, groups)
    }

    fn chunk_id(&self) -> &Vec<usize> {
        ObjectChunked::chunk_id(self)
    }

    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Box<dyn SeriesOps>> {
        ChunkFillNone::fill_none(self, strategy).map(|ca| ca.as_series_ops())
    }

    fn is_null(&self) -> BooleanChunked {
        ObjectChunked::is_null(self)
    }

    fn is_not_null(&self) -> BooleanChunked {
        ObjectChunked::is_not_null(self)
    }

    fn zip_with_series(&self, mask: &BooleanChunked, other: &Series) -> Result<Box<dyn SeriesOps>> {
        ChunkZip::zip_with_series(self, mask, other).map(|ca| ca.as_series_ops())
    }

    fn shift(&self, periods: i32, _fill_value: &Option<u8>) -> Result<Box<dyn SeriesOps>> {
        ChunkShift::shift(self, periods, &None).map(|ca| ca.as_series_ops())
    }

    fn name(&self) -> &str {
        ObjectChunked::name(self)
    }

    fn get_any(&self, index: usize) -> AnyType<'_> {
        ObjectChunked::get_any(self, index)
    }

    fn get_as_any(&self, index: usize) -> &dyn Any {
        ObjectChunked::get_as_any(self, index)
    }

    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Box<dyn SeriesOps>> {
        ChunkOps::rechunk(self, chunk_lengths).map(|ca| ca.as_series_ops())
    }

    fn reverse(&self) -> Box<dyn SeriesOps> {
        ChunkReverse::reverse(self).as_series_ops()
    }

    fn null_count(&self) -> usize {
        ObjectChunked::null_count(self)
    }

    fn head(&self, length: Option<usize>) -> Box<dyn SeriesOps> {
        ObjectChunked::head(self, length).as_series_ops()
    }

    fn tail(&self, length: Option<usize>) -> Box<dyn SeriesOps> {
        ObjectChunked::tail(self, length).as_series_ops()
    }

    fn len(&self) -> usize {
        ObjectChunked::len(self)
    }
    fn is_empty(&self) -> bool {
        ObjectChunked::is_empty(self)
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }
    fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        &mut self.chunks
    }

    fn dtype(&self) -> &DataType {
        ObjectChunked::dtype(self)
    }

    fn take(
        &self,
        indices: &mut dyn Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps> {
        ChunkTake::take(self, indices, capacity).into()
    }

    unsafe fn take_unchecked(
        &self,
        indices: &mut dyn Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps> {
        ChunkTake::take_unchecked(self, indices, capacity).into()
    }

    fn take_opt(
        &self,
        indices: &mut dyn Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps> {
        ChunkTake::take_opt(self, indices, capacity).into()
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: &mut dyn Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Box<dyn SeriesOps> {
        ChunkTake::take_opt_unchecked(self, indices, capacity).into()
    }

    fn expand_at_index(&self, index: usize, length: usize) -> Box<dyn SeriesOps> {
        ChunkExpandAtIndex::expand_at_index(self, index, length).as_series_ops()
    }

    fn filter(&self, filter: &BooleanChunked) -> Result<Box<dyn SeriesOps>> {
        ChunkFilter::filter(self, filter).map(|ca| ca.as_series_ops())
    }

    fn limit(&self, num_elements: usize) -> Result<Box<dyn SeriesOps>> {
        ObjectChunked::limit(self, num_elements).map(|ca| ca.as_series_ops())
    }

    fn slice(&self, offset: usize, length: usize) -> Result<Box<dyn SeriesOps>> {
        ObjectChunked::slice(self, offset, length).map(|ca| ca.as_series_ops())
    }
    fn clone(&self) -> Box<dyn SeriesOps> {
        Clone::clone(self).as_series_ops()
    }

    #[cfg(feature = "random")]
    #[doc(cfg(feature = "random"))]
    fn sample_n(&self, n: usize) -> Result<Box<dyn SeriesOps>> {
        self.sample_n(n).map(|ca| ca.as_series_ops())
    }
}
