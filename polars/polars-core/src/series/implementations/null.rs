use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::array::PolarsArray;
use polars_arrow::prelude::ArrayRef;
use polars_utils::IdxSize;

use crate::datatypes::IdxCa;
use crate::error::PolarsResult;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};
use crate::series::*;

impl Series {
    pub fn new_null(name: &str, len: usize) -> Series {
        NullChunked::new(name, len).into_series()
    }
}

#[derive(Clone)]
pub struct NullChunked {
    pub(crate) field: Arc<Field>,
    length: IdxSize,
    chunks: Vec<ArrayRef>,
}

impl NullChunked {
    fn new(name: &str, len: usize) -> Self {
        Self {
            field: Arc::new(Field::new(name, DataType::Null)),
            length: len as IdxSize,
            chunks: vec![Box::new(arrow::array::NullArray::new(
                ArrowDataType::Null,
                len,
            ))],
        }
    }
}
impl PrivateSeriesNumeric for NullChunked {}

impl PrivateSeries for NullChunked {
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.field.as_ref())
    }

    fn _dtype(&self) -> &DataType {
        &self.field.dtype
    }
}

impl SeriesTrait for NullChunked {
    fn name(&self) -> &str {
        self.field.name.as_ref()
    }

    fn rename(&mut self, name: &str) {
        let mut field = (*self.field).clone();
        field.set_name(name.to_string());
        self.field = Arc::new(field);
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], _sorted: IsSorted) -> Series {
        NullChunked::new(self.name(), by.len()).into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        NullChunked::new(self.name(), by.len()).into_series()
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        Ok(NullChunked::new(&self.field.name, iter.size_hint().0).into_series())
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        NullChunked::new(&self.field.name, iter.size_hint().0).into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        Ok(NullChunked::new(&self.field.name, idx.len()).into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        NullChunked::new(&self.field.name, iter.size_hint().0).into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(NullChunked::new(&self.field.name, indices.len()).into_series())
    }

    fn len(&self) -> usize {
        self.length as usize
    }

    fn take_every(&self, n: usize) -> Series {
        NullChunked::new(&self.field.name, self.len() / n).into_series()
    }

    fn has_validity(&self) -> bool {
        self.chunks[0].has_validity()
    }

    fn rechunk(&self) -> Series {
        if self.chunks.len() != 1 {
            NullChunked::new(self.name(), self.len()).into_series()
        } else {
            self.clone().into_series()
        }
    }

    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        Ok(Series::full_null(self.name(), self.len(), data_type))
    }

    fn null_count(&self) -> usize {
        self.len()
    }
}

unsafe impl IntoSeries for NullChunked {
    fn into_series(self) -> Series
    where
        Self: Sized,
    {
        Series(Arc::new(self))
    }
}
