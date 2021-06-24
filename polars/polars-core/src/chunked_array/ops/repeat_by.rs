use super::RepeatBy;
use crate::prelude::*;
use arrow::array::ListArray;
use polars_arrow::array::ListFromIter;

type LargeListArray = ListArray<i64>;

impl<T> RepeatBy for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn repeat_by(&self, by: &UInt32Chunked) -> ListChunked {
        let iter = self
            .into_iter()
            .zip(by.into_iter())
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize)));

        // Safety:
        // Length of iter is trusted
        ListChunked::new_from_chunks(
            self.name(),
            vec![Arc::new(unsafe {
                LargeListArray::from_iter_primitive_trusted_len::<T::Native, _, _>(
                    iter,
                    T::get_dtype().to_arrow(),
                )
            })],
        )
    }
}
impl RepeatBy for BooleanChunked {
    fn repeat_by(&self, by: &UInt32Chunked) -> ListChunked {
        let iter = self
            .into_iter()
            .zip(by.into_iter())
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize)));

        // Safety:
        // Length of iter is trusted
        ListChunked::new_from_chunks(
            self.name(),
            vec![Arc::new(unsafe {
                LargeListArray::from_iter_bool_trusted_len(iter)
            })],
        )
    }
}
impl RepeatBy for Utf8Chunked {
    fn repeat_by(&self, by: &UInt32Chunked) -> ListChunked {
        let iter = self
            .into_iter()
            .zip(by.into_iter())
            .map(|(opt_v, opt_by)| opt_by.map(|by| std::iter::repeat(opt_v).take(by as usize)));

        // Safety:
        // Length of iter is trusted
        ListChunked::new_from_chunks(
            self.name(),
            vec![Arc::new(unsafe {
                LargeListArray::from_iter_utf8_trusted_len(iter, self.len())
            })],
        )
    }
}

impl RepeatBy for ListChunked {}

impl RepeatBy for CategoricalChunked {
    fn repeat_by(&self, by: &UInt32Chunked) -> ListChunked {
        let mut ca = self.cast::<UInt32Type>().unwrap().repeat_by(by);
        ca.categorical_map = self.categorical_map.clone();
        ca
    }
}
#[cfg(feature = "object")]
impl<T> RepeatBy for ObjectChunked<T> {}
