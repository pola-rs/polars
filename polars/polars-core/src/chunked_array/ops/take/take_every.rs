use crate::prelude::*;
use crate::utils::NoNull;

impl<T> ChunkTakeEvery<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take_every(&self, n: usize) -> ChunkedArray<T> {
        if self.null_count() == 0 {
            let a: NoNull<_> = self.into_no_null_iter().step_by(n).collect();
            a.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<BooleanType> for BooleanChunked {
    fn take_every(&self, n: usize) -> BooleanChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<Utf8Type> for Utf8Chunked {
    fn take_every(&self, n: usize) -> Utf8Chunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<ListType> for ListChunked {
    fn take_every(&self, n: usize) -> ListChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<CategoricalType> for CategoricalChunked {
    fn take_every(&self, n: usize) -> CategoricalChunked {
        let mut ca = if self.null_count() == 0 {
            let ca: NoNull<UInt32Chunked> = self.into_no_null_iter().step_by(n).collect();
            ca.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.categorical_map = self.categorical_map.clone();
        ca.cast().unwrap()
    }
}
#[cfg(feature = "object")]
impl<T> ChunkTakeEvery<ObjectType<T>> for ObjectChunked<T> {
    fn take_every(&self, _n: usize) -> ObjectChunked<T> {
        todo!()
    }
}
