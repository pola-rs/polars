use crate::prelude::*;
use crate::utils::NoNull;
#[cfg(feature = "dtype-categorical")]
use std::ops::Deref;

impl<T> ChunkTakeEvery<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take_every(&self, n: usize) -> ChunkedArray<T> {
        let mut ca = if !self.has_validity() {
            let a: NoNull<_> = self.into_no_null_iter().step_by(n).collect();
            a.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.rename(self.name());
        ca
    }
}

impl ChunkTakeEvery<BooleanType> for BooleanChunked {
    fn take_every(&self, n: usize) -> BooleanChunked {
        let mut ca: Self = if !self.has_validity() {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.rename(self.name());
        ca
    }
}

impl ChunkTakeEvery<Utf8Type> for Utf8Chunked {
    fn take_every(&self, n: usize) -> Utf8Chunked {
        let mut ca: Self = if !self.has_validity() {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.rename(self.name());
        ca
    }
}

impl ChunkTakeEvery<ListType> for ListChunked {
    fn take_every(&self, n: usize) -> ListChunked {
        let mut ca: Self = if !self.has_validity() {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.rename(self.name());
        ca
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkTakeEvery<CategoricalType> for CategoricalChunked {
    fn take_every(&self, n: usize) -> CategoricalChunked {
        let mut ca: CategoricalChunked = self.deref().take_every(n).into();
        ca.categorical_map = self.categorical_map.clone();
        ca
    }
}
#[cfg(feature = "object")]
impl<T> ChunkTakeEvery<ObjectType<T>> for ObjectChunked<T> {
    fn take_every(&self, _n: usize) -> ObjectChunked<T> {
        todo!()
    }
}
