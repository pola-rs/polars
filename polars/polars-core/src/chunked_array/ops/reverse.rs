use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};

impl<T> ChunkReverse<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkOps,
{
    fn reverse(&self) -> ChunkedArray<T> {
        if let Ok(slice) = self.cont_slice() {
            let ca: NoNull<ChunkedArray<T>> = slice.iter().rev().copied().collect_trusted();
            let mut ca = ca.into_inner();
            ca.rename(self.name());
            ca
        } else {
            self.into_iter().rev().collect_trusted()
        }
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkReverse<CategoricalType> for CategoricalChunked {
    fn reverse(&self) -> ChunkedArray<CategoricalType> {
        self.cast::<UInt32Type>().unwrap().reverse().cast().unwrap()
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl ChunkReverse<$arrow_type> for $ca_type {
            fn reverse(&self) -> Self {
                self.into_iter().rev().collect_trusted()
            }
        }
    };
}

impl_reverse!(BooleanType, BooleanChunked);
impl_reverse!(Utf8Type, Utf8Chunked);
impl_reverse!(ListType, ListChunked);

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkReverse<ObjectType<T>> for ObjectChunked<T> {
    fn reverse(&self) -> Self {
        // Safety
        // we we know we don't get out of bounds
        unsafe { self.take_unchecked((0..self.len()).rev().into()) }
    }
}
