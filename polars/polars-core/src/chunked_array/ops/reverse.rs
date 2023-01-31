use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{CustomIterTools, NoNull};

impl<T> ChunkReverse<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn reverse(&self) -> ChunkedArray<T> {
        let mut out = if let Ok(slice) = self.cont_slice() {
            let ca: NoNull<ChunkedArray<T>> = slice.iter().rev().copied().collect_trusted();
            ca.into_inner()
        } else {
            self.into_iter().rev().collect_trusted()
        };
        out.rename(self.name());

        match self.is_sorted_flag2() {
            IsSorted::Ascending => out.set_sorted_flag(IsSorted::Descending),
            IsSorted::Descending => out.set_sorted_flag(IsSorted::Ascending),
            _ => {}
        }

        out
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl ChunkReverse<$arrow_type> for $ca_type {
            fn reverse(&self) -> Self {
                let mut ca: Self = self.into_iter().rev().collect_trusted();
                ca.rename(self.name());
                ca
            }
        }
    };
}

impl_reverse!(BooleanType, BooleanChunked);
impl_reverse!(Utf8Type, Utf8Chunked);
#[cfg(feature = "dtype-binary")]
impl_reverse!(BinaryType, BinaryChunked);
impl_reverse!(ListType, ListChunked);

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkReverse<ObjectType<T>> for ObjectChunked<T> {
    fn reverse(&self) -> Self {
        // Safety
        // we we know we don't get out of bounds
        unsafe { self.take_unchecked((0..self.len()).rev().into()) }
    }
}
