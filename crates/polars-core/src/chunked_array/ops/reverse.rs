#[cfg(feature = "dtype-array")]
use crate::chunked_array::builder::get_fixed_size_list_builder;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{CustomIterTools, NoNull};

impl<T> ChunkReverse for ChunkedArray<T>
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

        match self.is_sorted_flag() {
            IsSorted::Ascending => out.set_sorted_flag(IsSorted::Descending),
            IsSorted::Descending => out.set_sorted_flag(IsSorted::Ascending),
            _ => {},
        }

        out
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl ChunkReverse for $ca_type {
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
impl_reverse!(BinaryType, BinaryChunked);
impl_reverse!(ListType, ListChunked);

#[cfg(feature = "dtype-array")]
impl ChunkReverse for ArrayChunked {
    fn reverse(&self) -> Self {
        if !self.inner_dtype().is_numeric() {
            todo!("reverse for FixedSizeList with non-numeric dtypes not yet supported")
        }
        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        let values = arr.values().as_ref();

        let mut builder =
            get_fixed_size_list_builder(&ca.inner_dtype(), ca.len(), ca.width(), ca.name())
                .expect("not yet supported");

        // safety, we are within bounds
        unsafe {
            if arr.null_count() == 0 {
                for i in (0..arr.len()).rev() {
                    builder.push_unchecked(values, i)
                }
            } else {
                let validity = arr.validity().unwrap();
                for i in (0..arr.len()).rev() {
                    if validity.get_bit_unchecked(i) {
                        builder.push_unchecked(values, i)
                    } else {
                        builder.push_null()
                    }
                }
            }
        }
        builder.finish()
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkReverse for ObjectChunked<T> {
    fn reverse(&self) -> Self {
        // SAFETY: we know we don't go out of bounds.
        unsafe { self.take_unchecked(&(0..self.len() as IdxSize).rev().collect_ca("")) }
    }
}
