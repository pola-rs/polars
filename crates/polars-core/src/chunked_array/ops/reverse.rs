use arrow::bitmap::Bitmap;

#[cfg(feature = "dtype-array")]
use crate::chunked_array::array::array_values;
#[cfg(feature = "dtype-array")]
use crate::chunked_array::builder::get_fixed_size_list_builder;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::NoNull;

impl<T> ChunkReverse for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn reverse(&self) -> ChunkedArray<T> {
        let mut out = if let Some(slice) = self.as_flat().and_then(|ca| ca.cont_slice().ok()) {
            let ca: NoNull<ChunkedArray<T>> = slice.iter().rev().copied().collect_trusted();
            ca.into_inner()
        } else {
            self.iter().rev().collect_trusted()
        };
        out.rename(self.name().clone());

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
                if self.is_empty() {
                    return self.clone();
                };
                let mut ca: Self = self.iter().rev().collect_trusted();
                ca.rename(self.name().clone());
                ca
            }
        }
    };
}

impl_reverse!(BooleanType, BooleanChunked);
impl_reverse!(BinaryOffsetType, BinaryOffsetChunked);

impl ChunkReverse for ListChunked {
    fn reverse(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        };
        let ca: Self = self.series_iter().rev().collect_trusted();
        ca.with_name(self.name().clone())
    }
}

impl ChunkReverse for BinaryChunked {
    fn reverse(&self) -> Self {
        if self.chunks.len() == 1 {
            // The views are reversed one per element, so a chunk that is not laid out flat is
            // written out first.
            let arr = self.downcast_iter().next().unwrap().to_flat();
            let length = arr.len();
            let views = arr.views().iter().copied().rev().collect::<Vec<_>>();
            let validity = arr
                .validity()
                .map(|bitmap| bitmap.iter().rev().collect::<Bitmap>());

            unsafe {
                let arr = PlBinaryViewArray::new_unchecked(
                    views.into(),
                    arr.data_buffers().clone(),
                    length,
                    validity.map(PlBitmap::from_bitmap),
                )
                .into_boxed();
                BinaryChunked::from_chunks_and_dtype_unchecked(
                    self.name().clone(),
                    vec![arr],
                    self.dtype().clone(),
                )
            }
        } else {
            let ca = IdxCa::from_vec(
                PlSmallStr::EMPTY,
                (0..self.len() as IdxSize).rev().collect(),
            );
            unsafe { self.take_unchecked(&ca) }
        }
    }
}

impl ChunkReverse for StringChunked {
    fn reverse(&self) -> Self {
        unsafe { self.as_binary().reverse().to_string_unchecked() }
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkReverse for ArrayChunked {
    fn reverse(&self) -> Self {
        if !self.inner_dtype().is_primitive_numeric() {
            todo!("reverse for FixedSizeList with non-numeric dtypes not yet supported")
        }
        let ca = self.rechunk();
        let arr = ca.downcast_as_array();
        let values = array_values(arr);
        let values = &*values;

        let mut builder =
            get_fixed_size_list_builder(ca.inner_dtype(), ca.len(), ca.width(), ca.name().clone())
                .expect("not yet supported");

        // SAFETY, we are within bounds
        unsafe {
            if arr.null_count() == 0 {
                for i in (0..arr.len()).rev() {
                    builder.push_unchecked(values, i)
                }
            } else {
                let validity = arr.validity().unwrap().to_flat();
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
        unsafe {
            self.take_unchecked(
                &(0..self.len() as IdxSize)
                    .rev()
                    .collect_ca(PlSmallStr::EMPTY),
            )
        }
    }
}
