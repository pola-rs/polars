use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::NoNull;

/// A chunked array that is its own reverse, if it is one: a single chunk repeating one element.
fn reverses_to_itself<T: PolarsDataType>(ca: &ChunkedArray<T>) -> Option<ChunkedArray<T>> {
    let [chunk] = ca.chunks().as_slice() else {
        return None;
    };

    PlArray::is_scalar(&**chunk).then(|| ca.clone())
}

impl<T> ChunkReverse for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn reverse(&self) -> ChunkedArray<T> {
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }

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
                if let Some(ca) = reverses_to_itself(self) {
                    return ca;
                }
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
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }
        if self.is_empty() {
            return self.clone();
        };

        // Read out of the chunks by index, rather than collected back from a `Series` per element:
        // a collect carries no inner type of its own, so a column of nothing but nulls came back
        // as a `List(Null)` — the elements alone do not say what is under them.
        let idx = IdxCa::from_vec(
            PlSmallStr::EMPTY,
            (0..self.len() as IdxSize).rev().collect(),
        );
        // SAFETY: every index is below the length.
        let mut ca = unsafe { self.take_unchecked(&idx) };
        ca.rename(self.name().clone());
        ca
    }
}

impl ChunkReverse for BinaryChunked {
    fn reverse(&self) -> Self {
        if self.chunks.len() == 1 {
            if let Some(ca) = reverses_to_itself(self) {
                return ca;
            }

            // The views are reversed one per element, so a chunk that is not laid out flat is
            // written out first. The mask is reversed on its own, in whatever representation it
            // is in — a single bit stays a single bit.
            let chunk = self.downcast_iter().next().unwrap();
            let validity = chunk.validity().map(|v| PlBitmap::from(v).reversed());
            let arr = chunk.to_flat();
            let length = arr.len();
            let views = arr.views().iter().copied().rev().collect::<Vec<_>>();

            unsafe {
                let arr = PlBinaryViewArray::new_unchecked(
                    views.into(),
                    arr.data_buffers().clone(),
                    length,
                    validity,
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
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }

        // Read out of the chunks by index, as `ListChunked` does: the builder this used to push
        // into only exists for a numeric inner type, so every other one — a string, a boolean, a
        // list, a struct — reached a `todo!()` and panicked.
        let idx = IdxCa::from_vec(
            PlSmallStr::EMPTY,
            (0..self.len() as IdxSize).rev().collect(),
        );
        // SAFETY: every index is below the length.
        let mut ca = unsafe { self.take_unchecked(&idx) };
        ca.rename(self.name().clone());
        ca
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
