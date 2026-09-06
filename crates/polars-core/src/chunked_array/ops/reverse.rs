#[cfg(feature = "dtype-array")]
use crate::chunked_array::array::array_values;
#[cfg(feature = "dtype-array")]
use crate::chunked_array::builder::get_fixed_size_list_builder;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::NoNull;

/// A chunked array that is its own reverse, if it is one: a single chunk that repeats a single
/// element reads as that same element whichever way it is walked, so nothing has to be written
/// out. Several chunks reverse among themselves, which is why only one of them will do.
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
        let ca: Self = self.series_iter().rev().collect_trusted();
        ca.with_name(self.name().clone())
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
                let validity = arr.validity().unwrap();
                for i in (0..arr.len()).rev() {
                    if validity.get_unchecked(i) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether the single chunk of `ca` repeats one element rather than holding one slot each.
    fn is_repeated<T: PolarsDataType>(ca: &ChunkedArray<T>) -> bool {
        let [chunk] = ca.chunks().as_slice() else {
            return false;
        };

        PlArray::is_scalar(&**chunk)
    }

    /// A chunk that repeats one element reads as that element whichever way it is walked, so it
    /// is handed back as it is rather than written out backwards.
    #[test]
    fn a_repeated_element_is_its_own_reverse() {
        let name = PlSmallStr::from_static("a");

        let ints = Int32Chunked::full(name.clone(), 7, 1_000);
        assert!(is_repeated(&ints.reverse()));
        assert_eq!(ints.reverse().len(), 1_000);
        assert_eq!(ints.reverse().get(0), Some(7));

        let bools = BooleanChunked::full(name.clone(), true, 1_000);
        assert!(is_repeated(&bools.reverse()));

        let strings = StringChunked::full(name.clone(), "abc", 1_000);
        assert!(is_repeated(&strings.reverse()));
        assert_eq!(strings.reverse().get(999), Some("abc"));

        let nulls = Int32Chunked::full_null(name.clone(), 1_000);
        assert!(is_repeated(&nulls.reverse()));
        assert_eq!(nulls.reverse().null_count(), 1_000);
    }

    /// Reversing an array that holds one slot per element still walks it.
    #[test]
    fn a_flat_array_is_written_out_backwards() {
        let name = PlSmallStr::from_static("a");

        let ints = Int32Chunked::new(name.clone(), [Some(1), None, Some(3)]);
        assert_eq!(Vec::from(&ints.reverse()), vec![Some(3), None, Some(1)]);

        let strings = StringChunked::new(name, [Some("a"), None, Some("c")]);
        assert_eq!(
            Vec::from(&strings.reverse()),
            vec![Some("c"), None, Some("a")],
        );
    }
}
