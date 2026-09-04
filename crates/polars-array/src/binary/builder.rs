//! The builder of a [`PlBinaryArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;

use super::PlBinaryArray;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlBinaryArray`].
pub struct PlBinaryArrayBuilder {
    values: Vec<u8>,
    /// The start of every element appended so far, plus the end of the last: one slot more than the
    /// elements, which is what the offsets of a flat binary array hold.
    offsets: Vec<u64>,
    validity: OptBitmapBuilder,
}

impl PlBinaryArrayBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            // The end of the last element of an empty array, which is where the first one starts.
            offsets: vec![0],
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self::new();
        StaticArrayBuilder::reserve(&mut builder, capacity);
        builder
    }

    /// The end of the last element appended, which is where the next one starts.
    #[inline]
    fn last_offset(&self) -> u64 {
        // The offsets are never empty: they start out holding the end of no element at all.
        self.offsets[self.offsets.len() - 1]
    }

    /// Appends one element covering the `length` bytes that were just appended.
    #[inline]
    fn push_offset(&mut self, length: usize) {
        self.offsets.push(self.last_offset() + length as u64);
    }

    /// Appends `value` as an element of its own.
    #[inline]
    pub fn push_value(&mut self, value: &[u8]) {
        self.values.extend_from_slice(value);
        self.push_offset(value.len());
        self.validity.extend_constant(1, true);
    }

    /// Appends a null.
    #[inline]
    pub fn push_null(&mut self) {
        // The value of a null element is undetermined, so the empty byte string it covers reaches
        // no bytes to append: it starts and ends where the last element ended.
        self.push_offset(0);
        self.validity.extend_constant(1, false);
    }

    /// Appends `value`, or a null if it is [`None`].
    #[inline]
    pub fn push(&mut self, value: Option<&[u8]>) {
        match value {
            Some(value) => self.push_value(value),
            None => self.push_null(),
        }
    }

    /// Appends `element` `repeats` times over, one element per copy.
    fn extend_repeated(&mut self, element: &[u8], repeats: usize) {
        self.values.reserve(repeats * element.len());
        self.offsets.reserve(repeats);

        let mut offset = self.last_offset();
        for _ in 0..repeats {
            self.values.extend_from_slice(element);
            offset += element.len() as u64;
            self.offsets.push(offset);
        }
    }

    /// Appends the `length` elements of `other` starting at `start`, ignoring its validity mask.
    fn extend_values(&mut self, other: &PlBinaryArray, start: usize, length: usize) {
        let Some(offsets) = other.flat_offsets() else {
            // The offsets are not flat, so every element covers the same bytes — which are appended
            // once per element. An empty array holds no range for the subslice to cover, but the
            // subslice it admits covers no element either.
            if let Some(element) = other.scalar_values() {
                self.extend_repeated(element, length);
            }
            return;
        };

        let (first, last) = (offsets[start], offsets[start + length]);
        // SAFETY: the offsets are ordered and bounded by the length of the values of `other`.
        let bytes = unsafe { other.values().get_unchecked(first as usize..last as usize) };
        self.values.extend_from_slice(bytes);

        let base = self.last_offset();
        self.offsets
            .extend(offsets[start + 1..=start + length].iter().map(|offset| {
                // The offsets of `other` start at `first`, and the ones here at the end of the last
                // element appended.
                base + (offset - first)
            }));
    }
}

impl Default for PlBinaryArrayBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StaticArrayBuilder for PlBinaryArrayBuilder {
    type Array = PlBinaryArray;

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
        // The bytes are not reserved for: how many of them the elements reach is not implied by how
        // many elements there are.
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn freeze(self) -> PlBinaryArray {
        let length = self.offsets.len() - 1;
        let validity = self.validity.into_opt_validity();
        // SAFETY: the offsets hold the start of every element plus the end of the last, they are
        // pushed in non-decreasing order, and they reach exactly the bytes appended alongside them.
        unsafe {
            PlBinaryArray::new_unchecked(
                Buffer::from(self.values),
                Buffer::from(self.offsets),
                length,
                validity,
            )
        }
    }

    fn freeze_reset(&mut self) -> PlBinaryArray {
        let values = std::mem::take(&mut self.values);
        let offsets = std::mem::replace(&mut self.offsets, vec![0]);
        let validity = std::mem::take(&mut self.validity);
        let length = offsets.len() - 1;
        // SAFETY: as in `freeze`.
        unsafe {
            PlBinaryArray::new_unchecked(
                Buffer::from(values),
                Buffer::from(offsets),
                length,
                validity.into_opt_validity(),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The value of a null element is undetermined, so the empty byte string every one of them
        // covers reaches no bytes to append.
        let offset = self.last_offset();
        self.offsets.extend(std::iter::repeat_n(offset, length));
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &PlBinaryArray,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.offsets.reserve(length);

        self.extend_values(other, start, length);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlBinaryArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.offsets.reserve(length * repeats);

        if other.offsets_are_flat() {
            for i in start..start + length {
                // SAFETY: `i` is in bounds of the array, whose offsets are flat.
                let element = unsafe { other.value_unchecked(i) };
                self.extend_repeated(element, repeats);
            }
        } else {
            // Every element covers the same bytes, so which of them is repeated is immaterial.
            self.extend_values(other, start, length * repeats);
        }

        subslice_extend_each_repeated_validity(
            &mut self.validity,
            other.validity(),
            start,
            length,
            repeats,
        );
    }

    unsafe fn gather_extend(
        &mut self,
        other: &PlBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.offsets.reserve(idxs.len());

        if other.offsets_are_flat() {
            // A run of consecutive indices is a subslice, whose bytes are appended in one go.
            let mut run_start = 0;
            while run_start < idxs.len() {
                let first = idxs[run_start] as usize;
                let mut run_length = 1;
                while run_start + run_length < idxs.len()
                    && idxs[run_start + run_length] as usize == first + run_length
                {
                    run_length += 1;
                }

                self.extend_values(other, first, run_length);
                run_start += run_length;
            }
        } else {
            // Every index reads the one range the array holds.
            self.extend_values(other, 0, idxs.len());
        }

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.offsets.reserve(idxs.len());

        for idx in idxs {
            let idx = *idx as usize;
            if idx < other.len() {
                // SAFETY: the index was just checked against the length of the array.
                let element = unsafe { other.value_unchecked(idx) };
                self.values.extend_from_slice(element);
                self.push_offset(element.len());
            } else {
                // An out-of-bounds index stands for a null, which covers the empty byte string.
                self.push_offset(0);
            }
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;

    /// The three byte strings `foo`, null and `bar`.
    fn array() -> PlBinaryArray {
        PlBinaryArray::from_values_iter([b"foo".as_slice(), b"xxx", b"bar"])
            .with_validity(Some(Bitmap::from_iter([true, false, true])))
    }

    /// The optional elements of a binary array, as their bytes.
    fn elements(array: &PlBinaryArray) -> Vec<Option<Vec<u8>>> {
        array
            .iter()
            .map(|element| element.map(<[u8]>::to_vec))
            .collect()
    }

    #[test]
    fn appending_subslices_and_repeats() {
        let array = array();

        let mut builder = PlBinaryArrayBuilder::with_capacity(8);
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Never);
        builder.extend_nulls(1);
        builder.subslice_extend_repeated(&array, 0, 2, 2, ShareStrategy::Never);
        builder.subslice_extend_each_repeated(&array, 2, 1, 2, ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                None,
                Some(b"bar".to_vec()),
                None,
                Some(b"foo".to_vec()),
                None,
                Some(b"foo".to_vec()),
                None,
                Some(b"bar".to_vec()),
                Some(b"bar".to_vec()),
            ],
        );

        // The bytes of the built array are the ones its elements reach, and no more: the ones a
        // null element would have covered are appended as they were, since it is the mask and not
        // the offsets that makes an element null.
        assert!(built.is_flat());
        assert_eq!(built.values().len(), 24);
    }

    #[test]
    fn gathering() {
        let array = array();

        let mut builder = PlBinaryArrayBuilder::new();
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Never) };
        builder.opt_gather_extend(&array, &[0, 9], ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                Some(b"bar".to_vec()),
                Some(b"foo".to_vec()),
                None,
                Some(b"foo".to_vec()),
                None,
            ],
        );
    }

    #[test]
    fn pushing_elements_one_at_a_time() {
        let mut builder = PlBinaryArrayBuilder::with_capacity(4);
        builder.push_value(b"foo");
        builder.push_null();
        builder.push(Some(b"".as_slice()));
        builder.push(None);

        assert_eq!(builder.len(), 4);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [Some(b"foo".to_vec()), None, Some(Vec::new()), None],
        );
        // A null covers no bytes, so it leaves the values where the last element ended.
        assert_eq!(built.values().len(), 3);
    }

    #[test]
    fn a_scalar_array_is_appended_without_being_materialized() {
        let array = PlBinaryArray::new_scalar(b"ab", 1_000_000_000);

        let mut builder = PlBinaryArrayBuilder::new();
        builder.subslice_extend(&array, 999_999_998, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 0, 1, 2, ShareStrategy::Always);
        unsafe { builder.gather_extend(&array, &[999_999_999], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 1_000_000_000], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 7);
        assert_eq!(built.null_count(), 1);
        // The out-of-bounds index is a null, which covers no bytes.
        assert_eq!(built.values().as_slice(), b"abababababab");
        assert_eq!(built.get(5), Some(b"ab".as_slice()));
        assert_eq!(built.get(6), None);
    }
}
