//! The builder of a [`PlFixedSizeBinaryArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;

use super::PlFixedSizeBinaryArray;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlFixedSizeBinaryArray`].
pub struct PlFixedSizeBinaryArrayBuilder {
    values: Vec<u8>,
    width: usize,
    length: usize,
    validity: OptBitmapBuilder,
}

impl PlFixedSizeBinaryArrayBuilder {
    /// Creates an empty builder of the byte strings of `width` bytes.
    pub fn new(width: usize) -> Self {
        Self {
            values: Vec::new(),
            width,
            length: 0,
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(width: usize, capacity: usize) -> Self {
        let mut builder = Self::new(width);
        StaticArrayBuilder::reserve(&mut builder, capacity);
        builder
    }

    /// The number of bytes every element of the built array covers.
    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Panics unless `other` is as wide as the elements this builds.
    fn assert_width(&self, other: &PlFixedSizeBinaryArray) {
        assert_eq!(
            other.width(),
            self.width,
            "cannot append a fixed size binary array of width {} to a builder of width {}",
            other.width(),
            self.width,
        );
    }

    /// Appends the bytes of the `length` elements of `other` starting at `start`, ignoring its
    /// validity mask.
    ///
    /// Scalar values are not materialized to be read: the one element they hold is appended once
    /// per element the subslice covers.
    fn extend_values(&mut self, other: &PlFixedSizeBinaryArray, start: usize, length: usize) {
        if let Some(values) = other.flat_values() {
            let bytes = &values[start * self.width..(start + length) * self.width];
            self.values.extend_from_slice(bytes);
        } else if let Some(element) = other.scalar_values() {
            self.extend_repeated(element, length);
        }
        // An empty array is neither, and the subslice it admits covers no element to append.
    }

    /// Appends `element` `repeats` times over.
    fn extend_repeated(&mut self, element: &[u8], repeats: usize) {
        debug_assert_eq!(element.len(), self.width);

        self.values.reserve(repeats * self.width);
        for _ in 0..repeats {
            self.values.extend_from_slice(element);
        }
    }
}

impl StaticArrayBuilder for PlFixedSizeBinaryArrayBuilder {
    type Array = PlFixedSizeBinaryArray;

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional * self.width);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    fn freeze(self) -> PlFixedSizeBinaryArray {
        let (width, length) = (self.width, self.length);
        let validity = self.validity.into_opt_validity();
        // SAFETY: every element appended the width of bytes it covers, so the values hold the
        // width of every element laid end to end, and the mask holds one bit per element.
        unsafe {
            PlFixedSizeBinaryArray::new_unchecked(
                Buffer::from(self.values),
                width,
                length,
                validity,
            )
        }
    }

    fn freeze_reset(&mut self) -> PlFixedSizeBinaryArray {
        let values = std::mem::take(&mut self.values);
        let validity = std::mem::take(&mut self.validity);
        let length = std::mem::take(&mut self.length);
        // SAFETY: as in `freeze`.
        unsafe {
            PlFixedSizeBinaryArray::new_unchecked(
                Buffer::from(values),
                self.width,
                length,
                validity.into_opt_validity(),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The bytes of a null element are undetermined, but there are as many of them as there are
        // of any other element: the width is what every element covers.
        self.values
            .resize(self.values.len() + length * self.width, 0);
        self.validity.extend_constant(length, false);
        self.length += length;
    }

    fn subslice_extend(
        &mut self,
        other: &PlFixedSizeBinaryArray,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        self.assert_width(other);
        assert_subslice(other.len(), start, length);

        self.extend_values(other, start, length);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
        self.length += length;
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlFixedSizeBinaryArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        self.assert_width(other);
        assert_subslice(other.len(), start, length);
        self.values.reserve(length * repeats * self.width);

        if let Some(values) = other.flat_values() {
            for i in start..start + length {
                // SAFETY: the subslice was just checked against the length of the array.
                let range = unsafe { other.value_range_unchecked(i) };
                self.extend_repeated(&values[range], repeats);
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
        self.length += length * repeats;
    }

    unsafe fn gather_extend(
        &mut self,
        other: &PlFixedSizeBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.assert_width(other);
        self.values.reserve(idxs.len() * self.width);

        if let Some(values) = other.flat_values() {
            for idx in idxs {
                // SAFETY: the indices are in bounds of the array, whose values are flat.
                let range = unsafe { other.value_range_unchecked(*idx as usize) };
                self.values.extend_from_slice(&values[range]);
            }
        } else {
            // Every index reads the one element the values hold.
            self.extend_values(other, 0, idxs.len());
        }

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
        self.length += idxs.len();
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlFixedSizeBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.assert_width(other);
        self.values.reserve(idxs.len() * self.width);

        for idx in idxs {
            let idx = *idx as usize;
            if idx < other.len() {
                // SAFETY: the index was just checked against the length of the array.
                self.values
                    .extend_from_slice(unsafe { other.value_unchecked(idx) });
            } else {
                // An out-of-bounds index stands for a null, which covers a width of zeros.
                self.values.resize(self.values.len() + self.width, 0);
            }
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
        self.length += idxs.len();
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;

    /// Three elements of two bytes, of which the middle one is null.
    fn array() -> PlFixedSizeBinaryArray {
        PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4, 5, 6], 2)
            .with_validity(Some(Bitmap::from_iter([true, false, true])))
    }

    /// The optional elements of a fixed size binary array, as their bytes.
    fn elements(array: &PlFixedSizeBinaryArray) -> Vec<Option<Vec<u8>>> {
        array
            .iter()
            .map(|element| element.map(<[u8]>::to_vec))
            .collect()
    }

    #[test]
    fn appending_subslices_and_repeats() {
        let array = array();

        let mut builder = PlFixedSizeBinaryArrayBuilder::with_capacity(2, 8);
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Never);
        builder.extend_nulls(1);
        builder.subslice_extend_repeated(&array, 0, 2, 2, ShareStrategy::Never);
        builder.subslice_extend_each_repeated(&array, 2, 1, 2, ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                None,
                Some(vec![5, 6]),
                None,
                Some(vec![1, 2]),
                None,
                Some(vec![1, 2]),
                None,
                Some(vec![5, 6]),
                Some(vec![5, 6]),
            ],
        );

        // Every element covers the width, whether or not it is null.
        assert_eq!(built.flat_values().unwrap().len(), 18);
        assert!(built.is_flat());
    }

    #[test]
    fn gathering() {
        let array = array();

        let mut builder = PlFixedSizeBinaryArrayBuilder::new(2);
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Never) };
        builder.opt_gather_extend(&array, &[0, 9], ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                Some(vec![5, 6]),
                Some(vec![1, 2]),
                None,
                Some(vec![1, 2]),
                None,
            ],
        );
    }

    #[test]
    fn a_scalar_array_is_appended_without_being_materialized() {
        let array = PlFixedSizeBinaryArray::new_scalar(b"ab", 1_000_000_000);

        let mut builder = PlFixedSizeBinaryArrayBuilder::new(2);
        builder.subslice_extend(&array, 999_999_998, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 0, 1, 2, ShareStrategy::Always);
        unsafe { builder.gather_extend(&array, &[999_999_999], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 1_000_000_000], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 7);
        assert_eq!(built.null_count(), 1);
        // The out-of-bounds index is a null, whose bytes are written out as zeros.
        assert_eq!(built.flat_values().unwrap().as_slice(), b"abababababab\0\0");
        assert_eq!(built.get(5), Some(b"ab".as_slice()));
        assert_eq!(built.get(6), None);
    }
}
