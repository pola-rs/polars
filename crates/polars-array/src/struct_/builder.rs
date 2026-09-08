//! The builder of a [`PlStructArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_utils::IdxSize;

use super::PlStructArray;
use crate::array::PlArray;
use crate::bitmap::PlBitmap;
use crate::builder::{
    PlArrayBuilder, ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlStructArray`].
pub struct PlStructArrayBuilder {
    fields: Vec<Box<dyn PlArrayBuilder>>,
    length: usize,
    validity: OptBitmapBuilder,
}

impl PlStructArrayBuilder {
    /// Creates an empty builder over one builder per field of the built array.
    pub fn new(fields: Vec<Box<dyn PlArrayBuilder>>) -> Self {
        Self {
            fields,
            length: 0,
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(fields: Vec<Box<dyn PlArrayBuilder>>, capacity: usize) -> Self {
        let mut builder = Self::new(fields);
        StaticArrayBuilder::reserve(&mut builder, capacity);
        builder
    }

    /// The builders of the fields of the built array.
    #[inline]
    pub fn fields(&self) -> &[Box<dyn PlArrayBuilder>] {
        &self.fields
    }

    /// The number of fields the built array has.
    #[inline]
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// The builders of the fields of the built array, appended to directly.
    #[inline]
    pub fn fields_mut(&mut self) -> &mut [Box<dyn PlArrayBuilder>] {
        &mut self.fields
    }

    /// Closes one element, covering the element appended to every field since the last one was.
    #[inline]
    pub fn finish_row(&mut self) {
        for (i, field) in self.fields.iter().enumerate() {
            assert_eq!(
                field.len(),
                self.length + 1,
                "every field of a struct builder holds one element per element of the array, \
                 but field {} holds {} for the {} elements closed so far",
                i,
                field.len(),
                self.length,
            );
        }
        self.length += 1;
        self.validity.extend_constant(1, true);
    }

    /// The builders of the fields, paired with the fields of `other` they append.
    fn zip_fields<'a>(
        &'a mut self,
        other: &'a PlStructArray,
    ) -> impl Iterator<Item = (&'a mut Box<dyn PlArrayBuilder>, &'a dyn PlArray)> {
        assert_eq!(
            other.num_fields(),
            self.fields.len(),
            "cannot append a struct array of {} fields to a builder of {} fields",
            other.num_fields(),
            self.fields.len(),
        );

        self.fields
            .iter_mut()
            .zip(other.fields().iter().map(|field| &**field))
    }
}

impl StaticArrayBuilder for PlStructArrayBuilder {
    type Array = PlStructArray;

    fn reserve(&mut self, additional: usize) {
        for field in &mut self.fields {
            field.reserve(additional);
        }
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    fn freeze(self) -> PlStructArray {
        let length = self.length;
        let validity = self.validity.into_opt_validity().map(PlBitmap::from_bitmap);
        let fields = self
            .fields
            .into_iter()
            .map(PlArrayBuilder::freeze)
            .collect();
        // SAFETY: every element appended one element to every field, so each of them holds exactly
        // as many elements as this builder, and the mask holds one bit per element.
        unsafe { PlStructArray::new_unchecked(fields, length, validity) }
    }

    fn freeze_reset(&mut self) -> PlStructArray {
        let validity = std::mem::take(&mut self.validity);
        let length = std::mem::take(&mut self.length);
        let fields = self
            .fields
            .iter_mut()
            .map(|field| field.freeze_reset())
            .collect();
        // SAFETY: as in `freeze`.
        unsafe {
            PlStructArray::new_unchecked(
                fields,
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The value of a null element is undetermined, but its fields hold one like any other
        // element's: the fields of a struct array hold one element per element of the array.
        for field in &mut self.fields {
            field.extend_nulls(length);
        }
        self.validity.extend_constant(length, false);
        self.length += length;
    }

    fn subslice_extend(
        &mut self,
        other: &PlStructArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        for (builder, field) in self.zip_fields(other) {
            builder.subslice_extend(field, start, length, share);
        }
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
        self.length += length;
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlStructArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        for (builder, field) in self.zip_fields(other) {
            builder.subslice_extend_each_repeated(field, start, length, repeats, share);
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
        other: &PlStructArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        for (builder, field) in self.zip_fields(other) {
            // SAFETY: the indices are in bounds of the array, and therefore of every field, which
            // holds one element per element of it.
            unsafe { builder.gather_extend(field, idxs, share) };
        }
        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
        self.length += idxs.len();
    }

    fn opt_gather_extend(&mut self, other: &PlStructArray, idxs: &[IdxSize], share: ShareStrategy) {
        for (builder, field) in self.zip_fields(other) {
            builder.opt_gather_extend(field, idxs, share);
        }
        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
        self.length += idxs.len();
    }
}
