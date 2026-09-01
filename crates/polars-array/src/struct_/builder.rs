//! The builder of a [`PlStructArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_utils::IdxSize;

use super::PlStructArray;
use crate::array::PlArray;
use crate::builder::{
    PlArrayBuilder, ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlStructArray`].
///
/// A struct array holds no values of its own, only a validity mask over its field arrays, so this
/// builder is a mask and a length over one builder per field: appending an element appends that
/// element of every field to the builder of that field. [`ShareStrategy`] is passed straight
/// through to them, and so is a null element, whose fields hold a value like any other element's —
/// undetermined, but there.
///
/// # Example
/// ```
/// use polars_array::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
/// use polars_array::{PlPrimitiveArray, PlStructArray, PlStructArrayBuilder};
///
/// let array = PlStructArray::from_fields(vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2]))]);
///
/// let fields = array.fields().iter().map(|field| builder_like(&**field)).collect();
/// let mut builder = PlStructArrayBuilder::new(fields);
/// builder.extend_nulls(1);
/// builder.extend(&array, ShareStrategy::Always);
///
/// let built = builder.freeze();
/// assert_eq!(built.len(), 3);
/// assert_eq!(built.null_count(), 1);
/// assert_eq!(built.field(0).len(), 3);
/// ```
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

    /// The builders of the fields, paired with the fields of `other` they append.
    ///
    /// # Panics
    /// Panics unless `other` has one field per builder.
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
        let validity = self.validity.into_opt_validity();
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
        unsafe { PlStructArray::new_unchecked(fields, length, validity.into_opt_validity()) }
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

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::PlStructArrayBuilder;
    use crate::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
    use crate::{PlBooleanArray, PlPrimitiveArray, PlStructArray};

    /// Three rows of two fields, of which the middle one is null.
    fn array() -> PlStructArray {
        PlStructArray::new(
            vec![
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                Box::new(PlBooleanArray::new_scalar(true, 3)),
            ],
            3,
            Some(Bitmap::from_iter([true, false, true])),
        )
    }

    /// The elements of a struct array, as whether they are null and what their first field holds.
    fn elements(array: &PlStructArray) -> Vec<Option<i32>> {
        let field = array
            .field(0)
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap();

        (0..array.len())
            .map(|i| array.is_valid(i).then(|| field.value(i)))
            .collect()
    }

    fn builder() -> PlStructArrayBuilder {
        let fields = array()
            .fields()
            .iter()
            .map(|field| builder_like(&**field))
            .collect();
        PlStructArrayBuilder::with_capacity(fields, 8)
    }

    #[test]
    fn appending_subslices_and_repeats() {
        let array = array();

        let mut builder = builder();
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Always);
        builder.extend_nulls(1);
        builder.subslice_extend_repeated(&array, 0, 2, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 2, 1, 2, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                None,
                Some(3),
                None,
                Some(1),
                None,
                Some(1),
                None,
                Some(3),
                Some(3),
            ],
        );

        // Every field holds one element per element of the array.
        assert_eq!(built.field(0).len(), 9);
        assert_eq!(built.field(1).len(), 9);
    }

    #[test]
    fn gathering() {
        let array = array();

        let mut builder = builder();
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 9], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(elements(&built), [Some(3), Some(1), None, Some(1), None]);
        assert_eq!(built.field(1).len(), 5);
    }

    #[test]
    fn a_fully_null_array_appends_its_fields_too() {
        let array = PlStructArray::new_full_null(
            vec![Box::new(PlPrimitiveArray::<i32>::new_scalar(
                1,
                1_000_000_000,
            ))],
            1_000_000_000,
        );

        let fields = array
            .fields()
            .iter()
            .map(|field| builder_like(&**field))
            .collect();
        let mut builder = PlStructArrayBuilder::new(fields);
        builder.subslice_extend(&array, 0, 3, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 3);
        assert_eq!(built.null_count(), 3);
        assert_eq!(built.field(0).len(), 3);
    }

    #[test]
    fn a_struct_array_of_no_fields_is_a_length_and_a_mask() {
        let array = PlStructArray::new(vec![], 3, Some(Bitmap::from_iter([true, false, true])));

        let mut builder = PlStructArrayBuilder::new(vec![]);
        assert_eq!(builder.num_fields(), 0);
        builder.extend(&array, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 3);
        assert_eq!(built.null_count(), 1);
        assert!(built.fields().is_empty());
    }

    #[test]
    #[should_panic(expected = "cannot append a struct array of 1 fields to a builder of 2 fields")]
    fn appending_another_number_of_fields_panics() {
        let mut builder = builder();
        builder.extend(
            &PlStructArray::from_fields(vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32]))]),
            ShareStrategy::Always,
        );
    }

    #[test]
    fn freeze_reset_leaves_an_empty_builder() {
        let array = array();

        let mut builder = builder();
        builder.extend(&array, ShareStrategy::Always);
        assert_eq!(builder.freeze_reset().len(), 3);

        assert!(builder.is_empty());
        assert!(builder.fields()[0].is_empty());
        builder.extend_nulls(1);
        assert_eq!(builder.freeze().len(), 1);
    }
}
