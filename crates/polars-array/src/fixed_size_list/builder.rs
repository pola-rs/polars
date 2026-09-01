//! The builder of a [`PlFixedSizeListArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_utils::IdxSize;

use super::PlFixedSizeListArray;
use crate::builder::{
    PlArrayBuilder, ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlFixedSizeListArray`].
///
/// A fixed size list array is a values array read a width at a time, so this builder is a width
/// and a length over the builder of those values: appending an element appends the width of values
/// it covers to the child builder, and there is nothing else to hold. The values of the array this
/// builds are therefore laid end to end, one width per element, which makes it
/// [flat](crate::Flat) however many of the appended elements shared a list.
///
/// The child builder is what the values of the built array come out of, so it is what decides
/// their representation and how they are appended; [`ShareStrategy`] is passed straight through to
/// it. A null element covers a width of values like any other, which the child appends as nulls.
///
/// # Example
/// ```
/// use polars_array::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
/// use polars_array::{
///     PlArray, PlFixedSizeListArray, PlFixedSizeListArrayBuilder, PlPrimitiveArray,
/// };
///
/// // Two lists of two values: `[1, 2]` and `[3, 4]`.
/// let array = PlFixedSizeListArray::from_values(
///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
///     2,
/// );
///
/// let mut builder = PlFixedSizeListArrayBuilder::new(builder_like(array.values()), 2);
/// builder.extend_nulls(1);
/// builder.extend(&array, ShareStrategy::Always);
///
/// let built = builder.freeze();
/// assert_eq!(built.len(), 3);
/// assert_eq!(built.width(), 2);
/// assert_eq!(built.null_count(), 1);
/// // Every element covers a width of values, including the null one.
/// assert_eq!(built.values().len(), 6);
/// ```
pub struct PlFixedSizeListArrayBuilder<B: PlArrayBuilder = Box<dyn PlArrayBuilder>> {
    values: B,
    width: usize,
    length: usize,
    validity: OptBitmapBuilder,
}

impl<B: PlArrayBuilder> PlFixedSizeListArrayBuilder<B> {
    /// Creates an empty builder of the lists of `width` values the child builder builds.
    pub fn new(values: B, width: usize) -> Self {
        Self {
            values,
            width,
            length: 0,
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(values: B, width: usize, capacity: usize) -> Self {
        let mut builder = Self::new(values, width);
        StaticArrayBuilder::reserve(&mut builder, capacity);
        builder
    }

    /// The builder of the values the lists are taken over.
    #[inline]
    pub fn values(&self) -> &B {
        &self.values
    }

    /// The number of values every element of the built array covers.
    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Panics unless `other` is as wide as the lists this builds.
    fn assert_width(&self, other: &PlFixedSizeListArray) {
        assert_eq!(
            other.width(),
            self.width,
            "cannot append a fixed size list array of width {} to a builder of width {}",
            other.width(),
            self.width,
        );
    }

    /// Appends the `length` elements of `other` starting at `start`, ignoring its validity mask.
    fn extend_values(
        &mut self,
        other: &PlFixedSizeListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        if other.values_are_scalar() {
            // Every element covers the one list the values hold, which is appended once per
            // element.
            self.values
                .subslice_extend_repeated(other.values(), 0, self.width, length, share);
        } else {
            self.values.subslice_extend(
                other.values(),
                start * self.width,
                length * self.width,
                share,
            );
        }
    }
}

impl<B: PlArrayBuilder> StaticArrayBuilder for PlFixedSizeListArrayBuilder<B> {
    type Array = PlFixedSizeListArray;

    fn reserve(&mut self, additional: usize) {
        // Unlike a list array, the values an element covers are as many as the width, which is
        // what the child can be reserved for.
        self.values.reserve(additional * self.width);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    fn freeze(self) -> PlFixedSizeListArray {
        let (width, length) = (self.width, self.length);
        let validity = self.validity.into_opt_validity();
        // SAFETY: every element appended the width of values it covers to the child, so the values
        // hold the width of every element laid end to end, and the mask holds one bit per element.
        unsafe {
            PlFixedSizeListArray::new_unchecked(self.values.freeze(), width, length, validity)
        }
    }

    fn freeze_reset(&mut self) -> PlFixedSizeListArray {
        let validity = std::mem::take(&mut self.validity);
        let length = std::mem::take(&mut self.length);
        // SAFETY: as in `freeze`.
        unsafe {
            PlFixedSizeListArray::new_unchecked(
                self.values.freeze_reset(),
                self.width,
                length,
                validity.into_opt_validity(),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The values of a null element are undetermined, but there are as many of them as there
        // are of any other element: the width is what the elements are read a slot at a time.
        self.values.extend_nulls(length * self.width);
        self.validity.extend_constant(length, false);
        self.length += length;
    }

    fn subslice_extend(
        &mut self,
        other: &PlFixedSizeListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        assert_subslice(other.len(), start, length);

        self.extend_values(other, start, length, share);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
        self.length += length;
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlFixedSizeListArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        assert_subslice(other.len(), start, length);
        self.values.reserve(length * repeats * self.width);

        if other.values_are_scalar() {
            // Every element covers the same list, so which of them is repeated is immaterial.
            self.extend_values(other, start, length * repeats, share);
        } else {
            for i in start..start + length {
                self.values.subslice_extend_repeated(
                    other.values(),
                    i * self.width,
                    self.width,
                    repeats,
                    share,
                );
            }
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
        other: &PlFixedSizeListArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        self.values.reserve(idxs.len() * self.width);

        if other.values_are_scalar() {
            // Every index reads the one list the values hold.
            self.extend_values(other, 0, idxs.len(), share);
        } else {
            // A run of consecutive indices is a subslice, which the child appends in one go.
            let mut run_start = 0;
            while run_start < idxs.len() {
                let first = idxs[run_start] as usize;
                let mut run_length = 1;
                while run_start + run_length < idxs.len()
                    && idxs[run_start + run_length] as usize == first + run_length
                {
                    run_length += 1;
                }

                self.extend_values(other, first, run_length, share);
                run_start += run_length;
            }
        }

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
        self.length += idxs.len();
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlFixedSizeListArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        self.values.reserve(idxs.len() * self.width);

        for idx in idxs {
            let idx = *idx as usize;
            if idx < other.len() {
                // SAFETY: the index was just checked against the length of the array.
                let range = unsafe { other.value_range_unchecked(idx) };
                self.values
                    .subslice_extend(other.values(), range.start, self.width, share);
            } else {
                // An out-of-bounds index stands for a null, which covers a width of nulls.
                self.values.extend_nulls(self.width);
            }
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
        self.length += idxs.len();
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::PlFixedSizeListArrayBuilder;
    use crate::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
    use crate::{PlFixedSizeListArray, PlPrimitiveArray};

    /// Three lists of two values, of which the middle one is null.
    fn array() -> PlFixedSizeListArray {
        PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6])),
            2,
        )
        .with_validity(Some(Bitmap::from_iter([true, false, true])))
    }

    /// The elements of a fixed size list array, as the values of the lists they cover.
    fn elements(array: &PlFixedSizeListArray) -> Vec<Option<Vec<i32>>> {
        array
            .iter()
            .map(|element| {
                element.map(|element| {
                    element
                        .as_any()
                        .downcast_ref::<PlPrimitiveArray<i32>>()
                        .unwrap()
                        .values_iter()
                        .collect()
                })
            })
            .collect()
    }

    fn builder() -> PlFixedSizeListArrayBuilder {
        PlFixedSizeListArrayBuilder::with_capacity(builder_like(array().values()), 2, 8)
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
        assert_eq!(built.values().len(), 18);
    }

    #[test]
    fn gathering() {
        let array = array();

        let mut builder = builder();
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 9], ShareStrategy::Always);

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
        let array = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1_000_000_000,
        );

        let mut builder = PlFixedSizeListArrayBuilder::new(builder_like(array.values()), 2);
        builder.subslice_extend(&array, 999_999_998, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 0, 1, 2, ShareStrategy::Always);
        unsafe { builder.gather_extend(&array, &[999_999_999], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 1_000_000_000], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                None,
            ],
        );
        assert_eq!(built.null_count(), 1);
    }

    #[test]
    fn a_fully_null_array_appends_a_width_of_nulls_per_element() {
        let array = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::<i32>::new_scalar(1, 2)),
            1_000_000_000,
        );

        let mut builder = PlFixedSizeListArrayBuilder::new(builder_like(array.values()), 2);
        builder.subslice_extend(&array, 0, 3, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 3);
        assert_eq!(built.null_count(), 3);
        assert_eq!(built.values().len(), 6);
    }

    #[test]
    fn a_width_of_zero_holds_no_values() {
        // Three elements of no values at all, which is all a width of zero leaves of them.
        let array =
            PlFixedSizeListArray::new(Box::new(PlPrimitiveArray::<i32>::new_empty()), 0, 3, None);

        let mut builder = PlFixedSizeListArrayBuilder::new(builder_like(array.values()), 0);
        builder.extend_nulls(2);
        builder.extend(&array, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 5);
        assert_eq!(built.width(), 0);
        assert_eq!(built.null_count(), 2);
        assert!(built.values().is_empty());
    }

    #[test]
    #[should_panic(
        expected = "cannot append a fixed size list array of width 3 to a builder of \
                              width 2"
    )]
    fn appending_another_width_panics() {
        let mut builder = builder();
        builder.extend(
            &PlFixedSizeListArray::from_values(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                3,
            ),
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
        assert!(builder.values().is_empty());
        builder.extend_nulls(1);
        assert_eq!(builder.freeze().len(), 1);
    }
}
