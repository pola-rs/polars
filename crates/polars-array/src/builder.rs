//! Building arrays element by element, or array by array.

/// Whether a builder may adopt the buffers of the arrays it appends, rather than copying out of
/// them.
pub use arrow::array::builder::ShareStrategy;
use arrow::bitmap::OptBitmapBuilder;
use polars_utils::IdxSize;

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::static_array::StaticArray;
use crate::{
    PlBinaryArray, PlBinaryArrayBuilder, PlBinaryViewArray, PlBinaryViewArrayBuilder,
    PlBooleanArray, PlBooleanArrayBuilder, PlFixedSizeBinaryArray, PlFixedSizeBinaryArrayBuilder,
    PlFixedSizeListArray, PlFixedSizeListArrayBuilder, PlListArray, PlListArrayBuilder,
    PlNullArray, PlNullArrayBuilder, PlPrimitiveArray, PlPrimitiveArrayBuilder, PlStructArray,
    PlStructArrayBuilder, PlUtf8ViewArray, PlUtf8ViewArrayBuilder,
    with_match_pl_primitive_array_type,
};

/// A builder of one concrete array type.
pub trait StaticArrayBuilder: Send {
    /// The array this builder builds.
    type Array: StaticArray;

    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// The number of elements appended so far.
    fn len(&self) -> usize;

    /// Whether no element has been appended yet.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consumes this builder, returning the array it built.
    fn freeze(self) -> Self::Array;

    /// Returns the array this builder built, leaving it empty and ready to build another.
    fn freeze_reset(&mut self) -> Self::Array;

    /// Appends `length` nulls.
    fn extend_nulls(&mut self, length: usize);

    /// Appends every element of `other`, in order.
    #[inline]
    fn extend(&mut self, other: &Self::Array, share: ShareStrategy) {
        self.subslice_extend(other, 0, other.len(), share);
    }

    /// Appends the `length` elements of `other` starting at `start`, in order.
    ///
    /// # Panics
    /// Panics if `start + length > other.len()`.
    fn subslice_extend(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    );

    /// Appends the `length` elements of `other` starting at `start` `repeats` times over.
    ///
    /// # Panics
    /// Panics if `start + length > other.len()`.
    fn subslice_extend_repeated(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.reserve(length * repeats);
        for _ in 0..repeats {
            self.subslice_extend(other, start, length, share);
        }
    }

    /// Appends each of the `length` elements of `other` starting at `start` `repeats` times over.
    ///
    /// # Panics
    /// Panics if `start + length > other.len()`.
    fn subslice_extend_each_repeated(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Appends the element of `other` at every index of `idxs`, in the order they are given.
    ///
    /// # Safety
    /// Every index must be smaller than `other.len()`.
    unsafe fn gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy);

    /// Appends the element of `other` at every index of `idxs`, in the order they are given, with
    /// an out-of-bounds index standing for a null.
    fn opt_gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy);
}

/// A trait object over the builders in this crate.
#[allow(private_bounds)]
pub trait PlArrayBuilder: PlArrayBuilderBoxedHelper + Send {
    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// The number of elements appended so far.
    fn len(&self) -> usize;

    /// Whether no element has been appended yet.
    fn is_empty(&self) -> bool;

    /// Consumes this builder, returning the array it built.
    fn freeze(self) -> Box<dyn PlArray>;

    /// Returns the array this builder built, leaving it empty and ready to build another.
    fn freeze_reset(&mut self) -> Box<dyn PlArray>;

    /// Appends `length` nulls.
    fn extend_nulls(&mut self, length: usize);

    /// Appends every element of `other`, in order.
    ///
    /// # Panics
    /// Panics if `other` is not of the type this builder builds.
    fn extend(&mut self, other: &dyn PlArray, share: ShareStrategy);

    /// Appends the `length` elements of `other` starting at `start`, in order.
    ///
    /// # Panics
    /// Panics if `other` is not of the type this builder builds, or if `start + length >
    /// other.len()`.
    fn subslice_extend(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    );

    /// Appends the `length` elements of `other` starting at `start` `repeats` times over.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::subslice_extend`] panics.
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Appends each of the `length` elements of `other` starting at `start` `repeats` times over.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::subslice_extend`] panics.
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Appends the element of `other` at every index of `idxs`, in the order they are given.
    ///
    /// # Panics
    /// Panics if `other` is not of the type this builder builds.
    ///
    /// # Safety
    /// Every index must be smaller than `other.len()`.
    unsafe fn gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy);

    /// Appends the element of `other` at every index of `idxs`, in the order they are given, with
    /// an out-of-bounds index standing for a null.
    ///
    /// # Panics
    /// Panics if `other` is not of the type this builder builds.
    fn opt_gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy);
}

/// The [`PlArrayBuilder::freeze`] of a builder that is already in a box, which is the one form of
/// it a trait object admits.
trait PlArrayBuilderBoxedHelper {
    fn freeze_boxed(self: Box<Self>) -> Box<dyn PlArray>;
}

impl<B: PlArrayBuilder> PlArrayBuilderBoxedHelper for B {
    #[inline]
    fn freeze_boxed(self: Box<Self>) -> Box<dyn PlArray> {
        (*self).freeze()
    }
}

/// Downcasts `array` to the array `B` builds.
///
/// # Panics
/// Panics if `array` is not of that type.
#[inline]
fn downcast<B: StaticArrayBuilder>(array: &dyn PlArray) -> &B::Array {
    array
        .as_any()
        .downcast_ref::<B::Array>()
        .unwrap_or_else(|| {
            panic!(
                "cannot append a {:?} array to a builder of a different array type",
                array.array_type(),
            )
        })
}

impl<B: StaticArrayBuilder> PlArrayBuilder for B {
    #[inline(always)]
    fn reserve(&mut self, additional: usize) {
        StaticArrayBuilder::reserve(self, additional);
    }

    #[inline(always)]
    fn len(&self) -> usize {
        StaticArrayBuilder::len(self)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        StaticArrayBuilder::is_empty(self)
    }

    #[inline(always)]
    fn freeze(self) -> Box<dyn PlArray> {
        Box::new(StaticArrayBuilder::freeze(self))
    }

    #[inline(always)]
    fn freeze_reset(&mut self) -> Box<dyn PlArray> {
        Box::new(StaticArrayBuilder::freeze_reset(self))
    }

    #[inline(always)]
    fn extend_nulls(&mut self, length: usize) {
        StaticArrayBuilder::extend_nulls(self, length);
    }

    #[inline(always)]
    fn extend(&mut self, other: &dyn PlArray, share: ShareStrategy) {
        StaticArrayBuilder::extend(self, downcast::<Self>(other), share);
    }

    #[inline(always)]
    fn subslice_extend(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        StaticArrayBuilder::subslice_extend(self, downcast::<Self>(other), start, length, share);
    }

    #[inline(always)]
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        StaticArrayBuilder::subslice_extend_repeated(
            self,
            downcast::<Self>(other),
            start,
            length,
            repeats,
            share,
        );
    }

    #[inline(always)]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        StaticArrayBuilder::subslice_extend_each_repeated(
            self,
            downcast::<Self>(other),
            start,
            length,
            repeats,
            share,
        );
    }

    #[inline(always)]
    unsafe fn gather_extend(
        &mut self,
        other: &dyn PlArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        // SAFETY: the indices are in bounds of the array, which is the one downcast here.
        unsafe { StaticArrayBuilder::gather_extend(self, downcast::<Self>(other), idxs, share) };
    }

    #[inline(always)]
    fn opt_gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy) {
        StaticArrayBuilder::opt_gather_extend(self, downcast::<Self>(other), idxs, share);
    }
}

/// A boxed builder is a builder: this is what lets the nested builders be built out of one.
impl PlArrayBuilder for Box<dyn PlArrayBuilder> {
    #[inline(always)]
    fn reserve(&mut self, additional: usize) {
        (**self).reserve(additional);
    }

    #[inline(always)]
    fn len(&self) -> usize {
        (**self).len()
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }

    #[inline(always)]
    fn freeze(self) -> Box<dyn PlArray> {
        self.freeze_boxed()
    }

    #[inline(always)]
    fn freeze_reset(&mut self) -> Box<dyn PlArray> {
        (**self).freeze_reset()
    }

    #[inline(always)]
    fn extend_nulls(&mut self, length: usize) {
        (**self).extend_nulls(length);
    }

    #[inline(always)]
    fn extend(&mut self, other: &dyn PlArray, share: ShareStrategy) {
        (**self).extend(other, share);
    }

    #[inline(always)]
    fn subslice_extend(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend(other, start, length, share);
    }

    #[inline(always)]
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend_repeated(other, start, length, repeats, share);
    }

    #[inline(always)]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend_each_repeated(other, start, length, repeats, share);
    }

    #[inline(always)]
    unsafe fn gather_extend(
        &mut self,
        other: &dyn PlArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        // SAFETY: the indices are in bounds of `other`, which this only passes on.
        unsafe { (**self).gather_extend(other, idxs, share) };
    }

    #[inline(always)]
    fn opt_gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy) {
        (**self).opt_gather_extend(other, idxs, share);
    }
}

/// An empty builder of the arrays that `array` is one of.
pub fn builder_like(array: &dyn PlArray) -> Box<dyn PlArrayBuilder> {
    match array.array_type() {
        PlArrayType::Primitive(_) => with_match_pl_primitive_array_type!(array, |T| {
            Box::new(PlPrimitiveArrayBuilder::<T>::new()) as Box<dyn PlArrayBuilder>
        })
        .expect("a primitive array has a primitive element type"),
        PlArrayType::Boolean => Box::new(PlBooleanArrayBuilder::new()),
        PlArrayType::Binary => Box::new(PlBinaryArrayBuilder::new()),
        PlArrayType::BinaryView => Box::new(PlBinaryViewArrayBuilder::new()),
        PlArrayType::Utf8View => Box::new(PlUtf8ViewArrayBuilder::new()),
        PlArrayType::FixedSizeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<PlFixedSizeBinaryArray>()
                .unwrap();
            Box::new(PlFixedSizeBinaryArrayBuilder::new(array.width()))
        },
        PlArrayType::Null => Box::new(PlNullArrayBuilder::new()),
        PlArrayType::List => {
            let array = array.as_any().downcast_ref::<PlListArray>().unwrap();
            Box::new(PlListArrayBuilder::new(builder_like(array.values())))
        },
        PlArrayType::FixedSizeList => {
            let array = array
                .as_any()
                .downcast_ref::<PlFixedSizeListArray>()
                .unwrap();
            // The values hold the values the elements are made of in either representation,
            // which is all a builder for them is taken from.
            Box::new(PlFixedSizeListArrayBuilder::new(
                builder_like(array.values()),
                array.width(),
            ))
        },
        PlArrayType::Struct => {
            let array = array.as_any().downcast_ref::<PlStructArray>().unwrap();
            let fields = array
                .fields()
                .iter()
                .map(|field| builder_like(&**field))
                .collect();
            Box::new(PlStructArrayBuilder::new(fields))
        },
        x @ PlArrayType::Object { .. } => {
            panic!("polars-array: no PlArrayBuilder for {x:?}")
        },
    }
}

/// An array of `length` nulls of the type that `array` is one of.
///
/// # Panics
/// Panics for an object array, which has no builder in this crate either.
pub fn full_null_like(array: &dyn PlArray, length: usize) -> Box<dyn PlArray> {
    match array.array_type() {
        PlArrayType::Null => Box::new(PlNullArray::new(length)),
        PlArrayType::Boolean => Box::new(PlBooleanArray::new_full_null(length)),
        PlArrayType::Primitive(_) => with_match_pl_primitive_array_type!(array, |T| {
            Box::new(PlPrimitiveArray::<T>::new_full_null(length)) as Box<dyn PlArray>
        })
        .expect("a primitive array has a primitive element type"),
        PlArrayType::Binary => Box::new(PlBinaryArray::new_full_null(length)),
        PlArrayType::BinaryView => Box::new(PlBinaryViewArray::new_full_null(length)),
        PlArrayType::Utf8View => Box::new(PlUtf8ViewArray::new_full_null(length)),
        PlArrayType::FixedSizeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<PlFixedSizeBinaryArray>()
                .unwrap();
            Box::new(PlFixedSizeBinaryArray::new_full_null(array.width(), length))
        },
        PlArrayType::List => {
            let array = array.as_any().downcast_ref::<PlListArray>().unwrap();
            // Every element is an empty list, so the values are only there to carry their shape.
            Box::new(PlListArray::new_full_null(
                array.values().sliced(0, 0),
                length,
            ))
        },
        PlArrayType::FixedSizeList => {
            let array = array
                .as_any()
                .downcast_ref::<PlFixedSizeListArray>()
                .unwrap();
            // An element of a null list is as wide as any other, so the one element the values
            // stand for is as many nulls as the array is wide.
            Box::new(PlFixedSizeListArray::new_full_null(
                full_null_like(array.values(), array.width()),
                length,
            ))
        },
        PlArrayType::Struct => {
            let array = array.as_any().downcast_ref::<PlStructArray>().unwrap();
            let fields = array
                .fields()
                .iter()
                .map(|field| full_null_like(&**field, length))
                .collect();
            Box::new(PlStructArray::new_full_null(fields, length))
        },
        x @ PlArrayType::Object { .. } => {
            panic!("polars-array: cannot build a full null {x:?} typed array")
        },
    }
}

/// Panics unless the `length` elements starting at `start` are in bounds of an array of `array_len`
/// elements.
pub(crate) fn assert_subslice(array_len: usize, start: usize, length: usize) {
    assert!(
        start
            .checked_add(length)
            .is_some_and(|end| end <= array_len),
        "subslice of {length} elements at {start} is out of bounds of an array of length \
         {array_len}",
    );
}

/// Appends the `length` bits of `validity` starting at `start` to `dst`.
///
/// This and the three below are the validity half of every builder in this crate, and they read
/// nothing of the array they came from but its mask. They are `#[inline(never)]` so that the one
/// copy of each stays one copy: inlined, each would be pasted into every builder's method — and
/// into all seventeen of the primitive builder's, once per element type — for no gain, since a
/// mask is appended a whole array at a time.
#[inline(never)]
pub(crate) fn subslice_extend_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    start: usize,
    length: usize,
) {
    if length == 0 {
        return;
    }

    match validity {
        None => dst.extend_constant(length, true),
        Some(validity) => match validity.scalar_value() {
            Some(bit) => dst.extend_constant(length, bit),
            // The mask is not scalar, so it holds one bit per element.
            None => dst.subslice_extend_from_opt_validity(validity.flat_bitmap(), start, length),
        },
    }
}

/// Appends each of the `length` bits of `validity` starting at `start` `repeats` times over.
#[inline(never)]
pub(crate) fn subslice_extend_each_repeated_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    start: usize,
    length: usize,
    repeats: usize,
) {
    if length == 0 || repeats == 0 {
        return;
    }

    match validity {
        None => dst.extend_constant(length * repeats, true),
        Some(validity) => match validity.scalar_value() {
            Some(bit) => dst.extend_constant(length * repeats, bit),
            // The mask is not scalar, so it holds one bit per element.
            None => dst.subslice_extend_each_repeated_from_opt_validity(
                validity.flat_bitmap(),
                start,
                length,
                repeats,
            ),
        },
    }
}

/// Appends the bit of `validity` at every index of `idxs`, in the order they are given.
///
/// # Safety
/// Every index must be smaller than the length of `validity`.
#[inline(never)]
pub(crate) unsafe fn gather_extend_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    idxs: &[IdxSize],
) {
    match validity {
        None => dst.extend_constant(idxs.len(), true),
        Some(validity) => match validity.scalar_value() {
            Some(bit) => dst.extend_constant(idxs.len(), bit),
            // SAFETY: the mask is flat, so the indices are in bounds of the bitmap itself.
            None => unsafe {
                dst.gather_extend_from_opt_validity(validity.flat_bitmap(), idxs);
            },
        },
    }
}

/// Appends the bit of `validity` at every index of `idxs`, in the order they are given, with an
/// index that is not smaller than `length` standing for an unset bit.
#[inline(never)]
pub(crate) fn opt_gather_extend_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    idxs: &[IdxSize],
    length: usize,
) {
    match validity {
        None => dst.opt_gather_extend_from_opt_validity(None, idxs, length),
        Some(validity) => match validity.scalar_value() {
            // Every in-bounds index reads the one bit, and every other one is null.
            Some(bit) => {
                for idx in idxs {
                    dst.extend_constant(1, bit && (*idx as usize) < length);
                }
            },
            // The mask is not scalar, so it holds one bit per element.
            None => dst.opt_gather_extend_from_opt_validity(validity.flat_bitmap(), idxs, length),
        },
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_buffer::Buffer;

    use super::*;
    use crate::bitmap::PlBitmap;
    use crate::{
        PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray, PlNullArray,
        PlPrimitiveArray, PlStructArray,
    };

    /// One array of every array type, all of three elements, with a null in the middle.
    fn arrays() -> Vec<Box<dyn PlArray>> {
        let validity = Bitmap::from_iter([true, false, true]);
        vec![
            Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
                    .with_validity(Some(PlBitmap::from_bitmap(validity.clone()))),
            ),
            Box::new(
                PlBooleanArray::from_vec(vec![true, false, true])
                    .with_validity(Some(PlBitmap::from_bitmap(validity.clone()))),
            ),
            Box::new(
                PlBinaryArray::from_values_iter([b"foo".as_slice(), b"", b"bar"])
                    .with_validity(Some(PlBitmap::from_bitmap(validity.clone()))),
            ),
            Box::new(
                PlBinaryViewArray::from_values_iter([
                    b"foo".as_slice(),
                    b"bar",
                    b"a value that is too long to inline",
                ])
                .with_validity(Some(PlBitmap::from_bitmap(validity.clone()))),
            ),
            Box::new(
                PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4, 5, 6], 2)
                    .with_validity(Some(PlBitmap::from_bitmap(validity.clone()))),
            ),
            Box::new(PlStructArray::new(
                vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]))],
                3,
                Some(PlBitmap::from_bitmap(validity.clone())),
            )),
            Box::new(
                PlListArray::from_offsets(
                    Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                    Buffer::from(vec![0u64, 1, 2, 3]),
                )
                .with_validity(Some(PlBitmap::from_bitmap(validity.clone()))),
            ),
            Box::new(
                PlFixedSizeListArray::from_values(
                    Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6])),
                    2,
                )
                .with_validity(Some(PlBitmap::from_bitmap(validity))),
            ),
            Box::new(PlNullArray::new(3)),
        ]
    }

    #[test]
    fn a_builder_of_every_array_type_appends_arrays() {
        for array in arrays() {
            let mut builder = builder_like(&*array);
            assert!(builder.is_empty());

            builder.reserve(8);
            builder.extend(&*array, ShareStrategy::Always);
            builder.extend_nulls(2);
            builder.subslice_extend(&*array, 1, 2, ShareStrategy::Never);
            assert_eq!(builder.len(), 7);

            let built = builder.freeze();
            assert_eq!(built.array_type(), array.array_type());
            assert_eq!(built.len(), 7);

            // The elements are the ones appended, in the order they were appended in.
            assert_eq!(&built.sliced(0, 3), &array);
            assert_eq!(
                built.null_count(),
                array.null_count() + 2 + array.sliced(1, 2).null_count(),
            );
            assert_eq!(&built.sliced(6, 1), &array.sliced(2, 1));
        }
    }

    #[test]
    fn freeze_reset_leaves_an_empty_builder() {
        for array in arrays() {
            let mut builder = builder_like(&*array);
            builder.extend(&*array, ShareStrategy::Always);

            let built = builder.freeze_reset();
            assert_eq!(built.len(), 3);
            assert!(builder.is_empty());
            assert_eq!(builder.len(), 0);

            builder.extend_nulls(1);
            let built = builder.freeze();
            assert_eq!(built.len(), 1);
            assert_eq!(built.null_count(), 1);
        }
    }

    #[test]
    fn builder_like_follows_the_shape_of_a_nested_array() {
        let array = PlListArray::new_empty(Box::new(PlFixedSizeListArray::new_empty(
            Box::new(PlBooleanArray::new_empty()),
            3,
        )));

        let built = builder_like(&array).freeze();
        let built = built.as_any().downcast_ref::<PlListArray>().unwrap();
        let values = built
            .values()
            .as_any()
            .downcast_ref::<PlFixedSizeListArray>()
            .unwrap();

        assert_eq!(values.width(), 3);
        assert_eq!(
            values.flat_values().unwrap().array_type(),
            PlArrayType::Boolean
        );
    }
}
