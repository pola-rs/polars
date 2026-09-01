//! Building arrays element by element, or array by array.
//!
//! The arrays in this crate are immutable, so an array that is not built out of buffers that are
//! already laid out is built by a *builder*: a growable staging area that is appended to and then
//! [`freeze`](StaticArrayBuilder::freeze)n into the array it built. [`StaticArrayBuilder`] is the
//! typed trait every concrete builder implements, and [`PlArrayBuilder`] is its trait object,
//! which is what the builders of the nested arrays hold their children as.
//!
//! # What a builder appends
//!
//! A builder is fed whole arrays rather than single elements: [`extend`](StaticArrayBuilder::extend)
//! and its subslice, repeat and gather variants are what a concatenation, a slice-and-append or a
//! take is written in terms of. Which one to reach for follows the shape of the copy:
//!
//! * [`subslice_extend`](StaticArrayBuilder::subslice_extend) appends a contiguous run of
//!   elements, in order.
//! * [`subslice_extend_repeated`](StaticArrayBuilder::subslice_extend_repeated) appends that run
//!   `repeats` times over — `abcabcabc`.
//! * [`subslice_extend_each_repeated`](StaticArrayBuilder::subslice_extend_each_repeated) appends
//!   each of its elements `repeats` times — `aaabbbccc`.
//! * [`gather_extend`](StaticArrayBuilder::gather_extend) appends the elements at the given
//!   indices, in the order they are given, and
//!   [`opt_gather_extend`](StaticArrayBuilder::opt_gather_extend) does the same with
//!   out-of-bounds indices standing for nulls.
//!
//! The array being appended may be in either representation — a scalar array is not materialized
//! to be read, see [`crate::broadcast`] — but what a builder holds always has one slot per
//! element, so the array it freezes is [flat](crate::Flat). Concatenating arrays *while* keeping
//! the scalar representation wherever it survives is what [`crate::concatenate`] is for; a builder
//! is what to reach for when the result is going to be materialized anyway.
//!
//! # Sharing buffers
//!
//! Every buffer in this crate is cheaply cloneable, so appending an array can often adopt one of
//! its buffers instead of copying the bytes out of it. [`ShareStrategy`] is how the caller says
//! whether that is wanted: sharing is cheaper, but it keeps the whole buffer alive for as long as
//! the built array is, which is a poor trade when a handful of elements are copied out of a large
//! array that is about to be dropped. Only [`PlBinaryViewArrayBuilder`] and the builders of the
//! nested arrays over one have anything to share.
//!
//! # Example
//! ```
//! use polars_array::builder::{ShareStrategy, StaticArrayBuilder};
//! use polars_array::{PlPrimitiveArray, PlPrimitiveArrayBuilder};
//!
//! let lhs = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
//! let rhs = PlPrimitiveArray::new_scalar(7i32, 2);
//!
//! let mut builder = PlPrimitiveArrayBuilder::<i32>::new();
//! builder.extend(&lhs, ShareStrategy::Always);
//! builder.extend_nulls(1);
//! builder.subslice_extend(&rhs, 0, 2, ShareStrategy::Always);
//!
//! assert_eq!(builder.len(), 6);
//! let array = builder.freeze();
//! assert_eq!(array.values().as_slice(), [1, 2, 3, 0, 7, 7]);
//! assert_eq!(array.null_count(), 1);
//! ```

/// Whether a builder may adopt the buffers of the arrays it appends, rather than copying out of
/// them.
///
/// This is [`ShareStrategy`] of the Arrow builders, which the builders of this crate take for the
/// same reason: see the [module docs](self).
pub use arrow::array::builder::ShareStrategy;
use arrow::bitmap::OptBitmapBuilder;
use polars_utils::IdxSize;

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::static_array::StaticArray;
use crate::{
    PlBinaryViewArrayBuilder, PlBooleanArrayBuilder, PlFixedSizeListArray,
    PlFixedSizeListArrayBuilder, PlListArray, PlListArrayBuilder, PlNullArrayBuilder,
    PlPrimitiveArrayBuilder, PlStructArray, PlStructArrayBuilder,
    with_match_pl_primitive_array_type,
};

/// A builder of one concrete array type.
///
/// This is the typed builder trait: it names the array it builds, so what it appends are arrays of
/// that type rather than trait objects. See the [module docs](self) for what the extend methods do
/// and what a builder freezes; [`PlArrayBuilder`] is the trait object of this trait, which every
/// implementor gets for free.
pub trait StaticArrayBuilder: Send {
    /// The array this builder builds.
    type Array: StaticArray;

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// A [`PlListArrayBuilder`] is the one builder that does not pass this on to its child: how
    /// many values the elements of a list array reach is not implied by how many elements there
    /// are.
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
    /// The run of elements is what is repeated, so appending `abc` twice appends `abcabc`; it is
    /// [`Self::subslice_extend_each_repeated`] that appends `aabbcc`.
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
    /// It is each element that is repeated, so appending `abc` twice appends `aabbcc`; it is
    /// [`Self::subslice_extend_repeated`] that appends `abcabc`.
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
///
/// This is the counterpart of [`PlArray`] on the building side, and it is what the builders of the
/// nested arrays hold their children as: the values of a [`PlListArray`] are a `Box<dyn PlArray>`,
/// so the builder of those values is a `Box<dyn PlArrayBuilder>`. Every [`StaticArrayBuilder`] is
/// one, and the arrays it is fed are downcast to the type it builds — which is what makes appending
/// an array of the wrong type a panic rather than a compile error.
///
/// # Example
/// ```
/// use polars_array::builder::{PlArrayBuilder, ShareStrategy, builder_like};
/// use polars_array::{PlArray, PlPrimitiveArray};
///
/// let array = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
///
/// let mut builder = builder_like(&array);
/// builder.subslice_extend(&array, 1, 2, ShareStrategy::Always);
/// builder.extend_nulls(1);
///
/// let built = builder.freeze();
/// assert_eq!(built.len(), 3);
/// assert_eq!(built.null_count(), 1);
/// ```
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
    /// Panics if `other` is not of the type this builder builds, or if
    /// `start + length > other.len()`.
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
///
/// This is the counterpart of [`make_builder`](arrow::array::builder::make_builder), which takes
/// the dtype the built array is to have. The arrays in this crate carry no logical type, so what
/// stands in for it is an array of the physical shape the result is to have: the element type of a
/// [`PlPrimitiveArray`](crate::PlPrimitiveArray), the width of a [`PlFixedSizeListArray`], the
/// field arrays of a [`PlStructArray`] — recursively, since the builder of a nested array is built
/// out of the builders of its children. Nothing but the shape of `array` is read, so an empty array
/// does as well as one holding elements.
///
/// # Example
/// ```
/// use polars_array::builder::{PlArrayBuilder, ShareStrategy, builder_like};
/// use polars_array::{PlArray, PlListArray, PlPrimitiveArray};
///
/// let array = PlListArray::new_empty(Box::new(PlPrimitiveArray::<i32>::new_empty()));
///
/// let mut builder = builder_like(&array);
/// builder.extend_nulls(3);
///
/// let built = builder.freeze();
/// assert_eq!(built.array_type(), array.array_type());
/// assert_eq!(built.null_count(), 3);
/// ```
pub fn builder_like(array: &dyn PlArray) -> Box<dyn PlArrayBuilder> {
    match array.array_type() {
        PlArrayType::Primitive(_) => with_match_pl_primitive_array_type!(array, |T| {
            Box::new(PlPrimitiveArrayBuilder::<T>::new()) as Box<dyn PlArrayBuilder>
        })
        .expect("a primitive array has a primitive element type"),
        PlArrayType::Boolean => Box::new(PlBooleanArrayBuilder::new()),
        PlArrayType::BinaryView => Box::new(PlBinaryViewArrayBuilder::new()),
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
    }
}

/// Panics unless the `length` elements starting at `start` are in bounds of an array of
/// `array_len` elements.
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
/// A scalar mask is appended as the single bit it holds, repeated, rather than being materialized.
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
            None => dst.subslice_extend_from_opt_validity(Some(validity.bitmap()), start, length),
        },
    }
}

/// Appends each of the `length` bits of `validity` starting at `start` `repeats` times over.
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
            None => dst.subslice_extend_each_repeated_from_opt_validity(
                Some(validity.bitmap()),
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
                dst.gather_extend_from_opt_validity(Some(validity.bitmap()), idxs);
            },
        },
    }
}

/// Appends the bit of `validity` at every index of `idxs`, in the order they are given, with an
/// index that is not smaller than `length` standing for an unset bit.
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
            None => dst.opt_gather_extend_from_opt_validity(Some(validity.bitmap()), idxs, length),
        },
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_buffer::Buffer;

    use super::*;
    use crate::{
        PlBinaryViewArray, PlBooleanArray, PlNullArray, PlPrimitiveArray, PlStructArray,
        with_match_pl_primitive_array_type,
    };

    /// One array of every array type, all of three elements, with a null in the middle.
    fn arrays() -> Vec<Box<dyn PlArray>> {
        let validity = Bitmap::from_iter([true, false, true]);
        vec![
            Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).with_validity(Some(validity.clone())),
            ),
            Box::new(
                PlBooleanArray::from_vec(vec![true, false, true])
                    .with_validity(Some(validity.clone())),
            ),
            Box::new(
                PlBinaryViewArray::from_values_iter([
                    b"foo".as_slice(),
                    b"bar",
                    b"a value that is too long to inline",
                ])
                .with_validity(Some(validity.clone())),
            ),
            Box::new(PlStructArray::new(
                vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]))],
                3,
                Some(validity.clone()),
            )),
            Box::new(
                PlListArray::from_offsets(
                    Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                    Buffer::from(vec![0u64, 1, 2, 3]),
                )
                .with_validity(Some(validity.clone())),
            ),
            Box::new(
                PlFixedSizeListArray::from_values(
                    Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6])),
                    2,
                )
                .with_validity(Some(validity)),
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
    fn appending_no_elements_appends_nothing() {
        for array in arrays() {
            let mut builder = builder_like(&*array);

            // A subslice of no elements is in bounds wherever it starts, the end included.
            builder.subslice_extend(&*array, 0, 0, ShareStrategy::Always);
            builder.subslice_extend(&*array, array.len(), 0, ShareStrategy::Never);
            builder.subslice_extend_repeated(&*array, 1, 0, 3, ShareStrategy::Always);
            builder.subslice_extend_each_repeated(&*array, 1, 1, 0, ShareStrategy::Always);
            builder.extend_nulls(0);
            unsafe { builder.gather_extend(&*array, &[], ShareStrategy::Always) };
            builder.opt_gather_extend(&*array, &[], ShareStrategy::Always);

            // So is every element of an array that holds none.
            let empty = array.sliced(0, 0);
            builder.extend(&*empty, ShareStrategy::Always);

            assert!(builder.is_empty());
            assert!(builder.freeze().is_empty());
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
    fn an_empty_builder_freezes_an_empty_array() {
        for array in arrays() {
            let built = builder_like(&*array).freeze();

            assert!(built.is_empty());
            assert_eq!(built.array_type(), array.array_type());
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
        assert_eq!(values.values().array_type(), PlArrayType::Boolean);
    }

    #[test]
    fn builder_like_follows_the_element_type_of_a_primitive_array() {
        fn builds_the_same_element_type<T: arrow::types::NativeType>() -> bool {
            let array = PlPrimitiveArray::<T>::new_empty();
            let built = builder_like(&array).freeze();
            with_match_pl_primitive_array_type!(&*built, |E| {
                std::any::TypeId::of::<E>() == std::any::TypeId::of::<T>()
            })
            .unwrap()
        }

        assert!(builds_the_same_element_type::<u8>());
        assert!(builds_the_same_element_type::<i64>());
        assert!(builds_the_same_element_type::<f64>());
        assert!(builds_the_same_element_type::<arrow::array::View>());
    }

    #[test]
    #[should_panic(expected = "cannot append a Boolean array to a builder")]
    fn appending_an_array_of_another_type_panics() {
        let mut builder = builder_like(&PlPrimitiveArray::<i32>::new_empty());
        builder.extend(&PlBooleanArray::from_vec(vec![true]), ShareStrategy::Always);
    }
}
