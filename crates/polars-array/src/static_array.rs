//! The typed counterpart of [`PlArray`].

use std::ops::Range;

use arrow::bitmap::Bitmap;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use bytemuck::Zeroable;

use crate::array::PlArray;
use crate::binary::{PlBinaryIter, PlBinaryValuesIter};
use crate::binview::{PlBinaryViewIter, PlBinaryViewValuesIter};
use crate::bitmap::{PlBitmapIter, PlBitmapRef};
use crate::boolean::PlBooleanIter;
use crate::broadcast::assert_broadcastable;
use crate::builder::StaticArrayBuilder;
use crate::fixed_size_binary::{PlFixedSizeBinaryIter, PlFixedSizeBinaryValuesIter};
use crate::fixed_size_list::{PlFixedSizeListIter, PlFixedSizeListValuesIter};
use crate::flat::Flat;
use crate::list::{PlListIter, PlListValuesIter};
use crate::primitive::{PlPrimitiveIter, PlPrimitiveValuesIter};
use crate::utf8view::{PlUtf8ViewIter, PlUtf8ViewValuesIter};
use crate::{
    PlBinaryArray, PlBinaryArrayBuilder, PlBinaryViewArray, PlBinaryViewArrayBuilder,
    PlBooleanArray, PlBooleanArrayBuilder, PlFixedSizeBinaryArray, PlFixedSizeBinaryArrayBuilder,
    PlFixedSizeListArray, PlFixedSizeListArrayBuilder, PlListArray, PlListArrayBuilder,
    PlNullArray, PlNullArrayBuilder, PlPrimitiveArray, PlPrimitiveArrayBuilder, PlStructArray,
    PlStructArrayBuilder, PlUtf8ViewArray, PlUtf8ViewArrayBuilder,
};

/// An array whose element type is known statically.
///
/// This is the typed counterpart of [`PlArray`]: where that trait exposes only what does not
/// depend on the element type, so that it can be a trait object, this one names the element type
/// and hands out the values themselves. Code that is generic over the array rather than over the
/// element type — a kernel written once for every array of this crate — is written against this
/// trait, and reaches everything else through the [`PlArray`] supertrait.
///
/// Reading an element goes through the same broadcast the concrete arrays use, so it does not
/// matter whether the array is flat or scalar; see [`crate::broadcast`] for the rules. Every
/// method here is the trait's view of the inherent method of the same name on the concrete array,
/// which remains the one to call when the concrete type is known — the two agree, and the inherent
/// one wins name resolution. The two methods that would collide with a [`PlArray`] method of the
/// same name carry a `_typed` suffix instead, since they return `Self` where [`PlArray`] returns a
/// `Box<dyn PlArray>`.
///
/// # Construction
///
/// Unlike [`StaticArray`](arrow::array::StaticArray) of the Arrow arrays, this trait has no
/// constructors: the arrays of this crate carry no logical type, so there is nothing an array of a
/// nested type could be constructed *from* — a [`PlListArray`] needs the array its lists are taken
/// over, which no length and no element type imply. Building an array generically is what
/// [`Self::Builder`] is for, and building one shaped like an array at hand is what
/// [`builder_like`](crate::builder::builder_like) is for.
///
/// # Example
/// ```
/// use polars_array::{PlBooleanArray, PlPrimitiveArray, StaticArray};
///
/// /// The elements of an array that are not null, in order.
/// fn valid_elements<A: StaticArray>(array: &A) -> Vec<A::ValueT<'_>> {
///     array.iter().flatten().collect()
/// }
///
/// assert_eq!(valid_elements(&PlPrimitiveArray::from_vec(vec![1i32, 2])), [1, 2]);
/// assert!(valid_elements(&PlBooleanArray::new_full_null(2)).is_empty());
/// ```
pub trait StaticArray: PlArray + Clone {
    /// One element of this array, borrowed from it.
    ///
    /// This is `()` for the arrays whose elements carry no value of their own: a
    /// [`PlStructArray`], whose values live in its field arrays, and a [`PlNullArray`], which has
    /// no values at all.
    type ValueT<'a>: Clone
    where
        Self: 'a;

    /// One element of this array, in a type whose all-zero bit pattern is a value of its own.
    ///
    /// A kernel that fills a [`Vec`] with one slot per element needs something to leave in the
    /// slots it has no value for — the elements it is about to mask off as null. That is what
    /// this type is: a stand-in for [`Self::ValueT`] that every value converts into, and which
    /// has something to leave a slot at when there is no value, [`Zeroable::zeroed`].
    ///
    /// For an element that is already zeroable — a number, a `bool` — this is the element type
    /// unchanged, and the zero is zero or `false`. For one that is a reference, which has no
    /// zero, it is [`Option`] of it, and the zero is [`None`]. Either way what a zeroed slot
    /// holds is not a value the kernel meant to write: it is the validity mask, put on afterwards
    /// with [`Self::with_validity_typed`], that says which slots those were.
    ///
    /// The vector is turned into the array by collecting it — see
    /// [`ZeroableArrayFromIter`](crate::collect::ZeroableArrayFromIter), which is this crate's
    /// [`from_zeroable_vec`](arrow::array::StaticArray::from_zeroable_vec).
    type ZeroableValueT<'a>: Zeroable + From<Self::ValueT<'a>>
    where
        Self: 'a;

    /// The iterator [`Self::values_iter`] hands out.
    type ValueIterT<'a>: DoubleEndedIterator<Item = Self::ValueT<'a>>
        + ExactSizeIterator
        + TrustedLen
        + Send
        + Sync
    where
        Self: 'a;

    /// The iterator [`Self::iter`] hands out.
    type IterT<'a>: DoubleEndedIterator<Item = Option<Self::ValueT<'a>>>
        + ExactSizeIterator
        + TrustedLen
        + Send
        + Sync
    where
        Self: 'a;

    /// The builder that builds this array.
    ///
    /// Constructing one takes whatever the array is built out of — the builder of the values of a
    /// [`PlListArray`], the width of a [`PlFixedSizeListArray`] — so it is the concrete builder
    /// that has the constructor, not this trait.
    type Builder: StaticArrayBuilder<Array = Self>;

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    fn value(&self, i: usize) -> Self::ValueT<'_> {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: `i` was just checked against the length.
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    unsafe fn value_unchecked(&self, i: usize) -> Self::ValueT<'_>;

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    fn get(&self, i: usize) -> Option<Self::ValueT<'_>> {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: `i` was just checked against the length.
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<Self::ValueT<'_>> {
        // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
        unsafe { (!self.is_null_unchecked(i)).then(|| self.value_unchecked(i)) }
    }

    /// Returns an iterator over the elements, ignoring validity.
    fn values_iter(&self) -> Self::ValueIterT<'_>;

    /// Returns an iterator over the optional elements.
    fn iter(&self) -> Self::IterT<'_>;

    /// Returns an iterator over `length` elements, repeating the single element of this array if
    /// that is all it holds.
    ///
    /// This is [`Self::values_iter`] of this array broadcast to `length` elements, which is `O(1)`
    /// — the copies are never materialized. See [`crate::broadcast`].
    ///
    /// # Panics
    /// Panics if `self.len()` is neither `length` nor one.
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_>;

    /// Returns an iterator over `length` optional elements, repeating the single element of this
    /// array if that is all it holds.
    ///
    /// This is [`Self::iter`] of this array broadcast to `length` elements, which is `O(1)` — the
    /// copies are never materialized. See [`crate::broadcast`].
    ///
    /// # Panics
    /// Panics if `self.len()` is neither `length` nor one.
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_>;

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// This is [`PlArray::with_validity`] without the trait object, which is what the `_typed`
    /// suffix is for.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    /// [`Self::with_validity_broadcast_typed`] is what installs the single bit every element
    /// shares; this function never infers that from a mask that happens to hold one bit.
    #[must_use]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self;

    /// Returns this array with its validity mask replaced by one that broadcasts over it.
    ///
    /// This is [`PlArray::with_validity_broadcast`] without the trait object, which is what the
    /// `_typed` suffix is for.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    #[must_use]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self;

    /// Returns an array of `length` copies of the element at `index`.
    ///
    /// This is [`PlArray::new_from_index`] without the trait object, which is what the `_typed`
    /// suffix is for.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[must_use]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self;

    /// Whether every backing buffer of this array holds one slot per element.
    ///
    /// This is what [`StaticArray::as_flat`] answers with a borrow rather than with a `bool`; see
    /// [`crate::broadcast`] for the rules.
    fn is_flat(&self) -> bool;

    /// The element every element of this array equals, if it is entirely stored in the scalar
    /// representation.
    ///
    /// The inner [`Option`] is that element, so an array of nothing but nulls yields
    /// `Some(None)`. Returns `None` for an empty array, which has no element to share, and
    /// whenever a backing buffer is flat over more than one element.
    ///
    /// This is what an elementwise kernel dispatches on to hand a repeated value to the
    /// single-value kernel rather than materialize [`PlArray::len`] copies of it with
    /// [`StaticArray::to_flat`] — it is the `O(1)` shortcut past a scalar array of unbounded
    /// length. It is the trait's view of the inherent `scalar_value` of the concrete arrays,
    /// which is the one to call when the concrete type is known.
    #[inline]
    fn scalar_value(&self) -> Option<Option<Self::ValueT<'_>>> {
        // SAFETY: the array is not empty, so element 0 is in bounds.
        (PlArray::is_scalar(self) && !self.is_empty()).then(|| unsafe { self.get_unchecked(0) })
    }

    /// Returns this array in the flat representation, writing out every buffer that is scalar.
    ///
    /// This is `O(1)` for an array that is already flat and `O(len)` for one that is not — see
    /// [`Flat`] and the concrete arrays for the exact cost.
    #[must_use]
    fn to_flat(&self) -> Flat<Self>;

    /// Borrows this array as a flat one, or `None` if any backing buffer is scalar.
    ///
    /// This is the `O(1)` half of [`StaticArray::to_flat`]: it never writes a buffer out, so the
    /// caller decides what an array that is not laid out flat costs.
    fn as_flat(&self) -> Option<&Flat<Self>>;

    /// Boxes this array as a [`PlArray`] trait object.
    ///
    /// This is [`Box::new`] for every array in this crate. A wrapper in another crate that is
    /// `repr(transparent)` over one of them — carrying an invariant the array itself does not,
    /// the way a string array is a byte array whose bytes are UTF-8 — overrides this to box the
    /// array it wraps, so that the trait object always downcasts to the array it really is.
    #[inline]
    fn into_boxed(self) -> Box<dyn PlArray>
    where
        Self: Sized,
    {
        Box::new(self)
    }
}

impl<T: NativeType> StaticArray for PlPrimitiveArray<T> {
    type ValueT<'a> = T;
    type ZeroableValueT<'a> = T;
    type ValueIterT<'a> = PlPrimitiveValuesIter<'a, T>;
    type IterT<'a> = PlPrimitiveIter<'a, T>;
    type Builder = PlPrimitiveArrayBuilder<T>;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> T {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<T> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlBooleanArray {
    type ValueT<'a> = bool;
    type ZeroableValueT<'a> = bool;
    type ValueIterT<'a> = PlBitmapIter<'a>;
    type IterT<'a> = PlBooleanIter<'a>;
    type Builder = PlBooleanArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> bool {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<bool> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlBinaryArray {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
    type ValueIterT<'a> = PlBinaryValuesIter<'a>;
    type IterT<'a> = PlBinaryIter<'a>;
    type Builder = PlBinaryArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlBinaryViewArray {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
    type ValueIterT<'a> = PlBinaryViewValuesIter<'a>;
    type IterT<'a> = PlBinaryViewIter<'a>;
    type Builder = PlBinaryViewArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// The elements are the strings the wrapper promises they are — see [`crate::utf8view`] — so
/// [`Self::into_boxed`] hands out the *inner* array, which is what a `dyn PlArray` of
/// [`BinaryView`](crate::PlArrayType::BinaryView) is expected to be.
impl StaticArray for PlUtf8ViewArray {
    type ValueT<'a> = &'a str;
    type ZeroableValueT<'a> = Option<&'a str>;
    type ValueIterT<'a> = PlUtf8ViewValuesIter<'a>;
    type IterT<'a> = PlUtf8ViewIter<'a>;
    type Builder = PlUtf8ViewArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &str {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&str> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }

    #[inline]
    fn into_boxed(self) -> Box<dyn PlArray> {
        // The trait object is the array this wrapper is transparent over — see the module docs.
        Box::new(self.into_binview())
    }
}

impl StaticArray for PlFixedSizeBinaryArray {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
    type ValueIterT<'a> = PlFixedSizeBinaryValuesIter<'a>;
    type IterT<'a> = PlFixedSizeBinaryIter<'a>;
    type Builder = PlFixedSizeBinaryArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlListArray {
    type ValueT<'a> = Box<dyn PlArray>;
    type ZeroableValueT<'a> = Option<Box<dyn PlArray>>;
    type ValueIterT<'a> = PlListValuesIter<'a>;
    type IterT<'a> = PlListIter<'a>;
    type Builder = PlListArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> Box<dyn PlArray> {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<Box<dyn PlArray>> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlFixedSizeListArray {
    type ValueT<'a> = Box<dyn PlArray>;
    type ZeroableValueT<'a> = Option<Box<dyn PlArray>>;
    type ValueIterT<'a> = PlFixedSizeListValuesIter<'a>;
    type IterT<'a> = PlFixedSizeListIter<'a>;
    type Builder = PlFixedSizeListArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> Box<dyn PlArray> {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<Box<dyn PlArray>> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// A [`PlStructArray`] holds no values of its own: an element is a row across the field arrays,
/// which are reached through [`PlStructArray::fields`] and read as the arrays they are. What is
/// left of an element is whether it is null, so the value of one is `()`.
impl StaticArray for PlStructArray {
    type ValueT<'a> = ();
    type ZeroableValueT<'a> = ();
    type ValueIterT<'a> = std::iter::RepeatN<()>;
    type IterT<'a> = PlUnitIter<'a>;
    type Builder = PlStructArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, _i: usize) -> Self::ValueT<'_> {}

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        std::iter::repeat_n((), self.len())
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        PlUnitIter::new(self.validity(), self.len())
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        assert_broadcastable(self.len(), length);
        std::iter::repeat_n((), length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        assert_broadcastable(self.len(), length);
        PlUnitIter::new(self.validity().map(|v| v.broadcast(length)), length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// A [`PlNullArray`] is nothing but a length: every element is null, and there is no value under
/// the mask, so the value of an element is `()` and [`StaticArray::get`] is always `None`.
impl StaticArray for PlNullArray {
    type ValueT<'a> = ();
    type ZeroableValueT<'a> = ();
    type ValueIterT<'a> = std::iter::RepeatN<()>;
    type IterT<'a> = PlUnitIter<'a>;
    type Builder = PlNullArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, _i: usize) -> Self::ValueT<'_> {}

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        std::iter::repeat_n((), self.len())
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        PlUnitIter::new(Some(self.validity()), self.len())
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        assert_broadcastable(self.len(), length);
        std::iter::repeat_n((), length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        assert_broadcastable(self.len(), length);
        PlUnitIter::new(Some(self.validity().broadcast(length)), length)
    }

    /// Returns this array unchanged: an array of nothing but nulls has no element a mask could
    /// make valid, exactly as [`PlArray::set_validity`] documents.
    #[inline]
    fn with_validity_typed(self, _validity: Option<Bitmap>) -> Self {
        self
    }

    /// Returns this array unchanged, exactly as [`Self::with_validity_typed`] does.
    #[inline]
    fn with_validity_broadcast_typed(self, _validity: Option<Bitmap>) -> Self {
        self
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// Iterator over the optional elements of an array whose elements carry no value of their own.
///
/// This is what a [`PlStructArray`] and a [`PlNullArray`] iterate as: all there is to an element
/// of one is whether it is null, so an element is `()` and the iterator is the validity mask under
/// another name. A scalar mask is not materialized, so this is `O(1)` in memory regardless of the
/// length.
#[derive(Clone)]
pub struct PlUnitIter<'a> {
    validity: Option<PlBitmapRef<'a>>,
    range: Range<usize>,
}

impl<'a> PlUnitIter<'a> {
    /// # Panics
    /// Panics if `validity` does not have `length` bits.
    #[inline]
    fn new(validity: Option<PlBitmapRef<'a>>, length: usize) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            validity,
            range: 0..length,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Option<()> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the mask.
        self.validity
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
            .then_some(())
    }
}

impl Iterator for PlUnitIter<'_> {
    type Item = Option<()>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|i| self.get(i))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth(n).map(|i| self.get(i))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl DoubleEndedIterator for PlUnitIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlUnitIter<'_> {}
unsafe impl TrustedLen for PlUnitIter<'_> {}

#[cfg(test)]
mod tests {
    use polars_buffer::Buffer;

    use super::*;

    /// The elements of an array, in order, as they format — every accessor of the trait over one
    /// array.
    fn elements<A: StaticArray>(array: &A) -> Vec<String>
    where
        for<'a> A::ValueT<'a>: std::fmt::Debug,
    {
        let by_index: Vec<String> = (0..array.len())
            .map(|i| format!("{:?}", array.get(i)))
            .collect();

        // Every way of reading the elements agrees with reading them one by one.
        assert_eq!(
            array.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>(),
            by_index,
        );
        assert_eq!(
            array
                .iter()
                .rev()
                .map(|v| format!("{v:?}"))
                .collect::<Vec<_>>(),
            by_index.iter().rev().cloned().collect::<Vec<_>>(),
        );
        assert_eq!(array.values_iter().len(), array.len());
        assert_eq!(array.iter().len(), array.len());

        for (i, element) in by_index.iter().enumerate() {
            assert_eq!(&format!("{:?}", unsafe { array.get_unchecked(i) }), element);
            // The value is there whether or not the element is null.
            let _ = array.value(i);
            let _ = unsafe { array.value_unchecked(i) };
        }

        by_index
    }

    #[test]
    fn primitive_elements() {
        let array: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();

        assert_eq!(elements(&array), ["Some(1)", "None", "Some(3)"]);
        assert_eq!(StaticArray::value(&array, 0), 1);
        assert_eq!(array.values_iter().collect::<Vec<_>>().len(), 3);
    }

    #[test]
    fn boolean_elements() {
        let array: PlBooleanArray = [Some(true), None].into_iter().collect();

        assert_eq!(elements(&array), ["Some(true)", "None"]);
    }

    #[test]
    fn binary_elements() {
        let array: PlBinaryArray = [Some(b"foo".as_slice()), None].into_iter().collect();

        assert_eq!(elements(&array), ["Some([102, 111, 111])", "None"]);
        assert_eq!(StaticArray::value(&array, 0), b"foo");
    }

    #[test]
    fn binview_elements() {
        let array: PlBinaryViewArray = [Some(b"foo".as_slice()), None].into_iter().collect();

        assert_eq!(elements(&array), ["Some([102, 111, 111])", "None"]);
    }

    #[test]
    fn fixed_size_binary_elements() {
        let array = PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4], 2)
            .with_validity(Some(Bitmap::from_iter([true, false])));

        assert_eq!(elements(&array), ["Some([1, 2])", "None"]);
        assert_eq!(StaticArray::value(&array, 0), [1, 2]);
    }

    #[test]
    fn list_elements() {
        let array = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Buffer::from(vec![0u64, 2, 3]),
        );

        assert_eq!(elements(&array).len(), 2);
        assert_eq!(StaticArray::value(&array, 0).len(), 2);
    }

    #[test]
    fn fixed_size_list_elements() {
        let array = PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
            2,
        );

        assert_eq!(elements(&array).len(), 2);
        assert_eq!(StaticArray::value(&array, 1).len(), 2);
    }

    #[test]
    fn struct_elements() {
        let array = PlStructArray::new(
            vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2]))],
            2,
            Some(Bitmap::from_iter([true, false])),
        );

        assert_eq!(elements(&array), ["Some(())", "None"]);
    }

    #[test]
    fn null_elements() {
        let array = PlNullArray::new(2);

        assert_eq!(elements(&array), ["None", "None"]);
        // The mask cannot be replaced: every element of a null array stays null.
        assert!(array.with_validity_typed(None).is_null(1));
    }

    #[test]
    fn scalars_are_read_through_the_broadcast() {
        let array = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);

        assert_eq!(StaticArray::value(&array, 999_999_999), 7);
        assert_eq!(StaticArray::get(&array, 999_999_999), Some(7));
        assert_eq!(array.iter().nth(999_999_999), Some(Some(7)));

        // An array of a single element iterates as any number of copies of it, in `O(1)`.
        let one = PlPrimitiveArray::from_vec(vec![7i32]);
        assert_eq!(
            one.broadcast_values_iter(1_000_000_000).len(),
            1_000_000_000
        );
        assert_eq!(one.broadcast_iter(1_000_000_000).last(), Some(Some(7)));

        let one = PlStructArray::from_fields(vec![Box::new(one)]);
        assert_eq!(
            one.broadcast_values_iter(1_000_000_000).len(),
            1_000_000_000
        );
        assert_eq!(one.broadcast_iter(1_000_000_000).len(), 1_000_000_000);

        let nulls = PlNullArray::new(1);
        assert_eq!(nulls.broadcast_iter(1_000_000_000).last(), Some(None));
    }

    #[test]
    fn the_scalar_value_is_read_without_walking_the_array() {
        // A billion elements would not be walked in reasonable time; that this test finishes is
        // what shows the repeated element is read off the buffers.
        let array = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        assert_eq!(StaticArray::scalar_value(&array), Some(Some(7)));

        // An array of nothing but nulls repeats its null, whatever is under the mask.
        let nulls = PlBooleanArray::new_full_null(1_000_000_000);
        assert_eq!(StaticArray::scalar_value(&nulls), Some(None));
        assert_eq!(
            StaticArray::scalar_value(&PlNullArray::new(1_000_000_000)),
            Some(None)
        );

        // An array of one element repeats it; one that is flat over more elements repeats
        // nothing, and an empty one has no element to repeat.
        let one = PlPrimitiveArray::from_vec(vec![7i32]);
        assert_eq!(StaticArray::scalar_value(&one), Some(Some(7)));

        let flat = PlPrimitiveArray::from_vec(vec![7i32, 7]);
        assert_eq!(StaticArray::scalar_value(&flat), None);
        assert_eq!(StaticArray::scalar_value(&flat.sliced(0, 0)), None);

        // A flat mask over a scalar values buffer is still a buffer that is not shared.
        let masked = PlPrimitiveArray::new_scalar(7i32, 3)
            .with_validity(Some(Bitmap::from_iter([true, false, true])));
        assert_eq!(StaticArray::scalar_value(&masked), None);
    }

    #[test]
    #[should_panic(expected = "does not broadcast to length")]
    fn broadcasting_more_than_one_element_is_rejected() {
        let _ = PlNullArray::new(2).broadcast_iter(3);
    }

    #[test]
    fn the_builder_of_an_array_builds_that_array() {
        use crate::builder::ShareStrategy;
        use crate::{PlBooleanArrayBuilder, PlPrimitiveArrayBuilder};

        /// Every element of `array`, appended to the builder of the arrays it is one of.
        fn rebuild<A: StaticArray>(builder: &mut A::Builder, array: &A) -> A {
            builder.extend(array, ShareStrategy::Always);
            builder.freeze_reset()
        }

        let array: PlPrimitiveArray<i32> = [Some(1), None].into_iter().collect();
        assert_eq!(
            rebuild(&mut PlPrimitiveArrayBuilder::<i32>::new(), &array),
            array,
        );

        let array = PlBooleanArray::new_scalar(true, 3);
        assert_eq!(rebuild(&mut PlBooleanArrayBuilder::new(), &array), array);
    }

    #[test]
    fn typed_operations_keep_the_concrete_type() {
        let array = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        let nulled: PlPrimitiveArray<i32> = array
            .clone()
            .with_validity_broadcast_typed(Some(Bitmap::new_zeroed(1)));
        assert_eq!(nulled.null_count(), 3);

        let repeated: PlPrimitiveArray<i32> = array.new_from_index_typed(2, 4);
        assert_eq!(repeated, PlPrimitiveArray::new_scalar(3, 4));
    }

    /// Every array names a zeroable stand-in for its elements, including the ones that cannot be
    /// collected back from it: the stand-in is what a kernel keeps its slots in, and the slot of
    /// an element it has no value for is left at the zero.
    #[test]
    fn every_array_has_a_zeroable_stand_in_for_its_elements() {
        use bytemuck::Zeroable;

        /// The elements of `array`, with the slot of each null left zeroed.
        fn zeroable_values<A: StaticArray>(array: &A) -> Vec<A::ZeroableValueT<'_>> {
            array
                .iter()
                .map(|value| value.map_or_else(Zeroable::zeroed, Into::into))
                .collect()
        }

        let validity = Some(Bitmap::from_iter([true, false]));

        let array = PlPrimitiveArray::from_vec(vec![1i32, 2]).with_validity(validity.clone());
        assert_eq!(zeroable_values(&array), [1, 0]);

        let array = PlBooleanArray::new_scalar(true, 2).with_validity(validity.clone());
        assert_eq!(zeroable_values(&array), [true, false]);

        let array =
            PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4], 2).with_validity(validity.clone());
        assert_eq!(zeroable_values(&array), [Some([1, 2].as_slice()), None]);

        let array = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Buffer::from(vec![0u64, 2, 3]),
        )
        .with_validity(validity.clone());
        let values = zeroable_values(&array);
        assert_eq!(values[0].as_ref().map(|list| list.len()), Some(2));
        assert!(values[1].is_none());

        let array = PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1,
        )
        .with_validity(validity.clone());
        let values = zeroable_values(&array);
        assert_eq!(values[0].as_ref().map(|list| list.len()), Some(1));
        assert!(values[1].is_none());

        // The elements of these two carry no value, so there is nothing for the zero to stand in
        // for either.
        let array = PlStructArray::new(
            vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2]))],
            2,
            validity,
        );
        assert_eq!(zeroable_values(&array), [(), ()]);
        assert_eq!(zeroable_values(&PlNullArray::new(2)), [(), ()]);
    }
}
