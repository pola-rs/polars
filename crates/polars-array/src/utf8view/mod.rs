//! The string view of a [`PlBinaryViewArray`].
//!
//! The arrays in this crate are physical storage: a [`PlBinaryViewArray`] is a sequence of byte
//! strings and nothing in it says those bytes are a string, which is why it never validates UTF-8
//! and hands its elements out as `&[u8]`. [`PlUtf8ViewArray`] is the wrapper that carries the
//! promise that they *are* one, so that the code above this crate that knows its bytes are strings
//! — a `StringChunked` and the chunks it is made of — reads an element as the `&str` it is without
//! validating it again.
//!
//! The wrapper is `repr(transparent)` over the [`PlBinaryViewArray`] such a chunk actually is, so
//! a chunk can also be *borrowed* as one — [`PlUtf8ViewArray::from_binview_ref_unchecked`] —
//! without going through the buffers.
//!
//! # The invariant
//!
//! Every element of a [`PlUtf8ViewArray`] — including the ones masked off as null, which a
//! validity mask can be put back over at any time — is valid UTF-8. Constructing one is therefore
//! `unsafe`; [`PlUtf8ViewArray::from_binview`] is the checked constructor that establishes it.
//!
//! # It is its own array type
//!
//! This is the one array in the crate that carries a logical type, and it carries it all the way
//! through the trait object: a boxed [`PlUtf8ViewArray`] reports [`PlArrayType::Utf8View`], is
//! downcast back to a [`PlUtf8ViewArray`], and exports as an Arrow
//! [`Utf8ViewArray`](arrow::array::Utf8ViewArray) rather than as the `BinaryViewArray` its bytes
//! are stored as. So the promise survives being boxed, concatenated, built or round-tripped
//! through Arrow, and code that holds a `dyn PlArray` never has to reach for an `unsafe` wrapper
//! to recover it.

use std::any::Any;

use arrow::array::View;
use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_err};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::binview::PlBinaryViewArray;
use crate::bitmap::PlBitmapRef;
use crate::flat::Flat;

mod builder;
mod iterator;

pub use builder::PlUtf8ViewArrayBuilder;
pub use iterator::{PlUtf8ViewIter, PlUtf8ViewValuesIter};

/// A [`PlBinaryViewArray`] whose every element is valid UTF-8.
///
/// This is the counterpart of [`Utf8ViewArray`](arrow::array::Utf8ViewArray), and every method on
/// it is the method of the same name on the array it wraps, over `&str` instead of `&[u8]`. See the
/// [module docs](self) for the invariant and for why this is a borrowed view of a chunk rather than
/// the type a chunk is stored as.
///
/// # Example
/// ```
/// use polars_array::{PlBinaryViewArray, PlUtf8ViewArray};
///
/// let array: PlUtf8ViewArray = [Some("foo"), None].into_iter().collect();
/// assert_eq!(array.get(0), Some("foo"));
/// assert_eq!(array.get(1), None);
///
/// // A scalar array of a billion elements costs a single view of memory.
/// let scalar = PlUtf8ViewArray::new_scalar("foo", 1_000_000_000);
/// assert_eq!(scalar.value(999_999_999), "foo");
///
/// // The bytes of an array that is not known to be a string are checked once.
/// let bytes = PlBinaryViewArray::from_values_iter([b"\xff".as_slice()]);
/// assert!(PlUtf8ViewArray::from_binview(bytes).is_err());
/// ```
#[derive(Clone)]
#[repr(transparent)]
pub struct PlUtf8ViewArray(PlBinaryViewArray);

impl PlUtf8ViewArray {
    /// Wraps `array`, checking that every one of its elements is valid UTF-8.
    ///
    /// This is `O(total_bytes_len)`: the elements masked off as null are checked too, because a
    /// validity mask can be replaced without the bytes under it changing.
    pub fn from_binview(array: PlBinaryViewArray) -> PolarsResult<Self> {
        validate_utf8(&array)?;
        // SAFETY: just validated.
        Ok(unsafe { Self::from_binview_unchecked(array) })
    }

    /// Wraps `array` without checking that its elements are valid UTF-8.
    ///
    /// # Safety
    /// Every element of `array`, including the ones masked off as null, must be valid UTF-8.
    #[inline(always)]
    pub const unsafe fn from_binview_unchecked(array: PlBinaryViewArray) -> Self {
        Self(array)
    }

    /// Borrows `array` as an array of strings without checking that its elements are valid UTF-8.
    ///
    /// # Safety
    /// Every element of `array`, including the ones masked off as null, must be valid UTF-8.
    #[inline(always)]
    pub const unsafe fn from_binview_ref_unchecked(array: &PlBinaryViewArray) -> &Self {
        // SAFETY: `Self` is `repr(transparent)` over the array it wraps.
        unsafe { &*(std::ptr::from_ref(array).cast::<Self>()) }
    }

    /// The bytes of these strings, giving up the promise that they are one.
    #[inline(always)]
    pub const fn as_binview(&self) -> &PlBinaryViewArray {
        &self.0
    }

    /// The bytes of these strings, giving up the promise that they are one.
    #[inline(always)]
    pub fn into_binview(self) -> PlBinaryViewArray {
        self.0
    }

    /// An empty array.
    #[inline]
    pub fn new_empty() -> Self {
        Self(PlBinaryViewArray::new_empty())
    }

    /// An array of `length` nulls.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self(PlBinaryViewArray::new_full_null(length))
    }

    /// An array of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: &str, length: usize) -> Self {
        Self(PlBinaryViewArray::new_scalar(value.as_bytes(), length))
    }

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> &str {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.0.value(i)) }
    }

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &str {
        // SAFETY: the caller keeps `i` in bounds, and the elements are valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.0.value_unchecked(i)) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&str> {
        // SAFETY: the elements of this array are valid UTF-8.
        self.0
            .get(i)
            .map(|v| unsafe { std::str::from_utf8_unchecked(v) })
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<&str> {
        // SAFETY: the caller keeps `i` in bounds, and the elements are valid UTF-8.
        unsafe {
            self.0
                .get_unchecked(i)
                .map(|v| std::str::from_utf8_unchecked(v))
        }
    }

    /// Whether this array is entirely stored in the scalar representation — see
    /// [`PlBinaryViewArray::is_scalar`].
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    /// The single value every element of a scalar array is, or `None` if this array is not scalar.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<&str>> {
        // SAFETY: the elements of this array are valid UTF-8.
        self.0
            .scalar_value()
            .map(|v| v.map(|v| unsafe { std::str::from_utf8_unchecked(v) }))
    }

    /// Iterates the elements, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlUtf8ViewValuesIter<'_> {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { PlUtf8ViewValuesIter::new(self.0.values_iter()) }
    }

    /// Iterates the elements, `None` for the null ones.
    #[inline]
    pub fn iter(&self) -> PlUtf8ViewIter<'_> {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { PlUtf8ViewIter::new(self.0.iter()) }
    }

    /// Iterates `length` elements, repeating the single element of a scalar array.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlUtf8ViewValuesIter<'_> {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { PlUtf8ViewValuesIter::new(self.0.broadcast_values_iter(length)) }
    }

    /// Iterates `length` elements, repeating the single element of a scalar array.
    #[inline]
    pub fn broadcast_iter(&self, length: usize) -> PlUtf8ViewIter<'_> {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { PlUtf8ViewIter::new(self.0.broadcast_iter(length)) }
    }

    /// Returns this array with its validity mask replaced by a flat one.
    #[inline]
    #[must_use]
    pub fn with_validity(self, validity: Option<Bitmap>) -> Self {
        Self(self.0.with_validity(validity))
    }

    /// Returns this array with its validity mask replaced by one that broadcasts over it.
    #[inline]
    #[must_use]
    pub fn with_validity_broadcast(self, validity: Option<Bitmap>) -> Self {
        Self(self.0.with_validity_broadcast(validity))
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    #[inline]
    #[must_use]
    pub fn sliced(self, offset: usize, length: usize) -> Self {
        Self(self.0.sliced(offset, length))
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[inline]
    #[must_use]
    pub unsafe fn sliced_unchecked(self, offset: usize, length: usize) -> Self {
        // SAFETY: the caller keeps the slice in bounds.
        Self(unsafe { self.0.sliced_unchecked(offset, length) })
    }

    /// Returns an array of `length` copies of the element at `index`.
    #[inline]
    #[must_use]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        Self(self.0.new_from_index(index, length))
    }

    /// The total number of bytes the elements of this array are.
    #[inline]
    pub fn total_bytes_len(&self) -> usize {
        self.0.total_bytes_len()
    }

    /// The views of this array, which index its data buffers.
    #[inline]
    pub fn flat_views(&self) -> Option<&Buffer<View>> {
        self.0.flat_views()
    }

    /// The data buffers the views of this array point into.
    #[inline]
    pub const fn data_buffers(&self) -> &Buffer<Buffer<u8>> {
        self.0.data_buffers()
    }

    /// Whether every backing buffer of this array holds one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.0.is_flat()
    }

    /// Returns this array in the flat representation.
    #[inline]
    pub fn to_flat(&self) -> Flat<Self> {
        // SAFETY: the inner array is written out flat, and the wrapper is transparent over it.
        unsafe { Flat::new(Self(self.0.to_flat().into_array())) }
    }

    /// Returns this array with every view replaced by what `update_view` makes of it.
    ///
    /// # Safety
    /// The views the closure hands back must uphold every invariant of a view: each must read
    /// bytes this array's data buffers hold, and those bytes must be valid UTF-8.
    pub unsafe fn apply_views<F: FnMut(View, &str) -> View>(&self, mut update_view: F) -> Self {
        // TODO(polars-array-scalar): a scalar array holds one view standing for every element, so
        // the views could be mapped in `O(1)` rather than written out flat first.
        let flat = self.to_flat().into_array();
        let length = flat.len();
        let (views, buffers, validity) = flat.0.to_flat().into_inner();

        let views: Vec<View> = views
            .as_slice()
            .iter()
            .enumerate()
            // SAFETY: `i` is in bounds of the array the views came from.
            .map(|(i, &view)| update_view(view, unsafe { self.value_unchecked(i) }))
            .collect();

        // SAFETY: the caller keeps every view reading bytes the buffers hold, and valid UTF-8.
        unsafe {
            Self::from_binview_unchecked(PlBinaryViewArray::new_unchecked(
                views.into(),
                buffers,
                length,
                validity,
            ))
        }
    }

    /// Borrows this array as a flat one, or `None` if any backing buffer is scalar.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the inner array is flat, and the wrapper is transparent over it.
        self.0.as_flat().map(|_| unsafe { Flat::new_ref(self) })
    }
}

/// Checks that every element of `array`, including the ones masked off as null, is valid UTF-8.
fn validate_utf8(array: &PlBinaryViewArray) -> PolarsResult<()> {
    // The validity mask is dropped rather than honoured: a null element still holds bytes, and
    // replacing the mask must not be able to expose bytes that were never checked.
    for value in array
        .to_flat()
        .into_array()
        .without_validity()
        .values_iter()
    {
        std::str::from_utf8(value).map_err(|e| polars_err!(ComputeError: "invalid utf8: {}", e))?;
    }
    Ok(())
}

impl Default for PlUtf8ViewArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<'a> FromIterator<Option<&'a str>> for PlUtf8ViewArray {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        // SAFETY: every value collected was a `&str`.
        unsafe { Self::from_binview_unchecked(iter.into_iter().collect()) }
    }
}

/// Compares two arrays element-wise, exactly like comparing the bytes under them.
impl PartialEq for PlUtf8ViewArray {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PlUtf8ViewArray {}

impl std::fmt::Debug for PlUtf8ViewArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PlUtf8ViewArray")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for &'a PlUtf8ViewArray {
    type Item = Option<&'a str>;
    type IntoIter = PlUtf8ViewIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl PlArray for PlUtf8ViewArray {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::Utf8View
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.0.validity()
    }

    #[inline]
    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    #[inline]
    fn slice(&mut self, offset: usize, length: usize) {
        self.0.slice(offset, length);
    }

    #[inline]
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        // SAFETY: the caller keeps the slice in bounds.
        unsafe { self.0.slice_unchecked(offset, length) };
    }

    #[inline]
    fn set_validity(&mut self, validity: Option<Bitmap>) {
        self.0.set_validity(validity);
    }

    #[inline]
    fn set_validity_broadcast(&mut self, validity: Option<Bitmap>) {
        self.0.set_validity_broadcast(validity);
    }

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        // SAFETY: the caller keeps `index` in bounds. Repeating one element of a valid string
        // array keeps every element valid UTF-8.
        let array = unsafe { self.0.new_from_index_unchecked(index, length) };
        Box::new(unsafe { Self::from_binview_unchecked(array) })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.clone())
    }

    #[inline]
    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        // A string array equals another string array with the same elements; a byte array of the
        // same bytes is a different array type, which `PlBinaryViewArray::eq_dyn` also rejects.
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StaticArray;

    /// A value of more than [`View::MAX_INLINE_SIZE`] bytes, which no view inlines.
    const LONG: &str = "a value that is too long to inline";

    #[test]
    fn elements_are_the_strings_the_bytes_are() {
        let arr: PlUtf8ViewArray = [Some("foo"), None, Some(LONG)].into_iter().collect();

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
        assert_eq!(arr.null_count(), 1);
        assert_eq!(arr.value(0), "foo");
        assert_eq!(arr.get(0), Some("foo"));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.total_bytes_len(), 3 + LONG.len());
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some("foo"), None, Some(LONG)],
        );
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>().len(), 3);
        assert_eq!(unsafe { arr.value_unchecked(2) }, LONG);
        assert_eq!(unsafe { arr.get_unchecked(1) }, None);

        // A scalar array of a billion elements holds one view, as it does for the bytes.
        let scalar = PlUtf8ViewArray::new_scalar(LONG, 1_000_000_000);

        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value(), Some(Some(LONG)));
        assert_eq!(scalar.value(999_999_999), LONG);

        // A single element broadcasts to as many as the caller asks for.
        let one = PlUtf8ViewArray::new_scalar(LONG, 1);
        assert_eq!(one.broadcast_iter(3).collect::<Vec<_>>(), [Some(LONG); 3]);
        assert_eq!(one.broadcast_values_iter(3).collect::<Vec<_>>(), [LONG; 3]);
    }

    #[test]
    fn the_bytes_under_a_null_are_validated_too() {
        let bytes = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"\xff"]);

        assert!(PlUtf8ViewArray::from_binview(bytes.clone()).is_err());

        // Masking the invalid element off is not enough: the mask can be replaced afterwards,
        // which would expose bytes that were never checked.
        let masked = bytes.with_validity(Some(Bitmap::from_iter([true, false])));
        assert!(PlUtf8ViewArray::from_binview(masked).is_err());

        let valid = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), LONG.as_bytes()]);
        let arr = PlUtf8ViewArray::from_binview(valid).expect("the bytes are valid UTF-8");
        assert_eq!(arr.value(1), LONG);
    }

    #[test]
    fn a_chunk_is_borrowed_as_an_array_of_strings() {
        let bytes = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar"]);

        // SAFETY: the bytes are valid UTF-8.
        let arr = unsafe { PlUtf8ViewArray::from_binview_ref_unchecked(&bytes) };

        assert_eq!(arr.value(1), "bar");
        assert_eq!(arr.as_binview(), &bytes);
        assert!(
            arr.data_buffers().is_same_buffer(bytes.data_buffers()),
            "the wrapper must borrow the chunk, not copy it",
        );
    }

    #[test]
    fn the_boxed_array_is_the_string_array() {
        // The UTF-8 promise survives being boxed: a `dyn PlArray` over a string array reports
        // `Utf8View` and downcasts back to a `PlUtf8ViewArray`, never to the bytes underneath.
        let arr: PlUtf8ViewArray = [Some("foo"), None].into_iter().collect();

        assert_eq!(arr.array_type(), PlArrayType::Utf8View);
        for boxed in [
            arr.to_boxed(),
            unsafe { arr.new_from_index_unchecked(0, 2) },
            arr.clone().into_boxed(),
        ] {
            assert_eq!(boxed.array_type(), PlArrayType::Utf8View);
            assert!(boxed.as_any().downcast_ref::<PlUtf8ViewArray>().is_some());
            assert!(
                boxed.as_any().downcast_ref::<PlBinaryViewArray>().is_none(),
                "a string array must not downcast to the bytes it is stored as",
            );
        }
    }

    #[test]
    fn a_string_array_is_not_equal_to_the_bytes_it_is_stored_as() {
        let arr: PlUtf8ViewArray = [Some("foo"), None].into_iter().collect();
        let bytes = arr.clone().into_binview();

        assert!(!arr.eq_dyn(&bytes), "the array types differ");
        assert!(!bytes.eq_dyn(&arr), "and the comparison is symmetric");

        let same: PlUtf8ViewArray = [Some("foo"), None].into_iter().collect();
        assert!(arr.eq_dyn(&same));
    }

    #[test]
    fn slicing_and_validity() {
        let arr: PlUtf8ViewArray = [Some("foo"), Some("bar"), Some(LONG)].into_iter().collect();

        let sliced = arr.clone().sliced(1, 2);
        assert_eq!(sliced.iter().collect::<Vec<_>>(), [Some("bar"), Some(LONG)]);
        assert_eq!(unsafe { arr.clone().sliced_unchecked(2, 1) }.value(0), LONG,);

        let masked = arr
            .clone()
            .with_validity(Some(Bitmap::from_iter([true, false, true])));
        assert_eq!(masked.get(1), None);
        assert_eq!(
            masked.value(1),
            "bar",
            "the bytes under a null are still there"
        );

        assert_eq!(
            arr.new_from_index(0, 2).iter().collect::<Vec<_>>(),
            [Some("foo"); 2]
        );
        assert_eq!(PlUtf8ViewArray::new_full_null(2).null_count(), 2);
        assert!(PlUtf8ViewArray::new_empty().is_empty());
        assert!(PlUtf8ViewArray::default().is_empty());
    }

    #[test]
    fn a_scalar_array_is_written_out_flat() {
        let scalar = PlUtf8ViewArray::new_scalar(LONG, 3);

        assert!(scalar.as_flat().is_none());
        let flat = scalar.to_flat();
        assert!(flat.is_flat());
        assert_eq!(flat, scalar);
        assert_eq!(flat.value(2), LONG);

        // An array that is already flat is borrowed rather than written out again.
        let arr: PlUtf8ViewArray = [Some("foo")].into_iter().collect();
        let flat = arr.as_flat().expect("the array is flat");
        assert_eq!(flat.value(0), "foo");
    }

    #[test]
    fn applying_views_sees_the_strings_and_writes_out_flat() {
        let arr = PlUtf8ViewArray::new_scalar(LONG, 3);

        // Every element is replaced by its first character, which a view inlines rather than
        // reading out of a data buffer.
        // SAFETY: an inlined view reads no bytes of the array at all, and the character is valid
        // UTF-8 because it was a prefix of one.
        let truncated = unsafe {
            arr.apply_views(|_, value| {
                assert_eq!(value, LONG);
                View::new_inline(&value.as_bytes()[..1])
            })
        };

        assert_eq!(truncated.iter().collect::<Vec<_>>(), [Some("a"); 3]);
        assert!(
            truncated.is_flat(),
            "a scalar array is written out to be mapped view by view",
        );
    }

    #[test]
    fn debug_formats_the_strings() {
        let arr: PlUtf8ViewArray = [Some("foo"), None].into_iter().collect();

        assert_eq!(format!("{arr:?}"), r#"PlUtf8ViewArray[Some("foo"), None]"#);
        assert_eq!(arr, arr.clone());
        assert_ne!(arr, PlUtf8ViewArray::new_scalar("foo", 2));
    }
}
