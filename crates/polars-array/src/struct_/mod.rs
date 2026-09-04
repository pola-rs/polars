use std::borrow::Cow;

use arrow::bitmap::{Bitmap, and};
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::{
    is_flat_buffer_len, is_scalar_buffer_len, is_valid_buffer_len, normalize_validity,
    scalar_buffer_len, slice_validity,
};
use crate::flat::Flat;

mod builder;

pub use builder::PlStructArrayBuilder;

/// An immutable, cheaply cloneable sequence of `length` optional rows, one value per field array.
#[derive(Clone)]
pub struct PlStructArray {
    /// Scalar: every field is scalar
    fields: Vec<Box<dyn PlArray>>,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlStructArray {
    /// Creates a flat [`PlStructArray`] out of its internal components.
    ///
    /// # Errors
    /// This function errors if any field does not have exactly `length` elements, or if `validity`
    /// does not hold exactly `length` bits.
    pub fn try_new(
        fields: Vec<Box<dyn PlArray>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        validate_fields(&fields, length)?;

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_flat_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is not flat for an array of length {}",
                validity.len(), length,
            );
        }

        Ok(Self {
            fields,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlStructArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(fields: Vec<Box<dyn PlArray>>, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(fields, length, validity).unwrap()
    }

    /// Creates a flat [`PlStructArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// Every field must have exactly `length` elements, and `validity` must hold exactly `length`
    /// bits.
    #[inline]
    pub unsafe fn new_unchecked(
        fields: Vec<Box<dyn PlArray>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(fields.iter().all(|field| field.len() == length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_flat_buffer_len(v.len(), length))
            );
        }

        Self {
            fields,
            length,
            validity,
        }
    }

    /// Creates a [`PlStructArray`] out of its internal components and a scalar validity mask.
    ///
    /// # Errors
    /// This function errors if any field does not have exactly `length` elements, or if `validity`
    /// is not scalar for `length`, per [`is_scalar_buffer_len`].
    pub fn try_new_broadcast(
        fields: Vec<Box<dyn PlArray>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        validate_fields(&fields, length)?;

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_scalar_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is not the single bit the {} elements of a broadcast \
                 array share",
                validity.len(), length,
            );
        }

        Ok(Self {
            fields,
            length,
            validity: normalize_validity(validity, length),
        })
    }

    /// Creates a [`PlStructArray`] out of its internal components and a scalar validity mask.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(
        fields: Vec<Box<dyn PlArray>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new_broadcast(fields, length, validity).unwrap()
    }

    /// Creates a [`PlStructArray`] out of its internal components and a scalar validity mask,
    /// without validating them.
    ///
    /// # Safety
    /// Every field must have exactly `length` elements, and `validity` must be scalar for `length`,
    /// per [`is_scalar_buffer_len`].
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        fields: Vec<Box<dyn PlArray>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(fields.iter().all(|field| field.len() == length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_scalar_buffer_len(v.len(), length))
            );
        }

        Self {
            fields,
            length,
            validity: normalize_validity(validity, length),
        }
    }

    /// Creates an empty [`PlStructArray`] without fields.
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            fields: Vec::new(),
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid [`PlStructArray`] from `fields`, taking its length from them.
    ///
    /// # Panics
    /// Panics if `fields` is empty — an array without fields has no length to take — or if the
    /// fields do not all have the same length.
    pub fn from_fields(fields: Vec<Box<dyn PlArray>>) -> Self {
        let length = fields
            .first()
            .expect("cannot infer the length of a struct array without fields")
            .len();
        Self::new(fields, length, None)
    }

    /// Creates a [`PlStructArray`] of `length` nulls over `fields`, in `O(1)` extra memory.
    ///
    /// # Panics
    /// Panics if any field does not have exactly `length` elements.
    #[inline]
    pub fn new_full_null(fields: Vec<Box<dyn PlArray>>, length: usize) -> Self {
        Self::new_broadcast(
            fields,
            length,
            Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
        )
    }

    /// The number of elements in this array.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Whether this array holds no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The field arrays, each holding [`Self::len`] elements.
    #[inline]
    pub fn fields(&self) -> &[Box<dyn PlArray>] {
        &self.fields
    }

    /// The number of field arrays.
    #[inline]
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// The field array at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.num_fields()`.
    #[inline]
    pub fn field(&self, i: usize) -> &dyn PlArray {
        &*self.fields[i]
    }

    /// Consumes this array into its internal components.
    #[inline]
    pub fn into_inner(self) -> (Vec<Box<dyn PlArray>>, usize, Option<Bitmap>) {
        (self.fields, self.length, self.validity)
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.length);
        // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        !self.is_valid(i)
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        unsafe { !self.is_valid_unchecked(i) }
    }

    /// The number of null elements.
    ///
    /// Inlined so that an array with no mask to count is answered without a call at all: one left
    /// standing is an opaque write as far as the compiler is concerned, and sinks behind it every
    /// fact the caller had established about the array — the representation of its buffers
    /// included — which is exactly what a caller asking [`Self::has_nulls`] ahead of a walk is
    /// trying to hand the walk.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask with a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_flat_buffer_len(validity.len(), self.length),
                "validity mask of length {} is not flat for an array of length {}",
                validity.len(),
                self.length,
            );
        }
        self.validity = validity;
    }

    /// Returns this array with its validity mask replaced by one that broadcasts over it.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    #[must_use]
    pub fn with_validity_broadcast(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity_broadcast(validity);
        self
    }

    /// Replaces the validity mask with one that broadcasts over this array.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    pub fn set_validity_broadcast(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_valid_buffer_len(validity.len(), self.length),
                "validity mask of length {} is neither flat nor scalar for an array of length {}",
                validity.len(),
                self.length,
            );
        }
        self.validity = normalize_validity(validity, self.length);
    }

    /// Drops the validity mask, making every row valid.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.validity = None;
        self
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.length,
            "the offset of the new slice must be smaller than the length of the array",
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // Each field slices itself, keeping whichever representation it is in.
        for field in self.fields.iter_mut() {
            unsafe { field.slice_unchecked(offset, length) };
        }

        unsafe { slice_validity(&mut self.validity, self.length, offset, length) };

        self.length = length;
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    pub fn sliced(mut self, offset: usize, length: usize) -> Self {
        self.slice(offset, length);
        self
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
        unsafe { self.slice_unchecked(offset, length) };
        self
    }

    /// Creates a [`PlStructArray`] of `length` copies of the row at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlStructArray`] of `length` copies of the row at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        let is_null = unsafe { self.is_null_unchecked(index) };

        // The field values of a null row are undetermined, so they are repeated as they are found:
        // it is the mask that makes every row of the result null.
        let fields = self
            .fields
            .iter()
            .map(|field| unsafe {
                if is_null {
                    // The row is masked off in the field it is repeated out of, which holds this
                    // array's elements rather than the result's — and there is at least one of
                    // them, since `index` is in bounds — so the mask is a single bit either way.
                    field
                        .with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
                        .new_from_index_unchecked(index, length)
                } else {
                    field.new_from_index_unchecked(index, length)
                }
            })
            .collect();

        let validity = is_null.then(|| Bitmap::new_zeroed(scalar_buffer_len(length)));

        // SAFETY: every field repeated one element `length` times, so it holds `length` elements,
        // and the mask holds the slots a scalar mask of that length holds.
        unsafe { Self::new_broadcast_unchecked(fields, length, validity) }
    }

    /// Whether every backing buffer of this array holds one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        !self.validity_is_scalar()
    }

    /// Whether this array is a single row repeated over its length, in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.validity().is_none_or(|validity| validity.is_scalar())
            && self.fields.iter().all(|field| field.is_scalar())
    }

    /// Returns this array in the flat representation, writing out a scalar validity mask and
    /// borrowing this array itself if its mask is not scalar.
    #[must_use]
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let validity = self
            .validity()
            .map(|validity| validity.to_flat().into_owned());

        // SAFETY: the fields are untouched and still hold `length` elements each, and the mask was
        // just written out to one bit per element.
        let array = unsafe { Self::new_unchecked(self.fields.clone(), self.length, validity) };

        // SAFETY: the mask is flat, and a struct array has no other buffer of its own.
        Cow::Owned(unsafe { Flat::new(array) })
    }

    /// Borrows this array as a flat one, or `None` if its validity mask is scalar.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: `is_flat` is exactly the invariant of `Flat`.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

/// Returns `field` with `mask` merged into its validity, so that the undetermined values of rows
/// the struct array masks out are ignored when comparing fields.
fn masked(field: &dyn PlArray, mask: PlBitmapRef<'_>) -> Box<dyn PlArray> {
    let validity = match field.validity() {
        Some(field_validity) => and(&field_validity.to_flat(), &mask.to_flat()),
        None => mask.to_flat().into_owned(),
    };
    field.with_validity(Some(validity))
}

impl Default for PlStructArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

/// Compares two arrays row-wise; the representation (flat or scalar) is irrelevant, and so are the
/// field values of null rows.
impl PartialEq for PlStructArray {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length || self.fields.len() != other.fields.len() {
            return false;
        }

        if !validity_eq(self.validity(), other.validity(), self.length) {
            return false;
        }

        // Every row is null on both sides, so every field value is undetermined and there is
        // nothing left to compare. This is also what keeps comparing two fully null scalar arrays
        // `O(1)`.
        if self.length > 0 && self.null_count() == self.length {
            return true;
        }

        // Comparing the fields is `O(1)` for scalar fields, so a scalar array is never walked row
        // by row.
        let mask = self.has_nulls().then(|| self.validity().unwrap());
        std::iter::zip(&self.fields, &other.fields).all(|(lhs, rhs)| match mask {
            Some(mask) => masked(&**lhs, mask) == masked(&**rhs, mask),
            None => lhs == rhs,
        })
    }
}

impl Eq for PlStructArray {}

impl std::fmt::Debug for PlStructArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The fields format their own scalar representation, so this never materializes one.
        let mut s = f.debug_struct("PlStructArray");
        s.field("length", &self.length);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("fields", &self.fields).finish()
    }
}

impl PlArray for PlStructArray {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::Struct
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.is_scalar()
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    #[inline]
    fn slice(&mut self, offset: usize, length: usize) {
        self.slice(offset, length)
    }

    #[inline]
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        unsafe { self.slice_unchecked(offset, length) }
    }

    #[inline]
    fn set_validity(&mut self, validity: Option<Bitmap>) {
        self.set_validity(validity)
    }

    #[inline]
    fn set_validity_broadcast(&mut self, validity: Option<Bitmap>) {
        self.set_validity_broadcast(validity)
    }

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        Box::new(unsafe { self.new_from_index_unchecked(index, length) })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.clone())
    }

    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}

/// Checks that every field of a struct array of `length` elements has that many elements itself.
fn validate_fields(fields: &[Box<dyn PlArray>], length: usize) -> PolarsResult<()> {
    for (i, field) in fields.iter().enumerate() {
        polars_ensure!(
            field.len() == length,
            ComputeError:
            "field {} has {} elements, but the struct array has length {}",
            i, field.len(), length,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlBooleanArray, PlPrimitiveArray};

    fn flat_fields() -> Vec<Box<dyn PlArray>> {
        vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
        ]
    }

    fn scalar_fields(length: usize) -> Vec<Box<dyn PlArray>> {
        vec![
            Box::new(PlPrimitiveArray::<i32>::new_scalar(1, length)),
            Box::new(PlBooleanArray::new_scalar(true, length)),
        ]
    }

    #[test]
    fn flat() {
        let arr = PlStructArray::from_fields(flat_fields());

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.num_fields(), 2);
        assert_eq!(arr.null_count(), 0);
        assert!(arr.is_valid(1));
        assert!(!arr.is_null(1));
        assert_eq!(
            arr.field(0)
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .value(2),
            3,
        );
    }

    #[test]
    fn scalar_fields_cost_nothing() {
        // Nothing here may walk a billion rows: the fields keep their own scalar representation.
        let arr = PlStructArray::from_fields(scalar_fields(1_000_000_000));

        assert_eq!(arr.len(), 1_000_000_000);
        assert_eq!(arr.num_fields(), 2);
        assert_eq!(arr.null_count(), 0);
        assert!(arr.is_valid(999_999_999));
    }

    #[test]
    fn slicing_slices_every_field() {
        let arr = PlStructArray::new(
            flat_fields(),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        )
        .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.field(0).len(), 2);
        assert_eq!(arr.field(1).len(), 2);
        assert_eq!(arr.validity().unwrap().flat_bitmap().unwrap().len(), 2);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(
            arr.field(0)
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .flat_values()
                .unwrap()
                .as_slice(),
            [2, 3],
        );
    }

    #[test]
    fn an_array_of_no_elements_keeps_no_bit() {
        // A single bit is scalar for no elements too, but there is no element left to read it, so
        // it is not kept: the array is flat, like every empty array, rather than scalar.
        let arr = PlStructArray::new_broadcast(scalar_fields(0), 0, Some(Bitmap::new_zeroed(1)));

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.validity().unwrap().is_empty());
    }
}
