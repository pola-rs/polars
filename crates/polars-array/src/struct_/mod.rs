use arrow::bitmap::{Bitmap, and};
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::{is_flat_buffer_len, is_scalar_buffer_len, is_valid_buffer_len};
use crate::flat::Flat;

mod builder;

pub use builder::PlStructArrayBuilder;

/// An immutable, cheaply cloneable sequence of `length` optional rows, one value per field array.
///
/// This is the nested array of this crate: it holds no values of its own, only a validity mask
/// over a list of field arrays that all have `length` elements. It carries no logical type — the
/// fields are positional and unnamed, so the names and types a caller thinks of as part of a
/// struct live at a higher level.
///
/// The logical length is stored separately from the validity mask, which is what lets a fully null
/// array be represented in `O(1)` memory: the mask is a single shared bit, read through
/// [`broadcast_index(i, bitmap.len())`](crate::broadcast::broadcast_index) like every other backing
/// buffer in this crate. See [`crate::broadcast`] for the full rules.
///
/// The fields need no such treatment here — each is a [`PlArray`] that already carries its own
/// scalar representation. A struct array is therefore a single row repeated `length` times in
/// `O(1)` memory exactly when every field is scalar and its own validity mask is scalar or absent;
/// whether that is so is a question for the concrete field arrays, not for this one.
///
/// # Example
/// ```
/// use polars_array::{PlArray, PlBooleanArray, PlPrimitiveArray, PlStructArray};
///
/// let arr = PlStructArray::from_fields(vec![
///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
///     Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
/// ]);
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.num_fields(), 2);
/// assert_eq!(arr.null_count(), 0);
///
/// // A row repeated a billion times costs `O(1)` memory, fields included.
/// let scalar = PlStructArray::from_fields(vec![
///     Box::new(PlPrimitiveArray::<i32>::new_scalar(1, 1_000_000_000)),
///     Box::new(PlBooleanArray::new_scalar(true, 1_000_000_000)),
/// ]);
/// assert_eq!(scalar.len(), 1_000_000_000);
/// assert_eq!(scalar.null_count(), 0);
/// ```
#[derive(Clone)]
pub struct PlStructArray {
    fields: Vec<Box<dyn PlArray>>,
    length: usize,
    validity: Option<Bitmap>,
}

impl PlStructArray {
    /// Creates a flat [`PlStructArray`] out of its internal components.
    ///
    /// The validity mask has to hold one bit per element. [`Self::try_new_broadcast`] is what
    /// builds the scalar one; this function never infers it from a mask that happens to hold a
    /// single bit. The fields are the same either way — a struct array never broadcasts them, and
    /// a field that stands for one repeated value is a scalar array of `length` elements in its
    /// own right. This function is `O(num_fields)`.
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
    /// The mask has to hold the single bit every element shares, which is what makes a struct
    /// array of nothing but nulls `O(1)`. The fields are the same as [`Self::try_new`] asks for:
    /// a struct array never broadcasts them, so this is the only backing buffer the two families
    /// differ over. This function is `O(num_fields)`.
    ///
    /// # Errors
    /// This function errors if any field does not have exactly `length` elements, or if `validity`
    /// does not hold exactly one bit. An array of no elements reads no bit at all, so it
    /// additionally admits an empty mask.
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
            validity,
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
    /// Every field must have exactly `length` elements, and `validity` must hold exactly one bit,
    /// or none at all if `length` is zero.
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
            validity,
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
    /// This function is `O(num_fields)`.
    ///
    /// # Panics
    /// Panics if `fields` is empty — an array without fields has no length to take — or if the
    /// fields do not all have the same length. Use [`Self::new`] to build either.
    pub fn from_fields(fields: Vec<Box<dyn PlArray>>) -> Self {
        let length = fields
            .first()
            .expect("cannot infer the length of a struct array without fields")
            .len();
        Self::new(fields, length, None)
    }

    /// Creates a [`PlStructArray`] of `length` nulls over `fields`, in `O(1)` extra memory.
    ///
    /// The fields are kept as they are: every row is null, so their values are undetermined, but
    /// they still have to hold `length` elements each. Pass scalar fields to keep the whole array
    /// `O(1)`.
    ///
    /// # Panics
    /// Panics if any field does not have exactly `length` elements.
    #[inline]
    pub fn new_full_null(fields: Vec<Box<dyn PlArray>>, length: usize) -> Self {
        Self::new_broadcast(fields, length, Some(Bitmap::new_zeroed(1)))
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
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing bitmap
    /// is flat or scalar, so reading validity through it needs no knowledge of which
    /// representation this array is in. This mask says nothing about the fields: a valid row may
    /// still hold a null value in any of them.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_unchecked(validity, self.length) })
    }

    /// Whether the validity mask holds a single bit shared by every element.
    ///
    /// This says nothing about the fields, which carry their own representation.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// A valid row may still hold a null value in any of its fields.
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
    /// Null values inside the fields do not count: only rows this array itself masks out are null.
    ///
    /// This is `O(1)` for a scalar validity mask and `O(len)` for a flat one, amortized over
    /// repeated calls on the same [`Bitmap`].
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
    /// [`Self::with_validity_broadcast`] is what installs the single bit every element shares;
    /// this function never infers that from a mask that happens to hold one bit.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask with a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    /// [`Self::set_validity_broadcast`] is what installs the single bit every element shares;
    /// this function never infers that from a mask that happens to hold one bit.
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
    /// This is [`Self::set_validity`] widened to the scalar representation: the mask is either
    /// flat — one bit per element — or the single bit every element shares. See
    /// [`crate::broadcast`].
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
        self.validity = validity;
    }

    /// Drops the validity mask, making every row valid.
    ///
    /// The fields keep their own validity: a row that is valid may still hold null field values.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.validity = None;
        self
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// This function is `O(num_fields)`.
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
    /// This function is `O(num_fields)`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // Each field slices itself, keeping whichever representation it is in.
        for field in self.fields.iter_mut() {
            unsafe { field.slice_unchecked(offset, length) };
        }

        // A scalar mask is unaffected by slicing: every element reads the same bit.
        if let Some(validity) = self.validity.as_mut() {
            if validity.len() == self.length {
                unsafe { validity.slice_unchecked(offset, length) };
            }
        }

        self.length = length;
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// This function is `O(num_fields)`.
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
    /// This function is `O(num_fields)`.
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
    /// Every field repeats its own element, so this is `O(num_fields)` and the result is one row in
    /// `O(1)` memory. A null row repeats as `length` nulls.
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
    /// This function is `O(num_fields)`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        let validity = unsafe { self.is_null_unchecked(index) }.then(|| Bitmap::new_zeroed(1));

        // The field values of a null row are undetermined, so they are repeated as they are found:
        // it is the mask that makes every row of the result null.
        let fields = self
            .fields
            .iter()
            .map(|field| unsafe {
                if validity.is_some() {
                    field
                        .with_validity_broadcast(validity.clone())
                        .new_from_index_unchecked(index, length)
                } else {
                    field.new_from_index_unchecked(index, length)
                }
            })
            .collect();

        // SAFETY: every field repeated one element `length` times, so it holds `length` elements,
        // and the mask is a single bit and therefore scalar.
        unsafe { Self::new_broadcast_unchecked(fields, length, validity) }
    }

    /// Whether every backing buffer of this array holds one slot per element.
    ///
    /// A struct array never broadcasts its fields — each holds one element per row already — so
    /// the only buffer of its own is the validity mask, and this is
    /// [`validity_is_scalar`](Self::validity_is_scalar) answered the other way round. Whether a
    /// *field* is flat is a question for that field.
    #[inline]
    pub fn is_flat(&self) -> bool {
        !self.validity_is_scalar()
    }

    /// Whether this array is a single row repeated over its length, in `O(1)` memory.
    ///
    /// A struct array has no buffer of its own but the validity mask, so this asks that the mask
    /// be scalar or absent *and* that every field be scalar in turn: a row is the values its
    /// fields hold at that index, and it only repeats if each of them does. An array of no fields
    /// is a length and a mask, and is scalar whenever that mask is.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.validity().is_none_or(|validity| validity.is_scalar())
            && self.fields.iter().all(|field| field.is_scalar())
    }

    /// Returns this array in the flat representation, writing out a scalar validity mask.
    ///
    /// This is `O(1)` for an array that is already flat and `O(len)` for one whose mask is
    /// scalar; the fields are handed over as they are — see [`PlStructArray::is_flat`].
    #[must_use]
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            // SAFETY: just checked.
            return unsafe { Flat::new(self.clone()) };
        }

        let validity = self.validity().map(|validity| validity.to_flat());

        // SAFETY: the fields are untouched and still hold `length` elements each, and the mask was
        // just written out to one bit per element.
        let array = unsafe { Self::new_unchecked(self.fields.clone(), self.length, validity) };

        // SAFETY: the mask is flat, and a struct array has no other buffer of its own.
        unsafe { Flat::new(array) }
    }

    /// Borrows this array as a flat one, or `None` if its validity mask is scalar.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: `is_flat` is exactly the invariant of `Flat`.
        self.is_flat()
            .then(|| unsafe { Flat::new_ref(self) })
    }
}

/// Returns `field` with `mask` merged into its validity, so that the undetermined values of rows
/// the struct array masks out are ignored when comparing fields.
///
/// This is only reached for a mask that is neither all-set nor all-unset, which cannot be
/// scalar, so materializing it costs no more than the mask already does.
fn masked(field: &dyn PlArray, mask: PlBitmapRef<'_>) -> Box<dyn PlArray> {
    let validity = match field.validity() {
        Some(field_validity) => and(&field_validity.to_flat(), &mask.to_flat()),
        None => mask.to_flat(),
    };
    field.with_validity(Some(validity))
}

impl Default for PlStructArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

/// Compares two arrays row-wise; the representation (flat or scalar) is irrelevant, and so are
/// the field values of null rows.
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
///
/// This is the half of the validation that both families of constructors share: a struct array
/// never broadcasts its fields, so only its validity mask tells the representations apart.
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
    fn an_array_without_fields_has_a_length_of_its_own() {
        let arr = PlStructArray::new(Vec::new(), 1_000_000_000, None);

        assert_eq!(arr.len(), 1_000_000_000);
        assert_eq!(arr.num_fields(), 0);
        assert_eq!(arr, arr.clone());
        assert_ne!(arr, PlStructArray::new(Vec::new(), 999, None));
    }

    #[test]
    fn null_scalar() {
        let arr = PlStructArray::new_full_null(scalar_fields(1_000_000_000), 1_000_000_000);

        assert!(arr.validity_is_scalar());
        assert_eq!(arr.validity().unwrap().len(), 1_000_000_000);
        assert_eq!(arr.null_count(), 1_000_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999_999));

        // The fields are untouched: it is the struct array that masks the rows out.
        assert_eq!(arr.field(0).null_count(), 0);

        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 0);
        assert!(valid.validity().is_none());
    }

    #[test]
    fn flat_fields_with_scalar_validity() {
        let arr = PlStructArray::from_fields(flat_fields())
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_scalar());
        assert_eq!(arr.null_count(), 3);
    }

    #[test]
    fn scalar_fields_with_flat_validity() {
        let arr = PlStructArray::new(
            scalar_fields(3),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        );

        assert!(!arr.validity_is_scalar());
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1));
    }

    #[test]
    fn try_new_rejects_mismatched_lengths() {
        assert!(PlStructArray::try_new(flat_fields(), 2, None).is_err());
        assert!(PlStructArray::try_new(flat_fields(), 3, None).is_ok());

        // The fields are the same either way: a struct array never broadcasts them.
        assert!(PlStructArray::try_new_broadcast(flat_fields(), 2, None).is_err());
        assert!(PlStructArray::try_new_broadcast(flat_fields(), 3, None).is_ok());

        // Only the mask tells the two families apart.
        assert!(PlStructArray::try_new(flat_fields(), 3, Some(Bitmap::new_zeroed(2))).is_err());
        assert!(PlStructArray::try_new(flat_fields(), 3, Some(Bitmap::new_zeroed(1))).is_err());
        assert!(PlStructArray::try_new(flat_fields(), 3, Some(Bitmap::new_zeroed(3))).is_ok());
        assert!(
            PlStructArray::try_new_broadcast(flat_fields(), 3, Some(Bitmap::new_zeroed(3)))
                .is_err()
        );
        assert!(
            PlStructArray::try_new_broadcast(flat_fields(), 3, Some(Bitmap::new_zeroed(1))).is_ok()
        );

        let ragged = vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])) as Box<dyn PlArray>,
            Box::new(PlBooleanArray::from_vec(vec![true, false])),
        ];
        assert!(PlStructArray::try_new(ragged, 3, None).is_err());
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
    fn slicing_a_scalar_is_free() {
        let arr = PlStructArray::new_full_null(scalar_fields(1_000_000_000), 1_000_000_000)
            .sliced(500, 2);

        assert_eq!(arr.len(), 2);
        assert!(arr.validity().unwrap().is_scalar());
        assert_eq!(arr.field(1).len(), 2);
        assert_eq!(arr.null_count(), 2);
    }

    #[test]
    fn equality_ignores_representation() {
        let scalar = PlStructArray::from_fields(scalar_fields(3));
        let flat = PlStructArray::from_fields(vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 1, 1])),
            Box::new(PlBooleanArray::from_vec(vec![true, true, true])),
        ]);

        assert_eq!(scalar, flat);
        assert_ne!(scalar, PlStructArray::from_fields(scalar_fields(4)));
        assert_ne!(scalar, PlStructArray::from_fields(flat_fields()));

        // An absent mask and an all-set one are the same thing.
        assert_eq!(
            scalar,
            flat.clone()
                .with_validity(Some(Bitmap::new_with_value(true, 3))),
        );
        assert_ne!(
            scalar,
            flat.with_validity(Some(Bitmap::from_iter([true, false, true]))),
        );

        // The fields are positional, and a missing one is not the same array.
        assert_ne!(
            scalar,
            PlStructArray::from_fields(vec![Box::new(PlPrimitiveArray::<i32>::new_scalar(1, 3))]),
        );
    }

    #[test]
    fn equality_ignores_the_field_values_of_null_rows() {
        let mask = Bitmap::from_iter([true, false, true]);
        let lhs = PlStructArray::new(flat_fields(), 3, Some(mask.clone()));
        let rhs = PlStructArray::new(
            vec![
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 42, 3])),
                Box::new(PlBooleanArray::from_vec(vec![true, true, true])),
            ],
            3,
            Some(mask.clone()),
        );

        // The rows differ only where both arrays are null, so their values are undetermined.
        assert_eq!(lhs, rhs);

        // A null field value inside a valid row still counts.
        let with_null_field = PlStructArray::new(
            vec![
                Box::new(PlPrimitiveArray::from_iter([Some(1i32), Some(2), None]))
                    as Box<dyn PlArray>,
                Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
            ],
            3,
            Some(mask),
        );
        assert_ne!(lhs, with_null_field);
    }

    #[test]
    fn equality_of_scalars_does_not_walk_rows() {
        // Row-by-row comparison of a billion rows would not finish; the field arrays compare
        // themselves in `O(1)`.
        let arr = PlStructArray::from_fields(scalar_fields(1_000_000_000));
        let null = PlStructArray::new_full_null(scalar_fields(1_000_000_000), 1_000_000_000);

        assert_eq!(arr, arr.clone());
        assert_eq!(null, null.clone());
        assert_ne!(arr, null);

        // Two fully null arrays are equal whatever their fields hold.
        assert_eq!(
            null,
            PlStructArray::new_full_null(
                vec![
                    Box::new(PlPrimitiveArray::<i32>::new_scalar(42, 1_000_000_000)),
                    Box::new(PlBooleanArray::new_scalar(false, 1_000_000_000)),
                ],
                1_000_000_000,
            ),
        );
    }

    #[test]
    fn empty() {
        let arr = PlStructArray::new_empty();

        assert!(arr.is_empty());
        assert_eq!(arr.num_fields(), 0);
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr, PlStructArray::default());
    }

    #[test]
    fn into_inner_returns_the_components() {
        let arr = PlStructArray::new_full_null(flat_fields(), 3);
        let (fields, length, validity) = arr.into_inner();

        assert_eq!(fields.len(), 2);
        assert_eq!(length, 3);
        assert_eq!(validity, Some(Bitmap::new_zeroed(1)));
    }

    #[test]
    fn debug_does_not_materialize_scalars() {
        let arr = PlStructArray::new_full_null(scalar_fields(1_000_000_000), 1_000_000_000);
        assert_eq!(
            format!("{arr:?}"),
            "PlStructArray { length: 1000000000, validity: PlBitmapRef[false; 1000000000], \
             fields: [PlPrimitiveArray[1; 1000000000], PlBooleanArray[true; 1000000000]] }",
        );

        let arr = PlStructArray::from_fields(flat_fields());
        assert_eq!(
            format!("{arr:?}"),
            "PlStructArray { length: 3, fields: [PlPrimitiveArray[1, 2, 3], \
             PlBooleanArray[true, false, true]] }",
        );
    }

    #[test]
    fn behind_the_trait_object() {
        let arr: Box<dyn PlArray> = Box::new(PlStructArray::from_fields(scalar_fields(1_000)));

        assert_eq!(arr.array_type(), PlArrayType::Struct);
        assert!(arr.array_type().is_struct());
        assert_eq!(arr.len(), 1_000);
        assert_eq!(arr.null_count(), 0);

        let nulled = arr.with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert_eq!(nulled.null_count(), 1_000);
        assert_eq!(arr.null_count(), 0);

        let sliced = arr.sliced(500, 2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(
            sliced
                .as_any()
                .downcast_ref::<PlStructArray>()
                .unwrap()
                .field(0)
                .len(),
            2,
        );
    }

    #[test]
    fn new_from_index_repeats_one_row() {
        let arr = PlStructArray::from_fields(flat_fields());

        // Every field repeats its own element, so a billion rows cost `O(1)`.
        let repeated = arr.new_from_index(2, 1_000_000_000);
        assert_eq!(repeated.len(), 1_000_000_000);
        assert_eq!(repeated.num_fields(), 2);
        assert_eq!(repeated.null_count(), 0);
        assert_eq!(
            repeated,
            PlStructArray::from_fields(vec![
                Box::new(PlPrimitiveArray::<i32>::new_scalar(3, 1_000_000_000)),
                Box::new(PlBooleanArray::new_scalar(true, 1_000_000_000)),
            ]),
        );

        // A null row repeats as nulls, under a mask of a single bit; the fields still hold one
        // element per row, undetermined as their values are.
        let nulls = PlStructArray::new_full_null(flat_fields(), 3).new_from_index(0, 4);
        assert_eq!(nulls.null_count(), 4);
        assert!(nulls.validity_is_scalar());
        assert_eq!(nulls.field(0).len(), 4);

        assert!(arr.new_from_index(0, 0).is_empty());
        assert_eq!(
            unsafe { arr.new_from_index_unchecked(0, 2) }.field(0).len(),
            2,
        );
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn repeating_a_row_out_of_bounds_panics() {
        let _ = PlStructArray::from_fields(flat_fields()).new_from_index(3, 1);
    }
}
