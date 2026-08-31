use arrow::bitmap::{Bitmap, and};
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::broadcast::is_valid_buffer_len;

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
/// scalar representation. A struct array is therefore a *scalar* — one row repeated `length` times
/// in `O(1)` memory — exactly when every field [`is_scalar`](PlArray::is_scalar) and its own
/// validity mask is broadcast or absent.
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
/// assert!(scalar.is_scalar());
/// ```
#[derive(Clone)]
pub struct PlStructArray {
    fields: Vec<Box<dyn PlArray>>,
    length: usize,
    validity: Option<Bitmap>,
}

impl PlStructArray {
    /// Creates a [`PlStructArray`] out of its internal components.
    ///
    /// This function is `O(num_fields)`.
    ///
    /// # Errors
    /// This function errors if any field does not have exactly `length` elements, or if `validity`
    /// is neither dense (length equal to `length`) nor broadcast (length one).
    pub fn try_new(
        fields: Vec<Box<dyn PlArray>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        for (i, field) in fields.iter().enumerate() {
            polars_ensure!(
                field.len() == length,
                ComputeError:
                "field {} has {} elements, but the struct array has length {}",
                i, field.len(), length,
            );
        }

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_valid_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is neither dense nor broadcast for an array of length {}",
                validity.len(), length,
            );
        }

        Ok(Self {
            fields,
            length,
            validity,
        })
    }

    /// Creates a [`PlStructArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(fields: Vec<Box<dyn PlArray>>, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(fields, length, validity).unwrap()
    }

    /// Creates a [`PlStructArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// Every field must have exactly `length` elements, and `validity` must be either dense
    /// (length equal to `length`) or broadcast (length one).
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
                    .is_none_or(|v| is_valid_buffer_len(v.len(), length))
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
        Self::new(fields, length, Some(Bitmap::new_zeroed(1)))
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
    /// is dense or broadcast, so reading validity through it needs no knowledge of which
    /// representation this array is in. This mask says nothing about the fields: a valid row may
    /// still hold a null value in any of them.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is dense or broadcast for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_unchecked(validity, self.length) })
    }

    /// Whether every field holds a single value shared by every element.
    ///
    /// A struct array holds no values of its own, so this asks whether every field
    /// [`is_scalar`](PlArray::is_scalar) and the row at every index is therefore the same one. It
    /// is `false` for an array without fields, which has nothing to broadcast, and — like the
    /// fields it defers to — `false` for a dense array of length one, where the two representations
    /// coincide.
    #[inline]
    pub fn values_are_broadcast(&self) -> bool {
        !self.fields.is_empty() && self.fields.iter().all(|field| field.is_scalar())
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_broadcast(&self) -> bool {
        self.validity().is_some_and(|v| v.is_broadcast())
    }

    /// Whether every backing buffer, in this array and in its fields, has one slot per element.
    #[inline]
    pub fn is_dense(&self) -> bool {
        !self.validity_is_broadcast() && self.fields.iter().all(|field| field.is_dense())
    }

    /// Whether this array is entirely stored in the broadcast representation, and therefore is a
    /// single logical row repeated [`Self::len`] times in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_broadcast() && self.validity().is_none_or(|v| v.is_broadcast())
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
    /// This is `O(1)` for a broadcast validity mask and `O(len)` for a dense one, amortized over
    /// repeated calls on the same [`Bitmap`].
    pub fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither dense nor broadcast for this array's length.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither dense nor broadcast for this array's length.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_valid_buffer_len(validity.len(), self.length),
                "validity mask of length {} is neither dense nor broadcast for an array of length {}",
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

        // A broadcast mask is unaffected by slicing: every element reads the same bit.
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

    /// Returns an equivalent array whose backing buffers, fields included, all hold one slot per
    /// element.
    ///
    /// This materializes every broadcast buffer and is therefore `O(len)`; it is a no-op clone when
    /// this array [`is_dense`](Self::is_dense).
    pub fn to_dense(&self) -> Self {
        if self.is_dense() {
            return self.clone();
        }

        Self {
            fields: self
                .fields
                .iter()
                .map(|field| field.to_dense_boxed())
                .collect(),
            length: self.length,
            validity: self.validity().map(|validity| validity.to_dense()),
        }
    }
}

/// Returns `field` with `mask` merged into its validity, so that the undetermined values of rows
/// the struct array masks out are ignored when comparing fields.
///
/// This is only reached for a mask that is neither all-set nor all-unset, which cannot be
/// broadcast, so materializing it costs no more than the mask already does.
fn masked(field: &dyn PlArray, mask: PlBitmapRef<'_>) -> Box<dyn PlArray> {
    let validity = match field.validity() {
        Some(field_validity) => and(&field_validity.to_dense(), &mask.to_dense()),
        None => mask.to_dense(),
    };
    field.with_validity(Some(validity))
}

/// Compares two validity masks over `length` elements, treating an absent mask as all valid.
fn validity_eq(lhs: Option<PlBitmapRef<'_>>, rhs: Option<PlBitmapRef<'_>>, length: usize) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs == rhs,
        (Some(mask), None) | (None, Some(mask)) => mask.set_bits() == length,
        (None, None) => true,
    }
}

impl Default for PlStructArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

/// Compares two arrays row-wise; the representation (dense or broadcast) is irrelevant, and so are
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
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    #[inline]
    fn values_are_broadcast(&self) -> bool {
        self.values_are_broadcast()
    }

    #[inline]
    fn is_dense(&self) -> bool {
        self.is_dense()
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
    fn to_dense_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.to_dense())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlBooleanArray, PlPrimitiveArray};

    fn dense_fields() -> Vec<Box<dyn PlArray>> {
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
    fn dense() {
        let arr = PlStructArray::from_fields(dense_fields());

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.num_fields(), 2);
        assert!(arr.is_dense());
        assert!(!arr.is_scalar());
        assert!(!arr.values_are_broadcast());
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
    fn scalar_defers_to_its_fields() {
        let arr = PlStructArray::from_fields(scalar_fields(1_000_000_000));

        assert_eq!(arr.len(), 1_000_000_000);
        assert!(arr.values_are_broadcast());
        assert!(arr.is_scalar());
        assert!(!arr.is_dense());
        assert_eq!(arr.null_count(), 0);

        // Only every field being scalar makes the row constant.
        let mixed = PlStructArray::new(
            vec![
                Box::new(PlPrimitiveArray::<i32>::new_scalar(1, 3)),
                Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
            ],
            3,
            None,
        );
        assert!(!mixed.values_are_broadcast());
        assert!(!mixed.is_scalar());
        // ... and only every field being dense makes the array dense.
        assert!(!mixed.is_dense());
    }

    #[test]
    fn an_array_without_fields_is_neither_broadcast_nor_scalar() {
        let arr = PlStructArray::new(Vec::new(), 1_000_000_000, None);

        assert_eq!(arr.len(), 1_000_000_000);
        assert_eq!(arr.num_fields(), 0);
        assert!(!arr.values_are_broadcast());
        assert!(!arr.is_scalar());
        assert!(arr.is_dense());
        assert_eq!(arr, arr.clone());
        assert_eq!(arr, arr.to_dense());
        assert_ne!(arr, PlStructArray::new(Vec::new(), 999, None));
    }

    #[test]
    fn null_scalar() {
        let arr = PlStructArray::new_full_null(scalar_fields(1_000_000_000), 1_000_000_000);

        assert!(arr.is_scalar());
        assert!(arr.validity_is_broadcast());
        assert_eq!(arr.validity().unwrap().len(), 1_000_000_000);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 1);
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
    fn dense_fields_with_broadcast_validity() {
        let arr =
            PlStructArray::from_fields(dense_fields()).with_validity(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_broadcast());
        assert!(!arr.values_are_broadcast());
        assert!(!arr.is_dense());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
    }

    #[test]
    fn scalar_fields_with_dense_validity() {
        let arr = PlStructArray::new(
            scalar_fields(3),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        );

        assert!(arr.values_are_broadcast());
        assert!(!arr.validity_is_broadcast());
        assert!(!arr.is_dense());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1));
    }

    #[test]
    fn try_new_rejects_mismatched_lengths() {
        assert!(PlStructArray::try_new(dense_fields(), 2, None).is_err());
        assert!(PlStructArray::try_new(dense_fields(), 3, None).is_ok());
        assert!(PlStructArray::try_new(dense_fields(), 3, Some(Bitmap::new_zeroed(2))).is_err());
        assert!(PlStructArray::try_new(dense_fields(), 3, Some(Bitmap::new_zeroed(1))).is_ok());
        assert!(PlStructArray::try_new(dense_fields(), 3, Some(Bitmap::new_zeroed(3))).is_ok());

        let ragged = vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])) as Box<dyn PlArray>,
            Box::new(PlBooleanArray::from_vec(vec![true, false])),
        ];
        assert!(PlStructArray::try_new(ragged, 3, None).is_err());
    }

    #[test]
    fn slicing_slices_every_field() {
        let arr = PlStructArray::new(
            dense_fields(),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        )
        .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.field(0).len(), 2);
        assert_eq!(arr.field(1).len(), 2);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 2);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(
            arr.field(0)
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .values()
                .as_slice(),
            [2, 3],
        );
    }

    #[test]
    fn slicing_a_scalar_is_free() {
        let arr = PlStructArray::new_full_null(scalar_fields(1_000_000_000), 1_000_000_000)
            .sliced(500, 2);

        assert_eq!(arr.len(), 2);
        assert!(arr.is_scalar());
        assert_eq!(arr.validity().unwrap().bitmap().len(), 1);
        assert_eq!(arr.field(1).len(), 2);
        assert!(arr.field(1).is_scalar());
        assert_eq!(arr.null_count(), 2);
    }

    #[test]
    fn to_dense_materializes_the_fields_too() {
        let scalar = PlStructArray::from_fields(scalar_fields(3))
            .with_validity(Some(Bitmap::new_with_value(true, 1)));
        let dense = scalar.to_dense();

        assert!(dense.is_dense());
        assert!(!dense.is_scalar());
        assert_eq!(dense.len(), 3);
        assert!(dense.field(0).is_dense());
        assert!(dense.field(1).is_dense());
        assert_eq!(dense.validity().unwrap().bitmap().len(), 3);
        assert_eq!(dense, scalar);

        // A dense array is only cloned.
        let arr = PlStructArray::from_fields(dense_fields());
        assert_eq!(arr.to_dense(), arr);
    }

    #[test]
    fn equality_ignores_representation() {
        let scalar = PlStructArray::from_fields(scalar_fields(3));
        let dense = PlStructArray::from_fields(vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 1, 1])),
            Box::new(PlBooleanArray::from_vec(vec![true, true, true])),
        ]);

        assert_eq!(scalar, dense);
        assert_ne!(scalar, PlStructArray::from_fields(scalar_fields(4)));
        assert_ne!(scalar, PlStructArray::from_fields(dense_fields()));

        // An absent mask and an all-set one are the same thing.
        assert_eq!(
            scalar,
            dense
                .clone()
                .with_validity(Some(Bitmap::new_with_value(true, 3))),
        );
        assert_ne!(
            scalar,
            dense.with_validity(Some(Bitmap::from_iter([true, false, true]))),
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
        let lhs = PlStructArray::new(dense_fields(), 3, Some(mask.clone()));
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
        assert!(arr.is_dense());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr, PlStructArray::default());
    }

    #[test]
    fn into_inner_returns_the_components() {
        let arr = PlStructArray::new_full_null(dense_fields(), 3);
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

        let arr = PlStructArray::from_fields(dense_fields());
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
        assert!(arr.is_scalar());
        assert!(!arr.is_dense());
        assert_eq!(arr.null_count(), 0);

        let nulled = arr.with_validity(Some(Bitmap::new_zeroed(1)));
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

        let dense = arr.to_dense_boxed();
        assert!(dense.is_dense());
        assert_eq!(&dense, &arr);
    }
}
