use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure};

use super::{Array, Splitable, new_empty_array, new_null_array};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};

mod builder;
pub use builder::*;
pub(super) mod fmt;

/// The largest number of variants a union can have.
pub const MAX_UNION_VARIANTS: usize = i8::MAX as usize;

/// A dense union with `i64` offsets and its own validity, Polars' in-memory form of a union.
///
/// Slot `i` holds `fields()[types()[i]]` at position `offsets()[i]`.
///
/// Invariants:
/// 1. Every slot, null or not, has an in-bounds `(type, offset)`.
/// 2. Offsets are strictly increasing per child.
/// 3. A non-null slot points at a non-null child value.
#[derive(Clone)]
pub struct LargeUnionArray {
    dtype: ArrowDataType,
    types: Buffer<i8>,
    offsets: Buffer<i64>,
    fields: Vec<Box<dyn Array>>,
    validity: Option<Bitmap>,
}

impl LargeUnionArray {
    /// Returns a new [`LargeUnionArray`], checking invariants 1 and 2.
    pub fn try_new(
        dtype: ArrowDataType,
        types: Buffer<i8>,
        offsets: Buffer<i64>,
        fields: Vec<Box<dyn Array>>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        let dtype_fields = Self::try_get_fields(&dtype)?;
        polars_ensure!(
            !dtype_fields.is_empty() && dtype_fields.len() <= MAX_UNION_VARIANTS,
            ComputeError: "a union must have between 1 and {MAX_UNION_VARIANTS} variants"
        );
        polars_ensure!(
            dtype_fields.len() == fields.len(),
            ComputeError: "the number of union children must equal the number of variants"
        );
        for (i, (f, c)) in dtype_fields.iter().zip(&fields).enumerate() {
            polars_ensure!(
                f.dtype() == c.dtype(),
                ComputeError: "union variant {i} has dtype {:?} but its child has dtype {:?}",
                f.dtype(), c.dtype()
            );
        }
        polars_ensure!(
            offsets.len() == types.len(),
            ComputeError: "the union offsets must have the same length as the types"
        );
        if let Some(v) = &validity {
            polars_ensure!(
                v.len() == types.len(),
                ComputeError: "the union validity must have the same length as the types"
            );
        }

        let mut next = vec![0i64; fields.len()];
        for (&t, &o) in types.iter().zip(offsets.iter()) {
            let Some(child) = usize::try_from(t).ok().filter(|t| *t < fields.len()) else {
                polars_bail!(ComputeError: "union type id {t} is out of range");
            };
            if o < next[child] || o as u64 >= fields[child].len() as u64 {
                polars_bail!(
                    ComputeError: "union offset {o} of variant {child} is out of bounds or not strictly increasing"
                );
            }
            next[child] = o + 1;
        }

        Ok(Self {
            dtype,
            types,
            offsets,
            fields,
            validity,
        })
    }

    /// Returns a new [`LargeUnionArray`].
    ///
    /// # Panics
    /// Panics if [`Self::try_new`] would fail.
    pub fn new(
        dtype: ArrowDataType,
        types: Buffer<i8>,
        offsets: Buffer<i64>,
        fields: Vec<Box<dyn Array>>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(dtype, types, offsets, fields, validity).unwrap()
    }

    /// Returns a new [`LargeUnionArray`] without checking the invariants.
    ///
    /// # Safety
    /// The arguments must satisfy the checks of [`Self::try_new`].
    pub unsafe fn new_unchecked(
        dtype: ArrowDataType,
        types: Buffer<i8>,
        offsets: Buffer<i64>,
        fields: Vec<Box<dyn Array>>,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            return Self::new(dtype, types, offsets, fields, validity);
        }
        Self {
            dtype,
            types,
            offsets,
            fields,
            validity,
        }
    }

    /// Creates an empty [`LargeUnionArray`].
    pub fn new_empty(dtype: ArrowDataType) -> Self {
        let fields = Self::get_fields(&dtype)
            .iter()
            .map(|f| new_empty_array(f.dtype().clone()))
            .collect();
        Self::new(dtype, Buffer::new(), Buffer::new(), fields, None)
    }

    /// Creates an all-null [`LargeUnionArray`]; child 0 holds a null placeholder per slot.
    pub fn new_null(dtype: ArrowDataType, length: usize) -> Self {
        let fields = Self::get_fields(&dtype)
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if i == 0 {
                    new_null_array(f.dtype().clone(), length)
                } else {
                    new_empty_array(f.dtype().clone())
                }
            })
            .collect();
        let types = Buffer::from(vec![0i8; length]);
        let offsets = Buffer::from_iter(0..length as i64);
        unsafe {
            Self::new_unchecked(
                dtype,
                types,
                offsets,
                fields,
                Some(Bitmap::new_zeroed(length)),
            )
        }
    }

    /// Deconstructs the array into its type ids, offsets, children and validity.
    #[allow(clippy::type_complexity)]
    pub fn into_inner(
        self,
    ) -> (
        ArrowDataType,
        Buffer<i8>,
        Buffer<i64>,
        Vec<Box<dyn Array>>,
        Option<Bitmap>,
    ) {
        (
            self.dtype,
            self.types,
            self.offsets,
            self.fields,
            self.validity,
        )
    }

    /// Slices this array. Children are shared and not sliced.
    ///
    /// # Panics
    /// Panics iff `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "offset + length may not exceed length of array"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this array. Children are shared and not sliced.
    ///
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        unsafe {
            self.types.slice_in_place_unchecked(offset..offset + length);
            self.offsets
                .slice_in_place_unchecked(offset..offset + length);
        }
        self.validity = self
            .validity
            .take()
            .map(|bitmap| unsafe { bitmap.sliced_unchecked(offset, length) })
            .filter(|bitmap| bitmap.unset_bits() > 0);
    }

    impl_sliced!();
    impl_mut_validity!();
    impl_into_array!();
}

impl LargeUnionArray {
    #[inline]
    pub fn len(&self) -> usize {
        self.types.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The variant index of each slot.
    pub fn types(&self) -> &Buffer<i8> {
        &self.types
    }

    /// The position of each slot in its variant's child.
    pub fn offsets(&self) -> &Buffer<i64> {
        &self.offsets
    }

    /// One child per variant.
    pub fn fields(&self) -> &[Box<dyn Array>] {
        &self.fields
    }

    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Returns the variant index and child position of slot `index`.
    ///
    /// # Panics
    /// Panics iff `index >= self.len()`.
    #[inline]
    pub fn index(&self, index: usize) -> (usize, usize) {
        (self.types[index] as usize, self.offsets[index] as usize)
    }

    /// Returns the variant index and child position of slot `index`.
    ///
    /// # Safety
    /// `index < self.len()`.
    #[inline]
    pub unsafe fn index_unchecked(&self, index: usize) -> (usize, usize) {
        unsafe {
            (
                *self.types.get_unchecked(index) as usize,
                *self.offsets.get_unchecked(index) as usize,
            )
        }
    }

    pub fn variant_fields(&self) -> &[Field] {
        Self::get_fields(&self.dtype)
    }

    pub(crate) fn try_get_fields(dtype: &ArrowDataType) -> PolarsResult<&[Field]> {
        match dtype.to_storage() {
            ArrowDataType::LargeUnion(fields) => Ok(fields),
            _ => polars_bail!(ComputeError: "LargeUnionArray requires ArrowDataType::LargeUnion"),
        }
    }

    /// Returns the variants of an [`ArrowDataType::LargeUnion`].
    pub fn get_fields(dtype: &ArrowDataType) -> &[Field] {
        Self::try_get_fields(dtype).unwrap()
    }
}

impl Array for LargeUnionArray {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Swaps the validity without touching the children. Only narrow it: marking a null slot
    /// valid breaks invariant 3 unless its child value is known to be valid.
    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl Splitable for LargeUnionArray {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_types, rhs_types) = unsafe { self.types.split_at_unchecked(offset) };
        let (lhs_offsets, rhs_offsets) = unsafe { self.offsets.split_at_unchecked(offset) };
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };
        (
            Self {
                dtype: self.dtype.clone(),
                types: lhs_types,
                offsets: lhs_offsets,
                fields: self.fields.clone(),
                validity: lhs_validity,
            },
            Self {
                dtype: self.dtype.clone(),
                types: rhs_types,
                offsets: rhs_offsets,
                fields: self.fields.clone(),
                validity: rhs_validity,
            },
        )
    }
}
