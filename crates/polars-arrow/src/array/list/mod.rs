use super::specification::try_check_offsets_bounds;
use super::{new_empty_array, Array, Splitable};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};
use crate::offset::{Offset, Offsets, OffsetsBuffer};

mod ffi;
pub(super) mod fmt;
mod iterator;
pub use iterator::*;
mod mutable;
pub use mutable::*;
use polars_error::{polars_bail, PolarsResult};
use polars_utils::pl_str::PlSmallStr;

/// An [`Array`] semantically equivalent to `Vec<Option<Vec<Option<T>>>>` with Arrow's in-memory.
#[derive(Clone)]
pub struct ListArray<O: Offset> {
    dtype: ArrowDataType,
    offsets: OffsetsBuffer<O>,
    values: Box<dyn Array>,
    validity: Option<Bitmap>,
}

impl<O: Offset> ListArray<O> {
    /// Creates a new [`ListArray`].
    ///
    /// # Errors
    /// This function returns an error iff:
    /// * `offsets.last()` is greater than `values.len()`.
    /// * the validity's length is not equal to `offsets.len_proxy()`.
    /// * The `dtype`'s [`crate::datatypes::PhysicalType`] is not equal to either [`crate::datatypes::PhysicalType::List`] or [`crate::datatypes::PhysicalType::LargeList`].
    /// * The `dtype`'s inner field's data type is not equal to `values.dtype`.
    /// # Implementation
    /// This function is `O(1)`
    pub fn try_new(
        dtype: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        try_check_offsets_bounds(&offsets, values.len())?;

        if validity
            .as_ref()
            .is_some_and(|validity| validity.len() != offsets.len_proxy())
        {
            polars_bail!(ComputeError: "validity mask length must match the number of values")
        }

        let child_dtype = Self::try_get_child(&dtype)?.dtype();
        let values_dtype = values.dtype();
        if child_dtype != values_dtype {
            polars_bail!(ComputeError: "ListArray's child's DataType must match. However, the expected DataType is {child_dtype:?} while it got {values_dtype:?}.");
        }

        Ok(Self {
            dtype,
            offsets,
            values,
            validity,
        })
    }

    /// Creates a new [`ListArray`].
    ///
    /// # Panics
    /// This function panics iff:
    /// * `offsets.last()` is greater than `values.len()`.
    /// * the validity's length is not equal to `offsets.len_proxy()`.
    /// * The `dtype`'s [`crate::datatypes::PhysicalType`] is not equal to either [`crate::datatypes::PhysicalType::List`] or [`crate::datatypes::PhysicalType::LargeList`].
    /// * The `dtype`'s inner field's data type is not equal to `values.dtype`.
    /// # Implementation
    /// This function is `O(1)`
    pub fn new(
        dtype: ArrowDataType,
        offsets: OffsetsBuffer<O>,
        values: Box<dyn Array>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(dtype, offsets, values, validity).unwrap()
    }

    /// Returns a new empty [`ListArray`].
    pub fn new_empty(dtype: ArrowDataType) -> Self {
        let values = new_empty_array(Self::get_child_type(&dtype).clone());
        Self::new(dtype, OffsetsBuffer::default(), values, None)
    }

    /// Returns a new null [`ListArray`].
    #[inline]
    pub fn new_null(dtype: ArrowDataType, length: usize) -> Self {
        let child = Self::get_child_type(&dtype).clone();
        Self::new(
            dtype,
            Offsets::new_zeroed(length).into(),
            new_empty_array(child),
            Some(Bitmap::new_zeroed(length)),
        )
    }
}

impl<O: Offset> ListArray<O> {
    /// Slices this [`ListArray`].
    /// # Panics
    /// panics iff `offset + length > self.len()`
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this [`ListArray`].
    ///
    /// # Safety
    /// The caller must ensure that `offset + length < self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.offsets.slice_unchecked(offset, length + 1);
    }

    impl_sliced!();
    impl_mut_validity!();
    impl_into_array!();

    pub fn trim_to_normalized_offsets_recursive(&self) -> Self {
        let offsets = self.offsets();
        let values = self.values();

        let first_idx = *offsets.first();
        let len = offsets.range().to_usize();

        let values = if values.len() == len {
            values.clone()
        } else {
            values.sliced(first_idx.to_usize(), len)
        };

        let offsets = if first_idx.to_usize() == 0 {
            offsets.clone()
        } else {
            let v = offsets.iter().map(|x| *x - first_idx).collect::<Vec<_>>();
            unsafe { OffsetsBuffer::<O>::new_unchecked(v.into()) }
        };

        let values = match values.dtype() {
            ArrowDataType::List(_) => {
                let inner: &ListArray<i32> = values.as_ref().as_any().downcast_ref().unwrap();
                Box::new(inner.trim_to_normalized_offsets_recursive()) as Box<dyn Array>
            },
            ArrowDataType::LargeList(_) => {
                let inner: &ListArray<i64> = values.as_ref().as_any().downcast_ref().unwrap();
                Box::new(inner.trim_to_normalized_offsets_recursive()) as Box<dyn Array>
            },
            _ => values,
        };

        assert_eq!(offsets.first().to_usize(), 0);
        assert_eq!(values.len(), offsets.range().to_usize());

        Self::new(
            self.dtype().clone(),
            offsets,
            values,
            self.validity().cloned(),
        )
    }
}

// Accessors
impl<O: Offset> ListArray<O> {
    /// Returns the length of this array
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// Returns the element at index `i`
    /// # Panic
    /// Panics iff `i >= self.len()`
    #[inline]
    pub fn value(&self, i: usize) -> Box<dyn Array> {
        assert!(i < self.len());
        // SAFETY: invariant of this function
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i` as &str
    ///
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> Box<dyn Array> {
        // SAFETY: the invariant of the function
        let (start, end) = self.offsets.start_end_unchecked(i);
        let length = end - start;

        // SAFETY: the invariant of the struct
        self.values.sliced_unchecked(start, length)
    }

    /// The optional validity.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// The offsets [`Buffer`].
    #[inline]
    pub fn offsets(&self) -> &OffsetsBuffer<O> {
        &self.offsets
    }

    /// The values.
    #[inline]
    pub fn values(&self) -> &Box<dyn Array> {
        &self.values
    }
}

impl<O: Offset> ListArray<O> {
    /// Returns a default [`ArrowDataType`]: inner field is named "item" and is nullable
    pub fn default_datatype(dtype: ArrowDataType) -> ArrowDataType {
        let field = Box::new(Field::new(PlSmallStr::from_static("item"), dtype, true));
        if O::IS_LARGE {
            ArrowDataType::LargeList(field)
        } else {
            ArrowDataType::List(field)
        }
    }

    /// Returns a the inner [`Field`]
    /// # Panics
    /// Panics iff the logical type is not consistent with this struct.
    pub fn get_child_field(dtype: &ArrowDataType) -> &Field {
        Self::try_get_child(dtype).unwrap()
    }

    /// Returns a the inner [`Field`]
    /// # Errors
    /// Panics iff the logical type is not consistent with this struct.
    pub fn try_get_child(dtype: &ArrowDataType) -> PolarsResult<&Field> {
        if O::IS_LARGE {
            match dtype.to_logical_type() {
                ArrowDataType::LargeList(child) => Ok(child.as_ref()),
                _ => polars_bail!(ComputeError: "ListArray<i64> expects DataType::LargeList"),
            }
        } else {
            match dtype.to_logical_type() {
                ArrowDataType::List(child) => Ok(child.as_ref()),
                _ => polars_bail!(ComputeError: "ListArray<i32> expects DataType::List"),
            }
        }
    }

    /// Returns a the inner [`ArrowDataType`]
    /// # Panics
    /// Panics iff the logical type is not consistent with this struct.
    pub fn get_child_type(dtype: &ArrowDataType) -> &ArrowDataType {
        Self::get_child_field(dtype).dtype()
    }
}

impl<O: Offset> Array for ListArray<O> {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl<O: Offset> Splitable for ListArray<O> {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_offsets, rhs_offsets) = unsafe { self.offsets.split_at_unchecked(offset) };
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };

        (
            Self {
                dtype: self.dtype.clone(),
                offsets: lhs_offsets,
                validity: lhs_validity,
                values: self.values.clone(),
            },
            Self {
                dtype: self.dtype.clone(),
                offsets: rhs_offsets,
                validity: rhs_validity,
                values: self.values.clone(),
            },
        )
    }
}
