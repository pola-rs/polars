use std::any::Any;

use polars_error::{polars_bail, PolarsResult};

use super::Splitable;
use crate::array::{Array, FromFfi, MutableArray, ToFfi};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::{ArrowDataType, PhysicalType};
use crate::ffi;

/// The concrete [`Array`] of [`ArrowDataType::Null`].
#[derive(Clone)]
pub struct NullArray {
    data_type: ArrowDataType,

    /// Validity mask. This is always all-zeroes.
    validity: Bitmap,

    length: usize,
}

impl NullArray {
    /// Returns a new [`NullArray`].
    /// # Errors
    /// This function errors iff:
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to [`crate::datatypes::PhysicalType::Null`].
    pub fn try_new(data_type: ArrowDataType, length: usize) -> PolarsResult<Self> {
        if data_type.to_physical_type() != PhysicalType::Null {
            polars_bail!(ComputeError: "NullArray can only be initialized with a DataType whose physical type is Null");
        }

        let validity = Bitmap::new_zeroed(length);

        Ok(Self {
            data_type,
            validity,
            length,
        })
    }

    /// Returns a new [`NullArray`].
    /// # Panics
    /// This function errors iff:
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to [`crate::datatypes::PhysicalType::Null`].
    pub fn new(data_type: ArrowDataType, length: usize) -> Self {
        Self::try_new(data_type, length).unwrap()
    }

    /// Returns a new empty [`NullArray`].
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        Self::new(data_type, 0)
    }

    /// Returns a new [`NullArray`].
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        Self::new(data_type, length)
    }

    impl_sliced!();
    impl_into_array!();
}

impl NullArray {
    /// Returns a slice of the [`NullArray`].
    /// # Panic
    /// This function panics iff `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new array cannot exceed the arrays' length"
        );
        unsafe { self.slice_unchecked(offset, length) };
    }

    /// Returns a slice of the [`NullArray`].
    ///
    /// # Safety
    /// The caller must ensure that `offset + length < self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.length = length;
        self.validity.slice_unchecked(offset, length);
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl Array for NullArray {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        Some(&self.validity)
    }

    fn with_validity(&self, _: Option<Bitmap>) -> Box<dyn Array> {
        // Nulls with invalid nulls are also nulls.
        self.clone().boxed()
    }
}

#[derive(Debug)]
/// A distinct type to disambiguate
/// clashing methods
pub struct MutableNullArray {
    inner: NullArray,
}

impl MutableNullArray {
    /// Returns a new [`MutableNullArray`].
    /// # Panics
    /// This function errors iff:
    /// * The `data_type`'s [`crate::datatypes::PhysicalType`] is not equal to [`crate::datatypes::PhysicalType::Null`].
    pub fn new(data_type: ArrowDataType, length: usize) -> Self {
        let inner = NullArray::try_new(data_type, length).unwrap();
        Self { inner }
    }
}

impl From<MutableNullArray> for NullArray {
    fn from(value: MutableNullArray) -> Self {
        value.inner
    }
}

impl MutableArray for MutableNullArray {
    fn data_type(&self) -> &ArrowDataType {
        &ArrowDataType::Null
    }

    fn len(&self) -> usize {
        self.inner.length
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        self.inner.clone().boxed()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn push_null(&mut self) {
        self.inner.length += 1;
    }

    fn reserve(&mut self, _additional: usize) {
        // no-op
    }

    fn shrink_to_fit(&mut self) {
        // no-op
    }
}

impl std::fmt::Debug for NullArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NullArray({})", self.len())
    }
}

unsafe impl ToFfi for NullArray {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        // `None` is technically not required by the specification, but older C++ implementations require it, so leaving
        // it here for backward compatibility
        vec![None]
    }

    fn offset(&self) -> Option<usize> {
        Some(0)
    }

    fn to_ffi_aligned(&self) -> Self {
        self.clone()
    }
}

impl Splitable for NullArray {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs, rhs) = self.validity.split_at(offset);

        (
            Self {
                data_type: self.data_type.clone(),
                validity: lhs,
                length: offset,
            },
            Self {
                data_type: self.data_type.clone(),
                validity: rhs,
                length: self.len() - offset,
            },
        )
    }
}

impl<A: ffi::ArrowArrayRef> FromFfi<A> for NullArray {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let data_type = array.data_type().clone();
        Self::try_new(data_type, array.array().len())
    }
}

#[cfg(feature = "arrow_rs")]
mod arrow {
    use arrow_data::{ArrayData, ArrayDataBuilder};

    use super::*;
    impl NullArray {
        /// Convert this array into [`arrow_data::ArrayData`]
        pub fn to_data(&self) -> ArrayData {
            let builder = ArrayDataBuilder::new(arrow_schema::DataType::Null).len(self.len());

            // SAFETY: safe by construction
            unsafe { builder.build_unchecked() }
        }

        /// Create this array from [`ArrayData`]
        pub fn from_data(data: &ArrayData) -> Self {
            Self::new(ArrowDataType::Null, data.len())
        }
    }
}
