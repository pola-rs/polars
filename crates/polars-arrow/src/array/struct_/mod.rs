use super::{new_empty_array, new_null_array, Array, Splitable};
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};

#[cfg(feature = "arrow_rs")]
mod data;
mod ffi;
pub(super) mod fmt;
mod iterator;
mod mutable;
pub use mutable::*;
use polars_error::{polars_bail, PolarsResult};

use crate::compute::utils::combine_validities_and;

/// A [`StructArray`] is a nested [`Array`] with an optional validity representing
/// multiple [`Array`] with the same number of rows.
/// # Example
/// ```
/// use polars_arrow::array::*;
/// use polars_arrow::datatypes::*;
/// let boolean = BooleanArray::from_slice(&[false, false, true, true]).boxed();
/// let int = Int32Array::from_slice(&[42, 28, 19, 31]).boxed();
///
/// let fields = vec![
///     Field::new("b".into(), ArrowDataType::Boolean, false),
///     Field::new("c".into(), ArrowDataType::Int32, false),
/// ];
///
/// let array = StructArray::new(ArrowDataType::Struct(fields), vec![boolean, int], None);
/// ```
#[derive(Clone)]
pub struct StructArray {
    dtype: ArrowDataType,
    // invariant: each array has the same length
    values: Vec<Box<dyn Array>>,
    validity: Option<Bitmap>,
}

impl StructArray {
    /// Returns a new [`StructArray`].
    /// # Errors
    /// This function errors iff:
    /// * `dtype`'s physical type is not [`crate::datatypes::PhysicalType::Struct`].
    /// * the children of `dtype` are empty
    /// * the values's len is different from children's length
    /// * any of the values's data type is different from its corresponding children' data type
    /// * any element of values has a different length than the first element
    /// * the validity's length is not equal to the length of the first element
    pub fn try_new(
        dtype: ArrowDataType,
        values: Vec<Box<dyn Array>>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        let fields = Self::try_get_fields(&dtype)?;
        if fields.is_empty() {
            assert!(values.is_empty(), "invalid struct");
            assert_eq!(validity.map(|v| v.len()).unwrap_or(0), 0, "invalid struct");
            return Ok(Self {
                dtype,
                values,
                validity: None,
            });
        }
        if fields.len() != values.len() {
            polars_bail!(ComputeError:"a StructArray must have a number of fields in its DataType equal to the number of child values")
        }

        fields
            .iter().map(|a| &a.dtype)
            .zip(values.iter().map(|a| a.dtype()))
            .enumerate()
            .try_for_each(|(index, (dtype, child))| {
                if dtype != child {
                    polars_bail!(ComputeError:
                        "The children DataTypes of a StructArray must equal the children data types.
                         However, the field {index} has data type {dtype:?} but the value has data type {child:?}"
                    )
                } else {
                    Ok(())
                }
            })?;

        let len = values[0].len();
        values
            .iter()
            .map(|a| a.len())
            .enumerate()
            .try_for_each(|(index, a_len)| {
                if a_len != len {
                    polars_bail!(ComputeError: "The children must have an equal number of values.
                         However, the values at index {index} have a length of {a_len}, which is different from values at index 0, {len}.")
                } else {
                    Ok(())
                }
            })?;

        if validity
            .as_ref()
            .map_or(false, |validity| validity.len() != len)
        {
            polars_bail!(ComputeError:"The validity length of a StructArray must match its number of elements")
        }

        Ok(Self {
            dtype,
            values,
            validity,
        })
    }

    /// Returns a new [`StructArray`]
    /// # Panics
    /// This function panics iff:
    /// * `dtype`'s physical type is not [`crate::datatypes::PhysicalType::Struct`].
    /// * the children of `dtype` are empty
    /// * the values's len is different from children's length
    /// * any of the values's data type is different from its corresponding children' data type
    /// * any element of values has a different length than the first element
    /// * the validity's length is not equal to the length of the first element
    pub fn new(
        dtype: ArrowDataType,
        values: Vec<Box<dyn Array>>,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(dtype, values, validity).unwrap()
    }

    /// Creates an empty [`StructArray`].
    pub fn new_empty(dtype: ArrowDataType) -> Self {
        if let ArrowDataType::Struct(fields) = &dtype.to_logical_type() {
            let values = fields
                .iter()
                .map(|field| new_empty_array(field.dtype().clone()))
                .collect();
            Self::new(dtype, values, None)
        } else {
            panic!("StructArray must be initialized with DataType::Struct");
        }
    }

    /// Creates a null [`StructArray`] of length `length`.
    pub fn new_null(dtype: ArrowDataType, length: usize) -> Self {
        if let ArrowDataType::Struct(fields) = &dtype {
            let values = fields
                .iter()
                .map(|field| new_null_array(field.dtype().clone(), length))
                .collect();
            Self::new(dtype, values, Some(Bitmap::new_zeroed(length)))
        } else {
            panic!("StructArray must be initialized with DataType::Struct");
        }
    }
}

// must use
impl StructArray {
    /// Deconstructs the [`StructArray`] into its individual components.
    #[must_use]
    pub fn into_data(self) -> (Vec<Field>, Vec<Box<dyn Array>>, Option<Bitmap>) {
        let Self {
            dtype,
            values,
            validity,
        } = self;
        let fields = if let ArrowDataType::Struct(fields) = dtype {
            fields
        } else {
            unreachable!()
        };
        (fields, values, validity)
    }

    /// Slices this [`StructArray`].
    /// # Panics
    /// * `offset + length` must be smaller than `self.len()`.
    /// # Implementation
    /// This operation is `O(F)` where `F` is the number of fields.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "offset + length may not exceed length of array"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this [`StructArray`].
    /// # Implementation
    /// This operation is `O(F)` where `F` is the number of fields.
    ///
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.values
            .iter_mut()
            .for_each(|x| x.slice_unchecked(offset, length));
    }

    /// Set the outer nulls into the inner arrays.
    pub fn propagate_nulls(&self) -> StructArray {
        let has_nulls = self.null_count() > 0;
        let mut out = self.clone();
        if !has_nulls {
            return out;
        };

        for value_arr in &mut out.values {
            let new_validity = combine_validities_and(self.validity(), value_arr.validity());
            *value_arr = value_arr.with_validity(new_validity);
        }
        out
    }

    impl_sliced!();

    impl_mut_validity!();

    impl_into_array!();
}

// Accessors
impl StructArray {
    #[inline]
    fn len(&self) -> usize {
        #[cfg(debug_assertions)]
        if let Some(fst) = self.values.first() {
            for arr in self.values.iter().skip(1) {
                assert_eq!(
                    arr.len(),
                    fst.len(),
                    "StructArray invariant: each array has same length"
                );
            }
        }

        self.values.first().map(|arr| arr.len()).unwrap_or(0)
    }

    /// The optional validity.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Returns the values of this [`StructArray`].
    pub fn values(&self) -> &[Box<dyn Array>] {
        &self.values
    }

    /// Returns the fields of this [`StructArray`].
    pub fn fields(&self) -> &[Field] {
        let fields = Self::get_fields(&self.dtype);
        debug_assert_eq!(self.values().len(), fields.len());
        fields
    }
}

impl StructArray {
    /// Returns the fields the `DataType::Struct`.
    pub(crate) fn try_get_fields(dtype: &ArrowDataType) -> PolarsResult<&[Field]> {
        match dtype.to_logical_type() {
            ArrowDataType::Struct(fields) => Ok(fields),
            _ => {
                polars_bail!(ComputeError: "Struct array must be created with a DataType whose physical type is Struct")
            },
        }
    }

    /// Returns the fields the `DataType::Struct`.
    pub fn get_fields(dtype: &ArrowDataType) -> &[Field] {
        Self::try_get_fields(dtype).unwrap()
    }
}

impl Array for StructArray {
    impl_common_array!();

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    #[inline]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }
}

impl Splitable for StructArray {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (lhs_validity, rhs_validity) = unsafe { self.validity.split_at_unchecked(offset) };

        let mut lhs_values = Vec::with_capacity(self.values.len());
        let mut rhs_values = Vec::with_capacity(self.values.len());

        for v in self.values.iter() {
            let (lhs, rhs) = unsafe { v.split_at_boxed_unchecked(offset) };
            lhs_values.push(lhs);
            rhs_values.push(rhs);
        }

        (
            Self {
                dtype: self.dtype.clone(),
                values: lhs_values,
                validity: lhs_validity,
            },
            Self {
                dtype: self.dtype.clone(),
                values: rhs_values,
                validity: rhs_validity,
            },
        )
    }
}
