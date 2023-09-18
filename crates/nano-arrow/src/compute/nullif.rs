//! Contains the operator [`nullif`].
use crate::array::PrimitiveArray;
use crate::bitmap::Bitmap;
use crate::compute::comparison::{
    primitive_compare_values_op, primitive_compare_values_op_scalar, Simd8, Simd8PartialEq,
};
use crate::datatypes::DataType;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;
use crate::{array::Array, types::NativeType};

use super::utils::combine_validities;

/// Returns an array whose validity is null iff `lhs == rhs` or `lhs` is null.
/// This has the same semantics as postgres - the validity of the rhs is ignored.
/// # Panic
/// This function panics iff
/// * The arguments do not have the same logical type
/// * The arguments do not have the same length
/// # Example
/// ```rust
/// # use arrow2::array::Int32Array;
/// # use arrow2::datatypes::DataType;
/// # use arrow2::compute::nullif::primitive_nullif;
/// # fn main() {
/// let lhs = Int32Array::from(&[None, None, Some(1), Some(1), Some(1)]);
/// let rhs = Int32Array::from(&[None, Some(1), None, Some(1), Some(0)]);
/// let result = primitive_nullif(&lhs, &rhs);
///
/// let expected = Int32Array::from(&[None, None, Some(1), None, Some(1)]);
///
/// assert_eq!(expected, result);
/// # }
/// ```
pub fn primitive_nullif<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    let equal = primitive_compare_values_op(lhs.values(), rhs.values(), |lhs, rhs| lhs.neq(rhs));
    let equal: Option<Bitmap> = equal.into();

    let validity = combine_validities(lhs.validity(), equal.as_ref());

    PrimitiveArray::<T>::new(lhs.data_type().clone(), lhs.values().clone(), validity)
}

/// Returns a [`PrimitiveArray`] whose validity is null iff `lhs == rhs` or `lhs` is null.
///
/// This has the same semantics as postgres.
/// # Panic
/// This function panics iff
/// * The arguments do not have the same logical type
/// # Example
/// ```rust
/// # use arrow2::array::Int32Array;
/// # use arrow2::datatypes::DataType;
/// # use arrow2::compute::nullif::primitive_nullif_scalar;
/// # fn main() {
/// let lhs = Int32Array::from(&[None, None, Some(1), Some(0), Some(1)]);
/// let result = primitive_nullif_scalar(&lhs, 0);
///
/// let expected = Int32Array::from(&[None, None, Some(1), None, Some(1)]);
///
/// assert_eq!(expected, result);
/// # }
/// ```
pub fn primitive_nullif_scalar<T>(lhs: &PrimitiveArray<T>, rhs: T) -> PrimitiveArray<T>
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    let equal = primitive_compare_values_op_scalar(lhs.values(), rhs, |lhs, rhs| lhs.neq(rhs));
    let equal: Option<Bitmap> = equal.into();

    let validity = combine_validities(lhs.validity(), equal.as_ref());

    PrimitiveArray::<T>::new(lhs.data_type().clone(), lhs.values().clone(), validity)
}

/// Returns an [`Array`] with the same type as `lhs` and whose validity
/// is null iff either `lhs == rhs` or `lhs` is null.
///
/// This has the same semantics as postgres - the validity of the rhs is ignored.
/// # Panics
/// This function panics iff
/// * The arguments do not have the same logical type
/// * The arguments do not have the same length
/// * The physical type is not supported for this operation (use [`can_nullif`] to check)
/// # Example
/// ```rust
/// # use arrow2::array::Int32Array;
/// # use arrow2::datatypes::DataType;
/// # use arrow2::compute::nullif::nullif;
/// # fn main() {
/// let lhs = Int32Array::from(&[None, None, Some(1), Some(1), Some(1)]);
/// let rhs = Int32Array::from(&[None, Some(1), None, Some(1), Some(0)]);
/// let result = nullif(&lhs, &rhs);
///
/// let expected = Int32Array::from(&[None, None, Some(1), None, Some(1)]);
///
/// assert_eq!(expected, result.as_ref());
/// # }
/// ```
pub fn nullif(lhs: &dyn Array, rhs: &dyn Array) -> Box<dyn Array> {
    assert_eq!(lhs.data_type(), rhs.data_type());
    assert_eq!(lhs.len(), rhs.len());

    use crate::datatypes::PhysicalType::*;
    match lhs.data_type().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            Box::new(primitive_nullif::<$T>(
                lhs.as_any().downcast_ref().unwrap(),
                rhs.as_any().downcast_ref().unwrap(),
            ))
        }),
        other => unimplemented!("Nullif is not implemented for physical type {:?}", other),
    }
}

/// Returns an [`Array`] with the same type as `lhs` and whose validity
/// is null iff either `lhs == rhs` or `lhs` is null.
/// # Panics
/// iff
/// * Scalar is null
/// * lhs and rhs do not have the same type
/// * The physical type is not supported for this operation (use [`can_nullif`] to check)
/// # Example
/// ```rust
/// # use arrow2::array::Int32Array;
/// # use arrow2::scalar::PrimitiveScalar;
/// # use arrow2::datatypes::DataType;
/// # use arrow2::compute::nullif::nullif_scalar;
/// # fn main() {
/// let lhs = Int32Array::from(&[None, None, Some(1), Some(0), Some(1)]);
/// let rhs = PrimitiveScalar::<i32>::from(Some(0));
/// let result = nullif_scalar(&lhs, &rhs);
///
/// let expected = Int32Array::from(&[None, None, Some(1), None, Some(1)]);
///
/// assert_eq!(expected, result.as_ref());
/// # }
/// ```
pub fn nullif_scalar(lhs: &dyn Array, rhs: &dyn Scalar) -> Box<dyn Array> {
    assert_eq!(lhs.data_type(), rhs.data_type());
    use crate::datatypes::PhysicalType::*;
    match lhs.data_type().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let scalar = rhs.as_any().downcast_ref::<PrimitiveScalar<$T>>().unwrap();
            let scalar = scalar.value().expect("Scalar to be non-null");

            Box::new(primitive_nullif_scalar::<$T>(
                lhs.as_any().downcast_ref().unwrap(),
                scalar,
            ))
        }),
        other => unimplemented!("Nullif is not implemented for physical type {:?}", other),
    }
}

/// Returns whether [`nullif`] and [`nullif_scalar`] is implemented for the datatypes.
pub fn can_nullif(lhs: &DataType, rhs: &DataType) -> bool {
    if lhs != rhs {
        return false;
    };
    use crate::datatypes::PhysicalType;
    matches!(lhs.to_physical_type(), PhysicalType::Primitive(_))
}
