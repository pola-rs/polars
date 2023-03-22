use std::simd::{Simd, SimdElement, SimdFloat};

use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::PhysicalType::Primitive;
use arrow::types::NativeType;
use multiversion::multiversion;
use num_traits::ToPrimitive;

use crate::utils::with_match_primitive_type;

#[multiversion(targets = "simd")]
fn nonnull_mean<T>(values: &[T]) -> f64
where
    T: NativeType + SimdElement + ToPrimitive + std::iter::Sum<T>,
{
    // we choose 8 as that the maximum size of f64x8 -> 512bit wide
    let (head, simd_vals, tail) = unsafe { values.align_to::<Simd<T, 8>>() };

    let mut reduced: Simd<f64, 8> = Simd::splat(0.0);
    for chunk in simd_vals {
        reduced += chunk.cast::<f64>();
    }

    (reduced.reduce_sum()
        + head.iter().copied().sum::<T>().to_f64().unwrap()
        + tail.iter().copied().sum::<T>().to_f64().unwrap())
        / values.len() as f64
}

pub fn primitive_no_null_sum_as_f64<T>(array: &PrimitiveArray<T>) -> f64
where
    T: NativeType + SimdElement + ToPrimitive + std::iter::Sum<T>,
{
    assert_eq!(array.null_count(), 0);
    nonnull_mean(array.values())
}
pub fn no_null_sum_as_f64(values: &dyn Array) -> f64 {
    if let Primitive(primitive) = values.data_type().to_physical_type() {
        with_match_primitive_type!(primitive, |$T| {
            let arr: &PrimitiveArray<$T> = values.as_any().downcast_ref().unwrap();
            nonnull_mean(arr.values())
        })
    } else {
        unreachable!()
    }
}
