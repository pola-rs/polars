use std::simd::{Simd, SimdElement, SimdFloat};

use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::PhysicalType::Primitive;
use arrow::types::NativeType;
use multiversion::multiversion;
use num_traits::ToPrimitive;

use crate::utils::with_match_primitive_type;

#[multiversion(targets = "simd")]
fn nonnull_sum_as_f64<T>(values: &[T]) -> f64
where
    T: NativeType + SimdElement + ToPrimitive,
{
    // we choose 8 as that the maximum size of f64x8 -> 512bit wide
    let (head, simd_vals, tail) = unsafe { values.align_to::<Simd<T, 64>>() };

    let mut reduced: Simd<f64, 64> = Simd::splat(0.0);
    for chunk in simd_vals {
        reduced += chunk.cast::<f64>();
    }

    unsafe {
        reduced.reduce_sum()
            + head
                .iter()
                .map(|v| v.to_f64().unwrap_unchecked())
                .sum::<f64>()
            + tail
                .iter()
                .map(|v| v.to_f64().unwrap_unchecked())
                .sum::<f64>()
    }
}

pub fn no_null_sum_as_f64(values: &dyn Array) -> f64 {
    debug_assert_eq!(values.null_count(), 0);
    if let Primitive(primitive) = values.data_type().to_physical_type() {
        with_match_primitive_type!(primitive, |$T| {
            let arr: &PrimitiveArray<$T> = values.as_any().downcast_ref().unwrap();
            nonnull_sum_as_f64(arr.values())
        })
    } else {
        unreachable!()
    }
}
