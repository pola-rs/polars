use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, PrimitiveArray,
    Utf8Array, Utf8ViewArray,
};
use arrow::datatypes::PhysicalType;
use arrow::types::Offset;
use arrow::with_match_primitive_type_full;
use polars_utils::total_ord::ToTotalOrd;

use crate::hyperloglogplus::HyperLogLog;

/// Get an estimate for the *cardinality* of the array (i.e. the number of unique values)
///
/// This is not currently implemented for nested types.
pub fn estimate_cardinality(array: &dyn Array) -> usize {
    if array.is_empty() {
        return 0;
    }

    if array.null_count() == array.len() {
        return 1;
    }

    // Estimate the cardinality with HyperLogLog
    use PhysicalType as PT;
    match array.dtype().to_physical_type() {
        PT::Null => 1,

        PT::Boolean => {
            let mut cardinality = 0;

            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();

            cardinality += usize::from(array.has_nulls());

            if let Some(unset_bits) = array.values().lazy_unset_bits() {
                cardinality += 1 + usize::from(unset_bits != array.len());
            } else {
                cardinality += 2;
            }

            cardinality
        },

        PT::Primitive(primitive_type) => with_match_primitive_type_full!(primitive_type, |$T| {
             let mut hll = HyperLogLog::new();

             let array = array
                 .as_any()
                 .downcast_ref::<PrimitiveArray<$T>>()
                 .unwrap();

             if array.has_nulls() {
                 for v in array.iter() {
                     let v = v.copied().unwrap_or_default();
                     hll.add(&v.to_total_ord());
                 }
             } else {
                 for v in array.values_iter() {
                     hll.add(&v.to_total_ord());
                 }
             }

             hll.count()
        }),
        PT::FixedSizeBinary => {
            let mut hll = HyperLogLog::new();

            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();

            if array.has_nulls() {
                for v in array.iter() {
                    let v = v.unwrap_or_default();
                    hll.add(v);
                }
            } else {
                for v in array.values_iter() {
                    hll.add(v);
                }
            }

            hll.count()
        },
        PT::Binary => {
            binary_offset_array_estimate(array.as_any().downcast_ref::<BinaryArray<i32>>().unwrap())
        },
        PT::LargeBinary => {
            binary_offset_array_estimate(array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap())
        },
        PT::Utf8 => binary_offset_array_estimate(
            &array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap()
                .to_binary(),
        ),
        PT::LargeUtf8 => binary_offset_array_estimate(
            &array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .unwrap()
                .to_binary(),
        ),
        PT::BinaryView => {
            binary_view_array_estimate(array.as_any().downcast_ref::<BinaryViewArray>().unwrap())
        },
        PT::Utf8View => binary_view_array_estimate(
            &array
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap()
                .to_binview(),
        ),
        PT::List => unimplemented!(),
        PT::FixedSizeList => unimplemented!(),
        PT::LargeList => unimplemented!(),
        PT::Struct => unimplemented!(),
        PT::Union => unimplemented!(),
        PT::Map => unimplemented!(),
        PT::Dictionary(_) => unimplemented!(),
    }
}

fn binary_offset_array_estimate<O: Offset>(array: &BinaryArray<O>) -> usize {
    let mut hll = HyperLogLog::new();

    if array.has_nulls() {
        for v in array.iter() {
            let v = v.unwrap_or_default();
            hll.add(v);
        }
    } else {
        for v in array.values_iter() {
            hll.add(v);
        }
    }

    hll.count()
}

fn binary_view_array_estimate(array: &BinaryViewArray) -> usize {
    let mut hll = HyperLogLog::new();

    if array.has_nulls() {
        for v in array.iter() {
            let v = v.unwrap_or_default();
            hll.add(v);
        }
    } else {
        for v in array.values_iter() {
            hll.add(v);
        }
    }

    hll.count()
}
