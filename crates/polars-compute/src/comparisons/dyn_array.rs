use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeBinaryArray,
    ListArray, NullArray, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::types::{days_ms, f16, i256, months_days_ns};

use crate::comparisons::TotalEqKernel;

macro_rules! call_binary {
    ($T:ty, $lhs:expr, $rhs:expr, $op:path) => {{
        let lhs: &$T = $lhs.as_any().downcast_ref().unwrap();
        let rhs: &$T = $rhs.as_any().downcast_ref().unwrap();
        $op(lhs, rhs)
    }};
}

macro_rules! compare {
    ($lhs:expr, $rhs:expr, $op:path) => {{
        let lhs = $lhs;
        let rhs = $rhs;

        assert_eq!(lhs.data_type(), rhs.data_type());

        use arrow::datatypes::{IntegerType as I, PhysicalType as PH, PrimitiveType as PR};
        match lhs.data_type().to_physical_type() {
            PH::Boolean => call_binary!(BooleanArray, lhs, rhs, $op),
            PH::BinaryView => call_binary!(BinaryViewArray, lhs, rhs, $op),
            PH::Utf8View => call_binary!(Utf8ViewArray, lhs, rhs, $op),
            PH::Primitive(PR::Int8) => call_binary!(PrimitiveArray<i8>, lhs, rhs, $op),
            PH::Primitive(PR::Int16) => call_binary!(PrimitiveArray<i16>, lhs, rhs, $op),
            PH::Primitive(PR::Int32) => call_binary!(PrimitiveArray<i32>, lhs, rhs, $op),
            PH::Primitive(PR::Int64) => call_binary!(PrimitiveArray<i64>, lhs, rhs, $op),
            PH::Primitive(PR::Int128) => call_binary!(PrimitiveArray<i128>, lhs, rhs, $op),
            PH::Primitive(PR::UInt8) => call_binary!(PrimitiveArray<u8>, lhs, rhs, $op),
            PH::Primitive(PR::UInt16) => call_binary!(PrimitiveArray<u16>, lhs, rhs, $op),
            PH::Primitive(PR::UInt32) => call_binary!(PrimitiveArray<u32>, lhs, rhs, $op),
            PH::Primitive(PR::UInt64) => call_binary!(PrimitiveArray<u64>, lhs, rhs, $op),
            PH::Primitive(PR::UInt128) => call_binary!(PrimitiveArray<u128>, lhs, rhs, $op),
            PH::Primitive(PR::Float16) => call_binary!(PrimitiveArray<f16>, lhs, rhs, $op),
            PH::Primitive(PR::Float32) => call_binary!(PrimitiveArray<f32>, lhs, rhs, $op),
            PH::Primitive(PR::Float64) => call_binary!(PrimitiveArray<f64>, lhs, rhs, $op),
            PH::Primitive(PR::Int256) => call_binary!(PrimitiveArray<i256>, lhs, rhs, $op),
            PH::Primitive(PR::DaysMs) => call_binary!(PrimitiveArray<days_ms>, lhs, rhs, $op),
            PH::Primitive(PR::MonthDayNano) => {
                call_binary!(PrimitiveArray<months_days_ns>, lhs, rhs, $op)
            },

            #[cfg(feature = "dtype-array")]
            PH::FixedSizeList => call_binary!(arrow::array::FixedSizeListArray, lhs, rhs, $op),
            #[cfg(not(feature = "dtype-array"))]
            PH::FixedSizeList => todo!(
                "Comparison of FixedSizeListArray is not supported without dtype-array feature"
            ),

            PH::Null => call_binary!(NullArray, lhs, rhs, $op),
            PH::FixedSizeBinary => call_binary!(FixedSizeBinaryArray, lhs, rhs, $op),
            PH::Binary => call_binary!(BinaryArray<i32>, lhs, rhs, $op),
            PH::LargeBinary => call_binary!(BinaryArray<i64>, lhs, rhs, $op),
            PH::Utf8 => call_binary!(Utf8Array<i32>, lhs, rhs, $op),
            PH::LargeUtf8 => call_binary!(Utf8Array<i64>, lhs, rhs, $op),
            PH::List => call_binary!(ListArray<i32>, lhs, rhs, $op),
            PH::LargeList => call_binary!(ListArray<i64>, lhs, rhs, $op),
            PH::Struct => call_binary!(StructArray, lhs, rhs, $op),
            PH::Union => todo!("Comparison of UnionArrays is not yet supported"),
            PH::Map => todo!("Comparison of MapArrays is not yet supported"),
            PH::Dictionary(I::Int8) => call_binary!(DictionaryArray<i8>, lhs, rhs, $op),
            PH::Dictionary(I::Int16) => call_binary!(DictionaryArray<i16>, lhs, rhs, $op),
            PH::Dictionary(I::Int32) => call_binary!(DictionaryArray<i32>, lhs, rhs, $op),
            PH::Dictionary(I::Int64) => call_binary!(DictionaryArray<i64>, lhs, rhs, $op),
            PH::Dictionary(I::UInt8) => call_binary!(DictionaryArray<u8>, lhs, rhs, $op),
            PH::Dictionary(I::UInt16) => call_binary!(DictionaryArray<u16>, lhs, rhs, $op),
            PH::Dictionary(I::UInt32) => call_binary!(DictionaryArray<u32>, lhs, rhs, $op),
            PH::Dictionary(I::UInt64) => call_binary!(DictionaryArray<u64>, lhs, rhs, $op),
        }
    }};
}

pub fn array_tot_eq_missing_kernel(lhs: &dyn Array, rhs: &dyn Array) -> Bitmap {
    compare!(lhs, rhs, TotalEqKernel::tot_eq_missing_kernel)
}

pub fn array_tot_ne_missing_kernel(lhs: &dyn Array, rhs: &dyn Array) -> Bitmap {
    compare!(lhs, rhs, TotalEqKernel::tot_ne_missing_kernel)
}
