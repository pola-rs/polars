use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeBinaryArray,
    ListArray, NullArray, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::legacy::utils::CustomIterTools;
use arrow::types::{days_ms, f16, i256, months_days_ns, Offset};

use super::TotalEqKernel;

macro_rules! compare {
    (
        $lhs:expr, $rhs:expr,
        $op:path, $true_op:expr,
        $ineq_len_rv:literal, $invalid_rv:literal
    ) => {{
        let lhs = $lhs;
        let rhs = $rhs;

        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.dtype(), rhs.dtype());

        macro_rules! call_binary {
            ($T:ty) => {{
                let lhs_values: &$T = $lhs.values().as_any().downcast_ref().unwrap();
                let rhs_values: &$T = $rhs.values().as_any().downcast_ref().unwrap();

                (0..$lhs.len())
                    .map(|i| {
                        let lval = $lhs.validity().map_or(true, |v| v.get(i).unwrap());
                        let rval = $rhs.validity().map_or(true, |v| v.get(i).unwrap());

                        if !lval || !rval {
                            return $invalid_rv;
                        }

                        // SAFETY: ListArray's invariant offsets.len_proxy() == len
                        let (lstart, lend) = unsafe { $lhs.offsets().start_end_unchecked(i) };
                        let (rstart, rend) = unsafe { $rhs.offsets().start_end_unchecked(i) };

                        if lend - lstart != rend - rstart {
                            return $ineq_len_rv;
                        }

                        let mut lhs_values = lhs_values.clone();
                        lhs_values.slice(lstart, lend - lstart);
                        let mut rhs_values = rhs_values.clone();
                        rhs_values.slice(rstart, rend - rstart);

                        $true_op($op(&lhs_values, &rhs_values))
                    })
                    .collect_trusted()
            }};
        }

        use arrow::datatypes::{IntegerType as I, PhysicalType as PH, PrimitiveType as PR};
        match lhs.values().dtype().to_physical_type() {
            PH::Boolean => call_binary!(BooleanArray),
            PH::BinaryView => call_binary!(BinaryViewArray),
            PH::Utf8View => call_binary!(Utf8ViewArray),
            PH::Primitive(PR::Int8) => call_binary!(PrimitiveArray<i8>),
            PH::Primitive(PR::Int16) => call_binary!(PrimitiveArray<i16>),
            PH::Primitive(PR::Int32) => call_binary!(PrimitiveArray<i32>),
            PH::Primitive(PR::Int64) => call_binary!(PrimitiveArray<i64>),
            PH::Primitive(PR::Int128) => call_binary!(PrimitiveArray<i128>),
            PH::Primitive(PR::UInt8) => call_binary!(PrimitiveArray<u8>),
            PH::Primitive(PR::UInt16) => call_binary!(PrimitiveArray<u16>),
            PH::Primitive(PR::UInt32) => call_binary!(PrimitiveArray<u32>),
            PH::Primitive(PR::UInt64) => call_binary!(PrimitiveArray<u64>),
            PH::Primitive(PR::UInt128) => call_binary!(PrimitiveArray<u128>),
            PH::Primitive(PR::Float16) => call_binary!(PrimitiveArray<f16>),
            PH::Primitive(PR::Float32) => call_binary!(PrimitiveArray<f32>),
            PH::Primitive(PR::Float64) => call_binary!(PrimitiveArray<f64>),
            PH::Primitive(PR::Int256) => call_binary!(PrimitiveArray<i256>),
            PH::Primitive(PR::DaysMs) => call_binary!(PrimitiveArray<days_ms>),
            PH::Primitive(PR::MonthDayNano) => {
                call_binary!(PrimitiveArray<months_days_ns>)
            },

            #[cfg(feature = "dtype-array")]
            PH::FixedSizeList => call_binary!(arrow::array::FixedSizeListArray),
            #[cfg(not(feature = "dtype-array"))]
            PH::FixedSizeList => todo!(
                "Comparison of FixedSizeListArray is not supported without dtype-array feature"
            ),

            PH::Null => call_binary!(NullArray),
            PH::FixedSizeBinary => call_binary!(FixedSizeBinaryArray),
            PH::Binary => call_binary!(BinaryArray<i32>),
            PH::LargeBinary => call_binary!(BinaryArray<i64>),
            PH::Utf8 => call_binary!(Utf8Array<i32>),
            PH::LargeUtf8 => call_binary!(Utf8Array<i64>),
            PH::List => call_binary!(ListArray<i32>),
            PH::LargeList => call_binary!(ListArray<i64>),
            PH::Struct => call_binary!(StructArray),
            PH::Union => todo!("Comparison of UnionArrays is not yet supported"),
            PH::Map => todo!("Comparison of MapArrays is not yet supported"),
            PH::Dictionary(I::Int8) => call_binary!(DictionaryArray<i8>),
            PH::Dictionary(I::Int16) => call_binary!(DictionaryArray<i16>),
            PH::Dictionary(I::Int32) => call_binary!(DictionaryArray<i32>),
            PH::Dictionary(I::Int64) => call_binary!(DictionaryArray<i64>),
            PH::Dictionary(I::UInt8) => call_binary!(DictionaryArray<u8>),
            PH::Dictionary(I::UInt16) => call_binary!(DictionaryArray<u16>),
            PH::Dictionary(I::UInt32) => call_binary!(DictionaryArray<u32>),
            PH::Dictionary(I::UInt64) => call_binary!(DictionaryArray<u64>),
        }
    }};
}

macro_rules! compare_broadcast {
    (
        $lhs:expr, $rhs:expr,
        $offsets:expr, $validity:expr,
        $op:path, $true_op:expr,
        $ineq_len_rv:literal, $invalid_rv:literal
    ) => {{
        let lhs = $lhs;
        let rhs = $rhs;

        macro_rules! call_binary {
            ($T:ty) => {{
                let values: &$T = $lhs.as_any().downcast_ref().unwrap();
                let scalar: &$T = $rhs.as_any().downcast_ref().unwrap();

                let length = $offsets.len_proxy();

                (0..length)
                    .map(move |i| {
                        let v = $validity.map_or(true, |v| v.get(i).unwrap());

                        if !v {
                            return $invalid_rv;
                        }

                        let (start, end) = unsafe { $offsets.start_end_unchecked(i) };

                        if end - start != scalar.len() {
                            return $ineq_len_rv;
                        }

                        // @TODO: I feel like there is a better way to do this.
                        let mut values: $T = values.clone();
                        <$T>::slice(&mut values, start, end - start);

                        $true_op($op(&values, scalar))
                    })
                    .collect_trusted()
            }};
        }

        assert_eq!(lhs.dtype(), rhs.dtype());

        use arrow::datatypes::{IntegerType as I, PhysicalType as PH, PrimitiveType as PR};
        match lhs.dtype().to_physical_type() {
            PH::Boolean => call_binary!(BooleanArray),
            PH::BinaryView => call_binary!(BinaryViewArray),
            PH::Utf8View => call_binary!(Utf8ViewArray),
            PH::Primitive(PR::Int8) => call_binary!(PrimitiveArray<i8>),
            PH::Primitive(PR::Int16) => call_binary!(PrimitiveArray<i16>),
            PH::Primitive(PR::Int32) => call_binary!(PrimitiveArray<i32>),
            PH::Primitive(PR::Int64) => call_binary!(PrimitiveArray<i64>),
            PH::Primitive(PR::Int128) => call_binary!(PrimitiveArray<i128>),
            PH::Primitive(PR::UInt8) => call_binary!(PrimitiveArray<u8>),
            PH::Primitive(PR::UInt16) => call_binary!(PrimitiveArray<u16>),
            PH::Primitive(PR::UInt32) => call_binary!(PrimitiveArray<u32>),
            PH::Primitive(PR::UInt64) => call_binary!(PrimitiveArray<u64>),
            PH::Primitive(PR::UInt128) => call_binary!(PrimitiveArray<u128>),
            PH::Primitive(PR::Float16) => call_binary!(PrimitiveArray<f16>),
            PH::Primitive(PR::Float32) => call_binary!(PrimitiveArray<f32>),
            PH::Primitive(PR::Float64) => call_binary!(PrimitiveArray<f64>),
            PH::Primitive(PR::Int256) => call_binary!(PrimitiveArray<i256>),
            PH::Primitive(PR::DaysMs) => call_binary!(PrimitiveArray<days_ms>),
            PH::Primitive(PR::MonthDayNano) => {
                call_binary!(PrimitiveArray<months_days_ns>)
            },

            #[cfg(feature = "dtype-array")]
            PH::FixedSizeList => call_binary!(arrow::array::FixedSizeListArray),
            #[cfg(not(feature = "dtype-array"))]
            PH::FixedSizeList => todo!(
                "Comparison of FixedSizeListArray is not supported without dtype-array feature"
            ),

            PH::Null => call_binary!(NullArray),
            PH::FixedSizeBinary => call_binary!(FixedSizeBinaryArray),
            PH::Binary => call_binary!(BinaryArray<i32>),
            PH::LargeBinary => call_binary!(BinaryArray<i64>),
            PH::Utf8 => call_binary!(Utf8Array<i32>),
            PH::LargeUtf8 => call_binary!(Utf8Array<i64>),
            PH::List => call_binary!(ListArray<i32>),
            PH::LargeList => call_binary!(ListArray<i64>),
            PH::Struct => call_binary!(StructArray),
            PH::Union => todo!("Comparison of UnionArrays is not yet supported"),
            PH::Map => todo!("Comparison of MapArrays is not yet supported"),
            PH::Dictionary(I::Int8) => call_binary!(DictionaryArray<i8>),
            PH::Dictionary(I::Int16) => call_binary!(DictionaryArray<i16>),
            PH::Dictionary(I::Int32) => call_binary!(DictionaryArray<i32>),
            PH::Dictionary(I::Int64) => call_binary!(DictionaryArray<i64>),
            PH::Dictionary(I::UInt8) => call_binary!(DictionaryArray<u8>),
            PH::Dictionary(I::UInt16) => call_binary!(DictionaryArray<u16>),
            PH::Dictionary(I::UInt32) => call_binary!(DictionaryArray<u32>),
            PH::Dictionary(I::UInt64) => call_binary!(DictionaryArray<u64>),
        }
    }};
}

impl<O: Offset> TotalEqKernel for ListArray<O> {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        compare!(
            self,
            other,
            TotalEqKernel::tot_eq_missing_kernel,
            |bm: Bitmap| bm.unset_bits() == 0,
            false,
            true
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        compare!(
            self,
            other,
            TotalEqKernel::tot_ne_missing_kernel,
            |bm: Bitmap| bm.set_bits() > 0,
            true,
            false
        )
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        compare_broadcast!(
            self.values().as_ref(),
            other.as_ref(),
            self.offsets(),
            self.validity(),
            TotalEqKernel::tot_eq_missing_kernel,
            |bm: Bitmap| bm.unset_bits() == 0,
            false,
            true
        )
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        compare_broadcast!(
            self.values().as_ref(),
            other.as_ref(),
            self.offsets(),
            self.validity(),
            TotalEqKernel::tot_ne_missing_kernel,
            |bm: Bitmap| bm.set_bits() > 0,
            true,
            false
        )
    }
}
