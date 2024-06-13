use arrow::array::{
    Array, BinaryViewArray, BooleanArray, NullArray, PrimitiveArray, StructArray, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use super::TotalOrdKernel;

macro_rules! call_binary {
    ($T:ty, $lhs:expr, $rhs:expr, $op:path) => {{
        let lhs: &$T = $lhs.as_any().downcast_ref().unwrap();
        let rhs: &$T = $rhs.as_any().downcast_ref().unwrap();

        $op(lhs, rhs)
    }};
}

macro_rules! compare {
    ($lhs:expr, $rhs:expr, $op:path, $fold:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        assert_eq!(lhs.len(), rhs.len());
        let ArrowDataType::Struct(lhs_type) = lhs.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        let ArrowDataType::Struct(rhs_type) = rhs.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(lhs_type.len(), rhs_type.len());

        let lv = lhs.values();
        let rv = rhs.values();

        let mut fold = None;

        for i in 0..lhs_type.len() {
            assert_eq!(lhs_type[i].data_type(), rhs_type[i].data_type());

            use arrow::datatypes::PhysicalType as PH;
            use arrow::datatypes::PrimitiveType as PR;

            let lv = &lv[i];
            let rv = &rv[i];

            let new = match lhs_type[i].data_type().to_physical_type() {
                PH::Boolean => call_binary!(BooleanArray, lv, rv, $op),
                PH::BinaryView => call_binary!(BinaryViewArray, lv, rv, $op),
                PH::Utf8View => call_binary!(Utf8ViewArray, lv, rv, $op),
                PH::Primitive(PR::Int8) => call_binary!(PrimitiveArray<i8>, lv, rv, $op),
                PH::Primitive(PR::Int16) => call_binary!(PrimitiveArray<i16>, lv, rv, $op),
                PH::Primitive(PR::Int32) => call_binary!(PrimitiveArray<i32>, lv, rv, $op),
                PH::Primitive(PR::Int64) => call_binary!(PrimitiveArray<i64>, lv, rv, $op),
                PH::Primitive(PR::Int128) => call_binary!(PrimitiveArray<i128>, lv, rv, $op),
                PH::Primitive(PR::UInt8) => call_binary!(PrimitiveArray<u8>, lv, rv, $op),
                PH::Primitive(PR::UInt16) => call_binary!(PrimitiveArray<u16>, lv, rv, $op),
                PH::Primitive(PR::UInt32) => call_binary!(PrimitiveArray<u32>, lv, rv, $op),
                PH::Primitive(PR::UInt64) => call_binary!(PrimitiveArray<i64>, lv, rv, $op),
                PH::Primitive(PR::UInt128) => call_binary!(PrimitiveArray<u128>, lv, rv, $op),
                PH::Primitive(PR::Float16) => todo!("Comparison of Struct with Primitive(Float16) are not yet supported"),
                PH::Primitive(PR::Float32) => call_binary!(PrimitiveArray<f32>, lv, rv, $op),
                PH::Primitive(PR::Float64) => call_binary!(PrimitiveArray<f64>, lv, rv, $op),
                PH::Primitive(PR::Int256) => todo!("Comparison of Struct with Primitive(Int256) are not yet supported"),
                PH::Primitive(PR::DaysMs) => todo!("Comparison of Struct with Primitive(DaysMs) are not yet supported"),
                PH::Primitive(PR::MonthDayNano) => todo!("Comparison of Struct with Primitive(MonthDayNano) are not yet supported"),

                #[cfg(feature = "dtype-array")]
                PH::FixedSizeList => call_binary!(arrow::array::FixedSizeListArray, lv, rv, $op),
                #[cfg(not(feature = "dtype-array"))]
                PH::FixedSizeList => todo!("Comparison of Struct with FixedSizeList are not supported without the `dtype-array` feature"),

                PH::Null => call_binary!(NullArray, lv, rv, $op),
                PH::Binary => todo!("Comparison of Struct with Binary are not yet supported"),
                PH::FixedSizeBinary => todo!("Comparison of Struct with FixedSizeBinary are not yet supported"),
                PH::LargeBinary => todo!("Comparison of Struct with LargeBinary are not yet supported"),
                PH::Utf8 => todo!("Comparison of Struct with Utf8 are not yet supported"),
                PH::LargeUtf8 => todo!("Comparison of Struct with LargeUtf8 are not yet supported"),
                PH::List => todo!("Comparison of Struct with List are not yet supported"),
                PH::LargeList => todo!("Comparison of Struct with LargeList are not yet supported"),
                PH::Struct => call_binary!(StructArray, lv, rv, $op),
                PH::Union => todo!("Comparison of Struct with Union are not yet supported"),
                PH::Map => todo!("Comparison of Struct with Map are not yet supported"),
                PH::Dictionary(_) => todo!("Comparison of Struct with Dictionary are not yet supported"),
            };

            fold = if let Some(fold) = fold {
                Some($fold(fold, new))
            } else {
                Some(new)
            };
        }

        fold.unwrap()
    }};
}

impl TotalOrdKernel for StructArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        use std::ops::BitAnd;
        compare!(
            self,
            other,
            TotalOrdKernel::tot_eq_missing_kernel,
            |a: Bitmap, b: Bitmap| a.bitand(&b)
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        use std::ops::BitOr;
        compare!(
            self,
            other,
            TotalOrdKernel::tot_ne_missing_kernel,
            |a: Bitmap, b: Bitmap| a.bitor(&b)
        )
    }

    fn tot_lt_kernel(&self, _other: &Self) -> Bitmap {
        unimplemented!()
    }

    fn tot_le_kernel(&self, _other: &Self) -> Bitmap {
        unimplemented!()
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        todo!()
    }

    fn tot_lt_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_le_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_gt_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }

    fn tot_ge_kernel_broadcast(&self, _other: &Self::Scalar) -> Bitmap {
        unimplemented!()
    }
}
