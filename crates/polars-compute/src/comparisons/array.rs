use arrow::array::{Array, BinaryArray, FixedSizeListArray, PrimitiveArray, Utf8Array};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use crate::comparisons::TotalOrdKernel;

/// Condenses a bitmap of n * width elements into one with n elements.
/// 
/// For each block of width bits a zero count is done. The block of bits is then
/// replaced with a single bit: the result of true_zero_count(zero_count).
fn agg_array_bitmap<F>(bm: Bitmap, width: usize, true_zero_count: F) -> Bitmap
where
    F: Fn(usize) -> bool,
{
    assert!(width > 0 && bm.len() % width == 0);
    let (slice, offset, _len) = bm.as_slice();

    (0..bm.len() / width)
        .map(|i| true_zero_count(count_zeros(slice, offset + i * width, width)))
        .collect()
}

macro_rules! call_binary {
    ($T:ty, $lhs:expr, $rhs:expr, $op:path) => {{
        let lhs: &$T = $lhs.as_any().downcast_ref().unwrap();
        let rhs: &$T = $rhs.as_any().downcast_ref().unwrap();
        $op(lhs, rhs)
    }};
}

macro_rules! compare {
    ($lhs:expr, $rhs:expr, $wrong_width:expr, $op:path) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        assert_eq!(lhs.len(), rhs.len());
        let ArrowDataType::FixedSizeList(lhs_type, lhs_width) = lhs.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        let ArrowDataType::FixedSizeList(rhs_type, rhs_width) = rhs.data_type().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(lhs_type.data_type(), rhs_type.data_type());

        if lhs_width != rhs_width {
            return Bitmap::new_with_value($wrong_width, lhs.len());
        }

        use arrow::datatypes::PhysicalType::*;
        use arrow::datatypes::PrimitiveType::*;
        let lv = lhs.values();
        let rv = rhs.values();
        match lhs_type.data_type().to_physical_type() {
            // Boolean => call_binary!(BooleanArray, lhs, rhs, $op),
            Boolean => todo!(),
            LargeUtf8 => call_binary!(Utf8Array<i64>, lv, rv, $op),
            LargeBinary => call_binary!(BinaryArray<i64>, lv, rv, $op),
            Primitive(Int8) => call_binary!(PrimitiveArray<i8>, lv, rv, $op),
            Primitive(Int16) => call_binary!(PrimitiveArray<i16>, lv, rv, $op),
            Primitive(Int32) => call_binary!(PrimitiveArray<i32>, lv, rv, $op),
            Primitive(Int64) => call_binary!(PrimitiveArray<i64>, lv, rv, $op),
            Primitive(Int128) => call_binary!(PrimitiveArray<i128>, lv, rv, $op),
            Primitive(UInt8) => call_binary!(PrimitiveArray<u8>, lv, rv, $op),
            Primitive(UInt16) => call_binary!(PrimitiveArray<u16>, lv, rv, $op),
            Primitive(UInt32) => call_binary!(PrimitiveArray<u32>, lv, rv, $op),
            Primitive(UInt64) => call_binary!(PrimitiveArray<i64>, lv, rv, $op),
            Primitive(Float32) => call_binary!(PrimitiveArray<f32>, lv, rv, $op),
            Primitive(Float64) => call_binary!(PrimitiveArray<f64>, lv, rv, $op),
            _ => todo!(
                "Comparison between {:?} are not yet supported",
                lhs.data_type().to_physical_type()
            ),
        }
    }};
}

impl TotalOrdKernel for FixedSizeListArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        // Nested comparison always done with eq_missing, propagating doesn't
        // make any sense.
        let inner = compare!(self, other, false, TotalOrdKernel::tot_eq_missing_kernel);
        agg_array_bitmap(inner, self.size(), |zeroes| zeroes == 0)
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        let inner = compare!(self, other, true, TotalOrdKernel::tot_eq_missing_kernel);
        agg_array_bitmap(inner, self.size(), |zeroes| zeroes > 0)
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
