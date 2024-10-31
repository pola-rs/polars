use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeBinaryArray,
    FixedSizeListArray, ListArray, NullArray, PrimitiveArray, StructArray, Utf8Array,
    Utf8ViewArray,
};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use arrow::legacy::utils::CustomIterTools;
use arrow::types::{days_ms, f16, i256, months_days_ns};

use super::TotalEqKernel;
use crate::comparisons::dyn_array::{array_tot_eq_missing_kernel, array_tot_ne_missing_kernel};

/// Condenses a bitmap of n * width elements into one with n elements.
///
/// For each block of width bits a zero count is done. The block of bits is then
/// replaced with a single bit: the result of true_zero_count(zero_count).
fn agg_array_bitmap<F>(bm: Bitmap, width: usize, true_zero_count: F) -> Bitmap
where
    F: Fn(usize) -> bool,
{
    if bm.len() == 1 {
        bm
    } else {
        assert!(width > 0 && bm.len() % width == 0);

        let (slice, offset, _len) = bm.as_slice();
        (0..bm.len() / width)
            .map(|i| true_zero_count(count_zeros(slice, offset + i * width, width)))
            .collect()
    }
}

impl TotalEqKernel for FixedSizeListArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        // Nested comparison always done with eq_missing, propagating doesn't
        // make any sense.

        assert_eq!(self.len(), other.len());
        let ArrowDataType::FixedSizeList(self_type, self_width) = self.dtype().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        let ArrowDataType::FixedSizeList(other_type, other_width) = other.dtype().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(self_type.dtype(), other_type.dtype());

        if self_width != other_width {
            return Bitmap::new_with_value(false, self.len());
        }

        if *self_width == 0 {
            return Bitmap::new_with_value(true, self.len());
        }

        // @TODO: It is probably worth it to dispatch to a special kernel for when there are
        // several nested arrays because that can be rather slow with this code.
        let inner = array_tot_eq_missing_kernel(self.values().as_ref(), other.values().as_ref());

        agg_array_bitmap(inner, self.size(), |zeroes| zeroes == 0)
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        assert_eq!(self.len(), other.len());
        let ArrowDataType::FixedSizeList(self_type, self_width) = self.dtype().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        let ArrowDataType::FixedSizeList(other_type, other_width) = other.dtype().to_logical_type()
        else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(self_type.dtype(), other_type.dtype());

        if self_width != other_width {
            return Bitmap::new_with_value(true, self.len());
        }

        if *self_width == 0 {
            return Bitmap::new_with_value(false, self.len());
        }

        // @TODO: It is probably worth it to dispatch to a special kernel for when there are
        // several nested arrays because that can be rather slow with this code.
        let inner = array_tot_ne_missing_kernel(self.values().as_ref(), other.values().as_ref());

        agg_array_bitmap(inner, self.size(), |zeroes| zeroes < self.size())
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let ArrowDataType::FixedSizeList(self_type, width) = self.dtype().to_logical_type() else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(self_type.dtype(), other.dtype().to_logical_type());

        let width = *width;

        if width != other.len() {
            return Bitmap::new_with_value(false, self.len());
        }

        if width == 0 {
            return Bitmap::new_with_value(true, self.len());
        }

        // @TODO: It is probably worth it to dispatch to a special kernel for when there are
        // several nested arrays because that can be rather slow with this code.
        array_fsl_tot_eq_missing_kernel(self.values().as_ref(), other.as_ref(), self.len(), width)
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let ArrowDataType::FixedSizeList(self_type, width) = self.dtype().to_logical_type() else {
            panic!("array comparison called with non-array type");
        };
        assert_eq!(self_type.dtype(), other.dtype().to_logical_type());

        let width = *width;

        if width != other.len() {
            return Bitmap::new_with_value(true, self.len());
        }

        if width == 0 {
            return Bitmap::new_with_value(false, self.len());
        }

        // @TODO: It is probably worth it to dispatch to a special kernel for when there are
        // several nested arrays because that can be rather slow with this code.
        array_fsl_tot_ne_missing_kernel(self.values().as_ref(), other.as_ref(), self.len(), width)
    }
}

macro_rules! compare {
    ($lhs:expr, $rhs:expr, $length:expr, $width:expr, $op:path, $true_op:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;

        macro_rules! call_binary {
            ($T:ty) => {{
                let values: &$T = $lhs.as_any().downcast_ref().unwrap();
                let scalar: &$T = $rhs.as_any().downcast_ref().unwrap();

                (0..$length)
                    .map(move |i| {
                        // @TODO: I feel like there is a better way to do this.
                        let mut values: $T = values.clone();
                        <$T>::slice(&mut values, i * $width, $width);

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

fn array_fsl_tot_eq_missing_kernel(
    values: &dyn Array,
    scalar: &dyn Array,
    length: usize,
    width: usize,
) -> Bitmap {
    // @NOTE: Zero-Width Array are handled before
    debug_assert_eq!(values.len(), length * width);
    debug_assert_eq!(scalar.len(), width);

    compare!(
        values,
        scalar,
        length,
        width,
        TotalEqKernel::tot_eq_missing_kernel,
        |bm: Bitmap| bm.unset_bits() == 0
    )
}

fn array_fsl_tot_ne_missing_kernel(
    values: &dyn Array,
    scalar: &dyn Array,
    length: usize,
    width: usize,
) -> Bitmap {
    // @NOTE: Zero-Width Array are handled before
    debug_assert_eq!(values.len(), length * width);
    debug_assert_eq!(scalar.len(), width);

    compare!(
        values,
        scalar,
        length,
        width,
        TotalEqKernel::tot_ne_missing_kernel,
        |bm: Bitmap| bm.set_bits() > 0
    )
}
