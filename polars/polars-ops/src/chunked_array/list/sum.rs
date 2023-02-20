use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_arrow::utils::CustomIterTools;
use polars_core::datatypes::ListChunked;
use polars_core::export::num::{NumCast, ToPrimitive};
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;

fn sum_slice<T, S>(values: &[T]) -> S
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    values
        .iter()
        .copied()
        .map(|t| unsafe {
            let s: S = NumCast::from(t).unwrap_unchecked_release();
            s
        })
        .sum()
}

fn sum_between_offsets<T, S>(values: &[T], offset: &[i64]) -> Vec<S>
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Sum,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            sum_slice(slice)
        })
        .collect_trusted()
}

fn dispatch<T, S>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Sum,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    Box::new(PrimitiveArray::from_data_default(
        sum_between_offsets::<_, S>(values, offsets).into(),
        validity.cloned(),
    )) as ArrayRef
}

pub(super) fn sum_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let offsets = arr.offsets().as_slice();
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch::<i8, i64>(values, offsets, arr.validity()),
                Int16 => dispatch::<i16, i64>(values, offsets, arr.validity()),
                Int32 => dispatch::<i32, i32>(values, offsets, arr.validity()),
                Int64 => dispatch::<i64, i64>(values, offsets, arr.validity()),
                UInt8 => dispatch::<u8, i64>(values, offsets, arr.validity()),
                UInt16 => dispatch::<u16, i64>(values, offsets, arr.validity()),
                UInt32 => dispatch::<u32, u32>(values, offsets, arr.validity()),
                UInt64 => dispatch::<u64, u64>(values, offsets, arr.validity()),
                Float32 => dispatch::<f32, f32>(values, offsets, arr.validity()),
                Float64 => dispatch::<f64, f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name(), chunks)).unwrap()
}
