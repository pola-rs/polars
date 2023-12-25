use arrow::array::{Array, BooleanArray, Int64Array, PrimitiveArray};
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_core::datatypes::ListChunked;
use polars_core::export::num::{NumCast, ToPrimitive};
use polars_core::utils::CustomIterTools;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;

fn product<T>(s: &Series) -> PolarsResult<T>
where
    T: NumCast,
{
    let prod = s.product().cast(&DataType::Float64)?;
    Ok(T::from(prod.f64().unwrap().get(0).unwrap()).unwrap())
}

pub(super) fn product_with_nulls(ca: &ListChunked, inner_dtype: &DataType) -> PolarsResult<Series> {
    use DataType::*;
    let mut out = match inner_dtype {
        Boolean => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        Int8 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        UInt8 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        Int16 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        UInt16 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        Int32 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        UInt32 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        Int64 => {
            let out: Int64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<i64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        UInt64 => {
            let out: UInt64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<u64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        Float32 => {
            let out: Float32Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<f32>(s.as_ref()).unwrap()));
            out.into_series()
        },
        Float64 => {
            let out: Float64Chunked =
                ca.apply_amortized_generic(|s| s.map(|s| product::<f64>(s.as_ref()).unwrap()));
            out.into_series()
        },
        _ => panic!("Unsupported data type"),
    };
    out.rename(ca.name());
    Ok(out)
}

fn product_between_offsets<T, S>(values: &[T], offset: &[i64]) -> Vec<S>
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Product,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            product_slice(slice)
        })
        .collect_trusted()
}

fn dispatch_product<T, S>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + ToPrimitive,
    S: NativeType + NumCast + std::iter::Product,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    Box::new(PrimitiveArray::from_data_default(
        product_between_offsets::<_, S>(values, offsets).into(),
        validity.cloned(),
    )) as ArrayRef
}

pub(super) fn product_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let offsets = arr.offsets().as_slice();
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch_product::<i8, i64>(values, offsets, arr.validity()),
                Int16 => dispatch_product::<i16, i64>(values, offsets, arr.validity()),
                Int32 => dispatch_product::<i32, i64>(values, offsets, arr.validity()),
                Int64 => dispatch_product::<i64, i64>(values, offsets, arr.validity()),
                UInt8 => dispatch_product::<u8, i64>(values, offsets, arr.validity()),
                UInt16 => dispatch_product::<u16, i64>(values, offsets, arr.validity()),
                UInt32 => dispatch_product::<u32, i64>(values, offsets, arr.validity()),
                UInt64 => dispatch_product::<u64, u64>(values, offsets, arr.validity()),
                Float32 => dispatch_product::<f32, f32>(values, offsets, arr.validity()),
                Float64 => dispatch_product::<f64, f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name(), chunks)).unwrap()
}

pub(super) fn product_boolean(ca: &ListChunked) -> PolarsResult<Series> {
    let chunks = ca.downcast_iter().map(|arr| {
        let bitmap = arr
            .values()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .values();
        let offset = arr.offsets().as_slice();
        let (bits, bitmap_offset, _) = bitmap.as_slice();

        let mut start = offset[0];
        let out = (offset[1..]).iter().map(|end| {
            let current_offset = start;
            start = *end;
            let len = (end - current_offset) as usize;
            let zeroes = count_zeros(bits, bitmap_offset + current_offset as usize, len);
            (if zeroes != 0 { 0 } else { 1 }) as i64
        });
        Int64Array::from_trusted_len_values_iter(out).with_validity(arr.validity().cloned())
    });

    Ok(Int64Chunked::from_chunk_iter(ca.name(), chunks).into_series())
}

fn product_slice<T, S>(values: &[T]) -> S
where
    T: NativeType + ToPrimitive,
    S: NumCast + std::iter::Product,
{
    values
        .iter()
        .copied()
        .map(|t| unsafe {
            let s: S = NumCast::from(t).unwrap_unchecked_release();
            s
        })
        .product()
}
