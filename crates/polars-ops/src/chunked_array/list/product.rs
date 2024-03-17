use polars_core::export::num::NumCast;

use super::*;

fn product<T>(s: &Series) -> PolarsResult<T>
where
    T: NumCast,
{
    let prod = s.product()?.cast(&DataType::Float64)?;
    Ok(T::from(prod.f64().unwrap().get(0).unwrap()).unwrap())
}

pub(super) fn product_with_nulls(ca: &ListChunked, inner_dtype: &DataType) -> PolarsResult<Series> {
    use DataType::*;
    let out = match inner_dtype {
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
        _ => {
            polars_bail!(InvalidOperation: "`list.product` operation not supported for dtype `{inner_dtype}`")
        },
    };
    Ok(out.with_name(ca.name()))
}
