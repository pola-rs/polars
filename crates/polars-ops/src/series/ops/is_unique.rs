use std::hash::Hash;

use polars_arrow::bitmap::MutableBitmap;
use polars_core::prelude::row_encode::encode_rows_unordered;
use polars_core::prelude::*;
use polars_core::series::BitRepr;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use super::distinct::{repeated_element_len, repeated_element_len_series};

// If invert is true then this is an `is_duplicated`.
fn is_unique_ca<'a, T>(ca: &'a ChunkedArray<T>, invert: bool) -> BooleanChunked
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + Copy + ToTotalOrd,
    <Option<T::Physical<'a>> as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    if let Some(length) = repeated_element_len(ca) {
        return BooleanChunked::full(ca.name().clone(), invert, length);
    }

    let len = ca.len();
    let mut idx_key = PlHashMap::new();

    let mut offset: IdxSize = 0;
    for arr in ca.downcast_iter() {
        offset = is_unique_chunk(arr.iter(), offset, &mut idx_key);
    }

    let unique_idx = idx_key
        .into_iter()
        .filter_map(|(_k, v)| if v.1 { Some(v.0) } else { None });

    let (default, setter) = if invert { (true, false) } else { (false, true) };
    let mut values = MutableBitmap::with_capacity(len);
    values.extend_constant(len, default);
    for idx in unique_idx {
        unsafe { values.set_unchecked(idx as usize, setter) }
    }
    BooleanChunked::from_bitmap(ca.name().clone(), values.into())
}

/// Walks one chunk, recording where each value first appears and whether it appears just once.
fn is_unique_chunk<V, I>(
    values: I,
    offset: IdxSize,
    idx_key: &mut PlHashMap<<Option<V> as ToTotalOrd>::TotalOrdItem, (IdxSize, bool)>,
) -> IdxSize
where
    I: Iterator<Item = Option<V>>,
    Option<V>: ToTotalOrd,
    <Option<V> as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let mut idx = offset;
    values.for_each(|key| {
        idx_key
            .entry(key.to_total_ord())
            .and_modify(|v: &mut (IdxSize, bool)| v.1 = false)
            .or_insert((idx, true));
        idx += 1;
    });
    idx
}

fn is_unique_nested(s: &Series, invert: bool) -> PolarsResult<BooleanChunked> {
    if let Some(length) = repeated_element_len_series(s) {
        return Ok(BooleanChunked::full(s.name().clone(), invert, length));
    }

    let encoded = encode_rows_unordered(&[s.clone().into_column()])?.into_series();
    let ca = encoded.binary_offset().unwrap();
    Ok(is_unique_ca(ca, invert).with_name(s.name().clone()))
}

fn dispatcher(s: &Series, invert: bool) -> PolarsResult<BooleanChunked> {
    let s = s.to_physical_repr();
    use DataType::*;
    let out = match s.dtype() {
        Boolean => {
            let ca = s.bool().unwrap();
            is_unique_ca(ca, invert)
        },
        Binary => {
            let ca = s.binary().unwrap();
            is_unique_ca(ca, invert)
        },
        String => {
            let s = s.cast(&Binary).unwrap();
            let ca = s.binary().unwrap();
            is_unique_ca(ca, invert)
        },
        #[cfg(feature = "dtype-f16")]
        Float16 => {
            let ca = s.f16().unwrap();
            is_unique_ca(ca, invert)
        },
        Float32 => {
            let ca = s.f32().unwrap();
            is_unique_ca(ca, invert)
        },
        Float64 => {
            let ca = s.f64().unwrap();
            is_unique_ca(ca, invert)
        },
        List(_) => return is_unique_nested(&s, invert),
        #[cfg(feature = "dtype-array")]
        Array(_, _) => return is_unique_nested(&s, invert),
        #[cfg(feature = "dtype-struct")]
        Struct(_) => {
            let ca = s.struct_().unwrap().clone();

            if let Some(length) = repeated_element_len(&ca) {
                return Ok(BooleanChunked::full(s.name().clone(), invert, length));
            }

            let df = ca.unnest();
            return if invert {
                df.is_duplicated()
            } else {
                df.is_unique()
            };
        },
        Null => match s.len() {
            0 => BooleanChunked::new(s.name().clone(), [] as [bool; 0]),
            1 => BooleanChunked::new(s.name().clone(), [!invert]),
            len => BooleanChunked::full(s.name().clone(), invert, len),
        },
        dt if dt.is_primitive_numeric() => {
            use BitRepr as B;
            match s.bit_repr().unwrap() {
                B::U8(ca) => is_unique_ca(&ca, invert),
                B::U16(ca) => is_unique_ca(&ca, invert),
                B::U32(ca) => is_unique_ca(&ca, invert),
                B::U64(ca) => is_unique_ca(&ca, invert),
                #[cfg(feature = "dtype-u128")]
                B::U128(ca) => is_unique_ca(&ca, invert),
            }
        },
        dt => polars_bail!(opq = is_unique, dt),
    };
    Ok(out)
}

pub fn is_unique(s: &Series) -> PolarsResult<BooleanChunked> {
    dispatcher(s, false)
}

pub fn is_duplicated(s: &Series) -> PolarsResult<BooleanChunked> {
    dispatcher(s, true)
}
