#![allow(unsafe_op_in_unsafe_fn)]
use arrow::bitmap::{Bitmap, BitmapBuilder};

#[macro_export]
macro_rules! with_match_arrow_primitive_type {(
    $key_type:expr, impl<$T:ident> $($body:tt)*
) => ({
    use arrow::datatypes::ArrowDataType::*;
    use polars_utils::float16::pf16;
    match $key_type {
        Int8 => { type $T = i8; $($body)* },
        Int16 => { type $T = i16; $($body)* },
        Int32 => { type $T = i32; $($body)* },
        Int64 => { type $T = i64; $($body)* },
        Int128 => { type $T = i128; $($body)* },
        UInt8 => { type $T = u8; $($body)* },
        UInt16 => { type $T = u16; $($body)* },
        UInt32 => { type $T = u32; $($body)* },
        UInt64 => { type $T = u64; $($body)* },
        UInt128 => { type $T = u128; $($body)* },
        Float16 => { type $T = pf16; $($body)* },
        Float32 => { type $T = f32; $($body)* },
        Float64 => { type $T = f64; $($body)* },
        _ => unreachable!(),
    }
})}

pub(crate) unsafe fn decode_opt_nulls(rows: &[&[u8]], null_sentinel: u8) -> Option<Bitmap> {
    let first_null = rows
        .iter()
        .position(|row| *row.get_unchecked(0) == null_sentinel)?;

    let mut bm = BitmapBuilder::with_capacity(rows.len());
    bm.extend_constant(first_null, true);
    bm.push(false);

    bm.extend_trusted_len_iter(
        rows[first_null + 1..]
            .iter()
            .map(|row| *row.get_unchecked(0) != null_sentinel),
    );

    bm.into_opt_validity()
}
