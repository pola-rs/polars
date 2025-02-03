use arrow::bitmap::{Bitmap, BitmapBuilder};

#[macro_export]
macro_rules! with_match_arrow_primitive_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use arrow::datatypes::ArrowDataType::*;
    match $key_type {
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        Int128 => __with_ty__! { i128 },
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
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
