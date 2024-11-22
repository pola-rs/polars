use std::fmt::Debug;
use std::mem::MaybeUninit;

use arrow::array::{BooleanArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_utils::slice::*;
use polars_utils::total_ord::{canonical_f32, canonical_f64};

use crate::row::EncodingField;

pub(crate) trait FromSlice {
    fn from_slice(slice: &[u8]) -> Self;
}

impl<const N: usize> FromSlice for [u8; N] {
    #[inline]
    fn from_slice(slice: &[u8]) -> Self {
        slice.try_into().unwrap()
    }
}

/// Encodes a value of a particular fixed width type into bytes
pub trait FixedLengthEncoding: Copy + Debug {
    // 1 is validity 0 or 1
    // bit repr of encoding
    const ENCODED_LEN: usize = 1 + size_of::<Self::Encoded>();

    type Encoded: Sized + Copy + AsRef<[u8]> + AsMut<[u8]>;

    fn encode(self) -> Self::Encoded;

    fn decode(encoded: Self::Encoded) -> Self;

    fn decode_reverse(mut encoded: Self::Encoded) -> Self {
        for v in encoded.as_mut() {
            *v = !*v
        }
        Self::decode(encoded)
    }
}

// encode as big endian
macro_rules! encode_unsigned {
    ($n:expr, $t:ty) => {
        impl FixedLengthEncoding for $t {
            type Encoded = [u8; $n];

            fn encode(self) -> [u8; $n] {
                self.to_be_bytes()
            }

            fn decode(encoded: Self::Encoded) -> Self {
                Self::from_be_bytes(encoded)
            }
        }
    };
}

encode_unsigned!(1, u8);
encode_unsigned!(2, u16);
encode_unsigned!(4, u32);
encode_unsigned!(8, u64);

// toggle the sign bit and then encode as big indian
macro_rules! encode_signed {
    ($n:expr, $t:ty) => {
        impl FixedLengthEncoding for $t {
            type Encoded = [u8; $n];

            fn encode(self) -> [u8; $n] {
                #[cfg(target_endian = "big")]
                {
                    todo!()
                }

                let mut b = self.to_be_bytes();
                // Toggle top "sign" bit to ensure consistent sort order
                b[0] ^= 0x80;
                b
            }

            fn decode(mut encoded: Self::Encoded) -> Self {
                // Toggle top "sign" bit
                encoded[0] ^= 0x80;
                Self::from_be_bytes(encoded)
            }
        }
    };
}

encode_signed!(1, i8);
encode_signed!(2, i16);
encode_signed!(4, i32);
encode_signed!(8, i64);
encode_signed!(16, i128);

impl FixedLengthEncoding for f32 {
    type Encoded = [u8; 4];

    fn encode(self) -> [u8; 4] {
        // https://github.com/rust-lang/rust/blob/9c20b2a8cc7588decb6de25ac6a7912dcef24d65/library/core/src/num/f32.rs#L1176-L1260
        let s = canonical_f32(self).to_bits() as i32;
        let val = s ^ (((s >> 31) as u32) >> 1) as i32;
        val.encode()
    }

    fn decode(encoded: Self::Encoded) -> Self {
        let bits = i32::decode(encoded);
        let val = bits ^ (((bits >> 31) as u32) >> 1) as i32;
        Self::from_bits(val as u32)
    }
}

impl FixedLengthEncoding for f64 {
    type Encoded = [u8; 8];

    fn encode(self) -> [u8; 8] {
        // https://github.com/rust-lang/rust/blob/9c20b2a8cc7588decb6de25ac6a7912dcef24d65/library/core/src/num/f32.rs#L1176-L1260
        let s = canonical_f64(self).to_bits() as i64;
        let val = s ^ (((s >> 63) as u64) >> 1) as i64;
        val.encode()
    }

    fn decode(encoded: Self::Encoded) -> Self {
        let bits = i64::decode(encoded);
        let val = bits ^ (((bits >> 63) as u64) >> 1) as i64;
        Self::from_bits(val as u64)
    }
}

#[inline]
unsafe fn encode_value<T: FixedLengthEncoding>(
    value: &T,
    offset: &mut usize,
    descending: bool,
    buf: &mut [MaybeUninit<u8>],
) {
    let end_offset = *offset + T::ENCODED_LEN;
    let dst = unsafe { buf.get_unchecked_mut(*offset..end_offset) };
    // set valid
    dst[0] = MaybeUninit::new(1);
    let mut encoded = value.encode();

    // invert bits to reverse order
    if descending {
        for v in encoded.as_mut() {
            *v = !*v
        }
    }

    dst[1..].copy_from_slice(encoded.as_ref().as_uninit());
    *offset = end_offset;
}

unsafe fn encode_opt_value<T: FixedLengthEncoding>(
    opt_value: Option<T>,
    offset: &mut usize,
    field: &EncodingField,
    buffer: &mut [MaybeUninit<u8>],
) {
    if let Some(value) = opt_value {
        encode_value(&value, offset, field.descending, buffer);
    } else {
        unsafe { *buffer.get_unchecked_mut(*offset) = MaybeUninit::new(get_null_sentinel(field)) };
        let end_offset = *offset + T::ENCODED_LEN;

        // initialize remaining bytes
        let remainder = unsafe { buffer.get_unchecked_mut(*offset + 1..end_offset) };
        remainder.fill(MaybeUninit::new(0));

        *offset = end_offset;
    }
}

pub(crate) unsafe fn encode_slice<T: FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    input: &[T],
    field: &EncodingField,
    row_starts: &mut [usize],
) {
    for (offset, value) in row_starts.iter_mut().zip(input) {
        encode_value(value, offset, field.descending, buffer);
    }
}

#[inline]
pub(crate) fn get_null_sentinel(field: &EncodingField) -> u8 {
    if field.nulls_last {
        0xFF
    } else {
        0
    }
}

pub(crate) unsafe fn encode_iter<I: Iterator<Item = Option<T>>, T: FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    field: &EncodingField,
    row_starts: &mut [usize],
) {
    for (offset, opt_value) in row_starts.iter_mut().zip(input) {
        encode_opt_value(opt_value, offset, field, buffer);
    }
}

pub(crate) unsafe fn encode_bool_iter<I: Iterator<Item = Option<bool>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    field: &EncodingField,
    offsets: &mut [usize],
) {
    let null_sentinel = get_null_sentinel(field);
    let true_sentinel = field.bool_true_sentinel();
    let false_sentinel = field.bool_false_sentinel();

    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        let b = match opt_value {
            None => null_sentinel,
            Some(false) => false_sentinel,
            Some(true) => true_sentinel,
        };

        *buffer.get_unchecked_mut(*offset) = MaybeUninit::new(b);
        *offset += 1;
    }
}

pub(super) unsafe fn decode_primitive<T: NativeType + FixedLengthEncoding>(
    rows: &mut [&[u8]],
    field: &EncodingField,
) -> PrimitiveArray<T>
where
    T::Encoded: FromSlice,
{
    let dtype: ArrowDataType = T::PRIMITIVE.into();
    let mut has_nulls = false;
    let null_sentinel = get_null_sentinel(field);

    let values = rows
        .iter()
        .map(|row| {
            has_nulls |= *row.get_unchecked(0) == null_sentinel;
            // skip null sentinel
            let start = 1;
            let end = start + T::ENCODED_LEN - 1;
            let slice = row.get_unchecked(start..end);
            let bytes = T::Encoded::from_slice(slice);

            if field.descending {
                T::decode_reverse(bytes)
            } else {
                T::decode(bytes)
            }
        })
        .collect::<Vec<_>>();

    let validity = if has_nulls {
        let null_sentinel = get_null_sentinel(field);
        Some(decode_nulls(rows, null_sentinel))
    } else {
        None
    };

    // validity byte and data length
    let increment_len = T::ENCODED_LEN;

    increment_row_counter(rows, increment_len);
    PrimitiveArray::new(dtype, values.into(), validity)
}

pub(super) unsafe fn decode_bool(rows: &mut [&[u8]], field: &EncodingField) -> BooleanArray {
    let mut has_nulls = false;
    let null_sentinel = get_null_sentinel(field);
    let true_sentinel = field.bool_true_sentinel();

    let values = Bitmap::from_trusted_len_iter_unchecked(rows.iter().map(|row| {
        let b = *row.get_unchecked(0);
        has_nulls |= b == null_sentinel;
        b == true_sentinel
    }));

    if !has_nulls {
        rows.iter_mut()
            .for_each(|row| *row = row.get_unchecked(1..));
        return BooleanArray::new(ArrowDataType::Boolean, values, None);
    }

    let validity = Bitmap::from_trusted_len_iter_unchecked(rows.iter_mut().map(|row| {
        let v = *row.get_unchecked(0) != null_sentinel;
        *row = row.get_unchecked(1..);
        v
    }));
    BooleanArray::new(ArrowDataType::Boolean, values, Some(validity))
}
unsafe fn increment_row_counter(rows: &mut [&[u8]], fixed_size: usize) {
    for row in rows {
        *row = row.get_unchecked(fixed_size..);
    }
}

pub(super) unsafe fn decode_opt_nulls(rows: &[&[u8]], null_sentinel: u8) -> Option<Bitmap> {
    let first_null = rows
        .iter()
        .position(|row| *row.get_unchecked(0) == null_sentinel)?;

    let mut bm = MutableBitmap::with_capacity(rows.len());
    bm.extend_constant(first_null, true);
    bm.push(false);

    bm.extend_from_trusted_len_iter_unchecked(
        rows[first_null + 1..]
            .iter()
            .map(|row| *row.get_unchecked(0) != null_sentinel),
    );

    Some(bm.freeze())
}

pub(super) unsafe fn decode_nulls(rows: &[&[u8]], null_sentinel: u8) -> Bitmap {
    rows.iter()
        .map(|row| *row.get_unchecked(0) != null_sentinel)
        .collect()
}
