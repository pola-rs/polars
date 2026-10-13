#![allow(unsafe_op_in_unsafe_fn)]
use std::fmt::Debug;
use std::mem::MaybeUninit;

use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::types::NativeType;
use polars_utils::float16::pf16;
use polars_utils::total_ord::{canonical_f16, canonical_f32, canonical_f64};

use crate::row::RowEncodingOptions;
/// Encodes a value of a particular fixed width type into bytes
pub trait FixedLengthEncoding: Copy + Debug {
    // 1 is validity 0 or 1
    // bit repr of encoding
    const ENCODED_LEN: usize = 1 + size_of::<Self::Encoded>();

    type Encoded: Sized + Copy + AsRef<[u8]> + AsMut<[u8]>;

    fn encode(self) -> Self::Encoded;

    fn decode(encoded: Self::Encoded) -> Self;

    /// Invert all bits of an encoded value. Used for descending order.
    fn invert(encoded: Self::Encoded) -> Self::Encoded;

    /// Keep the encoded value if `keep`, otherwise all zeros.
    fn keep_or_zero(encoded: Self::Encoded, keep: bool) -> Self::Encoded;

    fn decode_reverse(encoded: Self::Encoded) -> Self {
        Self::decode(Self::invert(encoded))
    }
}

macro_rules! invert_and_keep_or_zero {
    ($t:ty) => {
        #[inline(always)]
        fn invert(encoded: Self::Encoded) -> Self::Encoded {
            (!Self::from_ne_bytes(encoded)).to_ne_bytes()
        }

        #[inline(always)]
        fn keep_or_zero(encoded: Self::Encoded, keep: bool) -> Self::Encoded {
            let mask = (0 as $t).wrapping_sub(keep as $t);
            (Self::from_ne_bytes(encoded) & mask).to_ne_bytes()
        }
    };
}

// encode as big endian
macro_rules! encode_unsigned {
    ($n:expr, $t:ty) => {
        impl FixedLengthEncoding for $t {
            type Encoded = [u8; $n];

            #[inline(always)]
            fn encode(self) -> [u8; $n] {
                self.to_be_bytes()
            }

            #[inline(always)]
            fn decode(encoded: Self::Encoded) -> Self {
                Self::from_be_bytes(encoded)
            }

            invert_and_keep_or_zero!($t);
        }
    };
}

encode_unsigned!(1, u8);
encode_unsigned!(2, u16);
encode_unsigned!(4, u32);
encode_unsigned!(8, u64);
encode_unsigned!(16, u128);

// toggle the sign bit and then encode as big indian
macro_rules! encode_signed {
    ($n:expr, $t:ty) => {
        impl FixedLengthEncoding for $t {
            type Encoded = [u8; $n];

            #[inline(always)]
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

            #[inline(always)]
            fn decode(mut encoded: Self::Encoded) -> Self {
                // Toggle top "sign" bit
                encoded[0] ^= 0x80;
                Self::from_be_bytes(encoded)
            }

            invert_and_keep_or_zero!($t);
        }
    };
}

encode_signed!(1, i8);
encode_signed!(2, i16);
encode_signed!(4, i32);
encode_signed!(8, i64);
encode_signed!(16, i128);

impl FixedLengthEncoding for pf16 {
    type Encoded = [u8; 2];

    fn encode(self) -> [u8; 2] {
        let s = canonical_f16(self).to_bits() as i16;
        let val = s ^ (((s >> 15) as u16) >> 1) as i16;
        val.encode()
    }

    fn decode(encoded: Self::Encoded) -> Self {
        let bits = i16::decode(encoded);
        let val = bits ^ (((bits >> 15) as u16) >> 1) as i16;
        Self::from_bits(val as u16)
    }

    #[inline(always)]
    fn invert(encoded: Self::Encoded) -> Self::Encoded {
        i16::invert(encoded)
    }

    #[inline(always)]
    fn keep_or_zero(encoded: Self::Encoded, keep: bool) -> Self::Encoded {
        i16::keep_or_zero(encoded, keep)
    }
}

impl FixedLengthEncoding for f32 {
    type Encoded = [u8; 4];

    #[inline]
    fn encode(self) -> [u8; 4] {
        // https://github.com/rust-lang/rust/blob/9c20b2a8cc7588decb6de25ac6a7912dcef24d65/library/core/src/num/f32.rs#L1176-L1260
        let s = canonical_f32(self).to_bits() as i32;
        let val = s ^ (((s >> 31) as u32) >> 1) as i32;
        val.encode()
    }

    #[inline]
    fn decode(encoded: Self::Encoded) -> Self {
        let bits = i32::decode(encoded);
        let val = bits ^ (((bits >> 31) as u32) >> 1) as i32;
        Self::from_bits(val as u32)
    }

    #[inline(always)]
    fn invert(encoded: Self::Encoded) -> Self::Encoded {
        i32::invert(encoded)
    }

    #[inline(always)]
    fn keep_or_zero(encoded: Self::Encoded, keep: bool) -> Self::Encoded {
        i32::keep_or_zero(encoded, keep)
    }
}

impl FixedLengthEncoding for f64 {
    type Encoded = [u8; 8];

    #[inline]
    fn encode(self) -> [u8; 8] {
        // https://github.com/rust-lang/rust/blob/9c20b2a8cc7588decb6de25ac6a7912dcef24d65/library/core/src/num/f32.rs#L1176-L1260
        let s = canonical_f64(self).to_bits() as i64;
        let val = s ^ (((s >> 63) as u64) >> 1) as i64;
        val.encode()
    }

    #[inline]
    fn decode(encoded: Self::Encoded) -> Self {
        let bits = i64::decode(encoded);
        let val = bits ^ (((bits >> 63) as u64) >> 1) as i64;
        Self::from_bits(val as u64)
    }

    #[inline(always)]
    fn invert(encoded: Self::Encoded) -> Self::Encoded {
        i64::invert(encoded)
    }

    #[inline(always)]
    fn keep_or_zero(encoded: Self::Encoded, keep: bool) -> Self::Encoded {
        i64::keep_or_zero(encoded, keep)
    }
}

pub unsafe fn encode<T: NativeType + FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    arr: &PrimitiveArray<T>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    if arr.null_count() == 0 {
        encode_slice(buffer, arr.values().as_slice(), opt, offsets)
    } else {
        encode_slice_with_validity(
            buffer,
            arr.values().as_slice(),
            arr.validity().unwrap(),
            opt,
            offsets,
        )
    }
}

/// Write the sentinel byte and the encoded value.
#[inline(always)]
unsafe fn write_value<T: FixedLengthEncoding>(
    dst: *mut MaybeUninit<u8>,
    sentinel: u8,
    encoded: T::Encoded,
) {
    *dst = MaybeUninit::new(sentinel);
    std::ptr::write_unaligned(dst.add(1) as *mut T::Encoded, encoded);
}

/// Write the null sentinel followed by zeros.
#[inline(always)]
unsafe fn write_null<T: FixedLengthEncoding>(dst: *mut MaybeUninit<u8>, null_sentinel: u8) {
    *dst = MaybeUninit::new(null_sentinel);
    std::ptr::write_bytes(dst.add(1), 0, T::ENCODED_LEN - 1);
}

#[inline(always)]
fn encode_value<T: FixedLengthEncoding>(value: T, descending: bool) -> T::Encoded {
    let encoded = value.encode();
    if descending {
        T::invert(encoded)
    } else {
        encoded
    }
}

pub(crate) unsafe fn encode_slice<T: FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    input: &[T],
    opt: RowEncodingOptions,
    row_starts: &mut [usize],
) {
    let descending = opt.contains(RowEncodingOptions::DESCENDING);
    let out = buffer.as_mut_ptr();
    for (offset, value) in row_starts.iter_mut().zip(input) {
        write_value::<T>(out.add(*offset), 1, encode_value(*value, descending));
        *offset += T::ENCODED_LEN;
    }
}

unsafe fn encode_slice_with_validity<T: FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    input: &[T],
    validity: &Bitmap,
    opt: RowEncodingOptions,
    row_starts: &mut [usize],
) {
    let descending = opt.contains(RowEncodingOptions::DESCENDING);
    let null_sentinel = opt.null_sentinel();
    let out = buffer.as_mut_ptr();

    // Nulls are written as the sentinel followed by zeros.
    let mut encode_chunk = |mask: u64, start: usize, end: usize| {
        for (i, value) in input[start..end].iter().enumerate() {
            let is_valid = (mask >> i) & 1 != 0;
            let sentinel = if is_valid { 1 } else { null_sentinel };
            let encoded = T::keep_or_zero(encode_value(*value, descending), is_valid);
            let offset = row_starts.get_unchecked_mut(start + i);
            write_value::<T>(out.add(*offset), sentinel, encoded);
            *offset += T::ENCODED_LEN;
        }
    };

    let mut masks = validity.fast_iter_u64();
    let mut start = 0;
    for mask in &mut masks {
        encode_chunk(mask, start, start + 64);
        start += 64;
    }
    // The remainder holds fewer than 128 bits: `lo` has up to 64 and `hi` the rest.
    let ([lo, hi], _) = masks.remainder();
    let mid = start + (input.len() - start).min(64);
    encode_chunk(lo, start, mid);
    encode_chunk(hi, mid, input.len());
}

pub(crate) unsafe fn encode_iter<I: Iterator<Item = Option<T>>, T: FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    opt: RowEncodingOptions,
    row_starts: &mut [usize],
) {
    let descending = opt.contains(RowEncodingOptions::DESCENDING);
    let null_sentinel = opt.null_sentinel();
    let out = buffer.as_mut_ptr();
    for (offset, opt_value) in row_starts.iter_mut().zip(input) {
        match opt_value {
            Some(value) => write_value::<T>(out.add(*offset), 1, encode_value(value, descending)),
            None => write_null::<T>(out.add(*offset), null_sentinel),
        }
        *offset += T::ENCODED_LEN;
    }
}

/// Decode one value and return whether it is valid.
#[inline(always)]
unsafe fn decode_value<T: FixedLengthEncoding>(
    ptr: *const u8,
    descending: bool,
    null_sentinel: u8,
) -> (T, bool) {
    let is_valid = *ptr != null_sentinel;
    let bytes = std::ptr::read_unaligned(ptr.add(1) as *const T::Encoded);
    let value = if descending {
        T::decode_reverse(bytes)
    } else {
        T::decode(bytes)
    };
    (value, is_valid)
}

pub(crate) unsafe fn decode_primitive<T: NativeType + FixedLengthEncoding>(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
) -> PrimitiveArray<T> {
    let descending = opt.contains(RowEncodingOptions::DESCENDING);
    let null_sentinel = opt.null_sentinel();
    let num_rows = rows.len();
    let mut values = Vec::<T>::with_capacity(num_rows);
    let mut validity = BitmapBuilder::with_capacity(num_rows);
    let mut has_nulls = false;

    // Validity is pushed 64 rows at a time as one word.
    let out = values.as_mut_ptr();
    let mut row = 0;
    while row < num_rows {
        let len = (num_rows - row).min(64);
        let mut word = 0u64;
        for i in 0..len {
            let slice = rows.get_unchecked_mut(row + i);
            let (value, is_valid) = decode_value::<T>(slice.as_ptr(), descending, null_sentinel);
            *slice = slice.get_unchecked(T::ENCODED_LEN..);
            word |= (is_valid as u64) << i;
            out.add(row + i).write(value);
        }
        has_nulls |= word != (u64::MAX >> (64 - len));
        validity.push_word_with_len_unchecked(word, len);
        row += len;
    }
    values.set_len(num_rows);

    let validity = if has_nulls {
        validity.into_opt_validity()
    } else {
        None
    };
    PrimitiveArray::new(T::PRIMITIVE.into(), values.into(), validity)
}
