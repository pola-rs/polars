use polars_utils::slice::*;

use crate::row::{RowsEncoded, SortField};

/// Encodes a value of a particular fixed width type into bytes
pub trait FixedLengthEncoding: Copy {
    // 1 is validity 0 or 1
    // bit repr of encoding
    const ENCODED_LEN: usize = 1 + std::mem::size_of::<Self::Encoded>();

    type Encoded: Sized + Copy + AsRef<[u8]> + AsMut<[u8]>;

    fn encode(self) -> Self::Encoded;

    fn decode(encoded: Self::Encoded) -> Self;
}

impl FixedLengthEncoding for bool {
    type Encoded = [u8; 1];
    fn encode(self) -> Self::Encoded {
        [self as u8]
    }

    fn decode(encoded: Self::Encoded) -> Self {
        encoded[0] != 0
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

impl FixedLengthEncoding for f32 {
    type Encoded = [u8; 4];

    fn encode(self) -> [u8; 4] {
        // https://github.com/rust-lang/rust/blob/9c20b2a8cc7588decb6de25ac6a7912dcef24d65/library/core/src/num/f32.rs#L1176-L1260
        let s = self.to_bits() as i32;
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
        let s = self.to_bits() as i64;
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
fn encode_value<T: FixedLengthEncoding>(
    value: &T,
    offset: &mut usize,
    descending: bool,
    buf: &mut [u8],
) {
    let end_offset = *offset + T::ENCODED_LEN;
    let dst = unsafe { buf.get_unchecked_release_mut(*offset..end_offset) };
    // set valid
    dst[0] = 1;
    let mut encoded = value.encode();

    // invert bits to reverse order
    if descending {
        for v in encoded.as_mut() {
            *v = !*v
        }
    }

    dst[1..].copy_from_slice(encoded.as_ref());
    *offset = end_offset;
}

pub(crate) fn encode_slice<T: FixedLengthEncoding>(
    input: &[T],
    out: &mut RowsEncoded,
    field: &SortField,
) {
    for (offset, value) in out.offsets.iter_mut().skip(1).zip(input) {
        encode_value(value, offset, field.descending, &mut out.buf);
    }
}

#[inline]
pub(super) fn null_sentinel(field: &SortField) -> u8 {
    if field.nulls_last {
        0xFF
    } else {
        0
    }
}

pub(crate) fn encode_iter<I: Iterator<Item = Option<T>>, T: FixedLengthEncoding>(
    input: I,
    out: &mut RowsEncoded,
    field: &SortField,
) {
    for (offset, opt_value) in out.offsets.iter_mut().skip(1).zip(input) {
        if let Some(value) = opt_value {
            encode_value(&value, offset, field.descending, &mut out.buf);
        } else {
            unsafe { *out.buf.get_unchecked_release_mut(*offset) = null_sentinel(field) };
            let end_offset = *offset + T::ENCODED_LEN;
            *offset = end_offset;
        }
    }
}
