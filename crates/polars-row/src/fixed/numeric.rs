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

            #[inline(always)]
            fn invert(encoded: Self::Encoded) -> Self::Encoded {
                (!Self::from_ne_bytes(encoded)).to_ne_bytes()
            }

            #[inline(always)]
            fn keep_or_zero(encoded: Self::Encoded, keep: bool) -> Self::Encoded {
                let mask = (0 as $t).wrapping_sub(keep as $t);
                (Self::from_ne_bytes(encoded) & mask).to_ne_bytes()
            }
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

            #[inline(always)]
            fn invert(encoded: Self::Encoded) -> Self::Encoded {
                (!Self::from_ne_bytes(encoded)).to_ne_bytes()
            }

            #[inline(always)]
            fn keep_or_zero(encoded: Self::Encoded, keep: bool) -> Self::Encoded {
                let mask = (0 as $t).wrapping_sub(keep as $t);
                (Self::from_ne_bytes(encoded) & mask).to_ne_bytes()
            }
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

/// Encode `values` with nulls as the sentinel followed by zeros. `dst` gives the location of
/// each row.
#[inline(always)]
unsafe fn encode_values_with_validity<T: FixedLengthEncoding>(
    values: &[T],
    validity: &Bitmap,
    opt: RowEncodingOptions,
    mut dst: impl FnMut(usize) -> *mut MaybeUninit<u8>,
) {
    let descending = opt.contains(RowEncodingOptions::DESCENDING);
    let null_sentinel = opt.null_sentinel();

    let mut encode_chunk = |mask: u64, start: usize, end: usize| {
        for (i, value) in values[start..end].iter().enumerate() {
            let is_valid = (mask >> i) & 1 != 0;
            let sentinel = if is_valid { 1 } else { null_sentinel };
            let encoded = T::keep_or_zero(encode_value(*value, descending), is_valid);
            write_value::<T>(dst(start + i), sentinel, encoded);
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
    let mid = start + (values.len() - start).min(64);
    encode_chunk(lo, start, mid);
    encode_chunk(hi, mid, values.len());
}

unsafe fn encode_slice_with_validity<T: FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    input: &[T],
    validity: &Bitmap,
    opt: RowEncodingOptions,
    row_starts: &mut [usize],
) {
    let out = buffer.as_mut_ptr();
    encode_values_with_validity(input, validity, opt, |i| {
        let offset = row_starts.get_unchecked_mut(i);
        let dst = out.add(*offset);
        *offset += T::ENCODED_LEN;
        dst
    });
}

/// Encode values `stride` bytes apart starting at `out`.
pub(crate) unsafe fn encode_strided<T: NativeType + FixedLengthEncoding>(
    out: *mut MaybeUninit<u8>,
    stride: usize,
    arr: &PrimitiveArray<T>,
    opt: RowEncodingOptions,
) {
    let values = arr.values().as_slice();
    match arr.validity().filter(|_| arr.null_count() > 0) {
        None => {
            let descending = opt.contains(RowEncodingOptions::DESCENDING);
            for (i, value) in values.iter().enumerate() {
                write_value::<T>(out.add(i * stride), 1, encode_value(*value, descending));
            }
        },
        Some(validity) => {
            encode_values_with_validity(values, validity, opt, |i| out.add(i * stride));
        },
    }
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

/// Collects decoded values and their validity. Validity is pushed 64 rows at a time as one
/// word.
pub(crate) struct PrimitiveCollector<T> {
    values: Vec<T>,
    validity: BitmapBuilder,
    has_nulls: bool,
}

impl<T: NativeType> PrimitiveCollector<T> {
    pub fn with_capacity(num_rows: usize) -> Self {
        Self {
            values: Vec::with_capacity(num_rows),
            validity: BitmapBuilder::with_capacity(num_rows),
            has_nulls: false,
        }
    }

    #[inline(always)]
    unsafe fn push_chunk(&mut self, word: u64, len: usize) {
        self.has_nulls |= word != (u64::MAX >> (64 - len));
        self.validity.push_word_with_len_unchecked(word, len);
    }

    pub fn finish(self) -> PrimitiveArray<T> {
        let validity = if self.has_nulls {
            self.validity.into_opt_validity()
        } else {
            None
        };
        PrimitiveArray::new(T::PRIMITIVE.into(), self.values.into(), validity)
    }
}

impl<T: NativeType + FixedLengthEncoding> PrimitiveCollector<T> {
    /// Decode `num_rows` values. `src` gives the location of each row.
    #[inline(always)]
    unsafe fn decode(
        &mut self,
        num_rows: usize,
        opt: RowEncodingOptions,
        mut src: impl FnMut(usize) -> *const u8,
    ) {
        let descending = opt.contains(RowEncodingOptions::DESCENDING);
        let null_sentinel = opt.null_sentinel();
        self.values.reserve(num_rows);
        self.validity.reserve(num_rows);

        let out = self.values.as_mut_ptr().add(self.values.len());
        let mut row = 0;
        while row < num_rows {
            let len = (num_rows - row).min(64);
            let mut word = 0u64;
            for i in 0..len {
                let (value, is_valid) = decode_value::<T>(src(row + i), descending, null_sentinel);
                word |= (is_valid as u64) << i;
                out.add(row + i).write(value);
            }
            self.push_chunk(word, len);
            row += len;
        }
        self.values.set_len(self.values.len() + num_rows);
    }

    /// Decode `num_rows` values that are `stride` bytes apart starting at `ptr`.
    pub unsafe fn decode_strided(
        &mut self,
        ptr: *const u8,
        stride: usize,
        num_rows: usize,
        opt: RowEncodingOptions,
    ) {
        self.decode(num_rows, opt, |i| ptr.add(i * stride));
    }
}

pub(crate) unsafe fn decode_primitive<T: NativeType + FixedLengthEncoding>(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
) -> PrimitiveArray<T> {
    let mut out = PrimitiveCollector::<T>::with_capacity(rows.len());
    out.decode(rows.len(), opt, |i| {
        let row = rows.get_unchecked_mut(i);
        let ptr = row.as_ptr();
        *row = row.get_unchecked(T::ENCODED_LEN..);
        ptr
    });
    out.finish()
}
