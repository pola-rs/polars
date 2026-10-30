//! Bit-packed rows, see the "Bit-Packed Rows" section of the crate documentation.
use std::ops::BitOrAssign;

use num_traits::AsPrimitive;
use polars_arrow::array::{Array, BooleanArray, NullArray, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::types::NativeType;

use crate::fixed::numeric::FixedLengthEncoding;
use crate::layout::{PackedField, PackedKind, Packing, RowEncodingLayout};
use crate::{ArrayRef, with_match_arrow_primitive_type};

#[inline(always)]
fn mask(num_bits: u32) -> u128 {
    if num_bits == 0 {
        0
    } else {
        u128::MAX >> (128 - num_bits)
    }
}

/// The order-preserving bits of `x`.
#[inline(always)]
fn to_bits<T: FixedLengthEncoding>(x: T) -> u128 {
    x.encode()
        .as_ref()
        .iter()
        .fold(0, |acc, b| (acc << 8) | *b as u128)
}

#[inline(always)]
fn from_bits<T: FixedLengthEncoding>(bits: u128) -> T {
    let mut encoded = T::Encoded::default();
    let bytes = encoded.as_mut();
    let n = bytes.len();
    bytes.copy_from_slice(&bits.to_be_bytes()[16 - n..]);
    T::decode(encoded)
}

/// Calls `put(row, value_bits, null_bit)` for every row of the column. `null_bit` is false
/// for fields without a null bit.
#[inline(always)]
fn put_values<T>(
    values: impl Iterator<Item = T>,
    validity: Option<&Bitmap>,
    field: &PackedField,
    to_bits: impl Fn(T) -> u128,
    mut put: impl FnMut(usize, u128, bool),
) {
    let invert = if field.descending {
        mask(field.value_bits)
    } else {
        0
    };
    let valid_bit = field.has_null_bit && !field.nulls_last;
    let null_bit = field.has_null_bit && field.nulls_last;
    match validity.filter(|v| v.unset_bits() > 0) {
        None => {
            for (i, x) in values.enumerate() {
                put(i, to_bits(x) ^ invert, valid_bit);
            }
        },
        Some(validity) => {
            for (i, (x, is_valid)) in values.zip(validity.iter()).enumerate() {
                if is_valid {
                    put(i, to_bits(x) ^ invert, valid_bit);
                } else {
                    put(i, 0, null_bit);
                }
            }
        },
    }
}

fn for_each_field_bits(array: &dyn Array, field: &PackedField, put: impl FnMut(usize, u128, bool)) {
    match field.kind {
        PackedKind::Null => {},
        PackedKind::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            put_values(
                array.values().iter(),
                array.validity(),
                field,
                |x| x as u128,
                put,
            );
        },
        PackedKind::Decimal => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap();
            let value_mask = mask(field.value_bits);
            let sign_mask = 1u128 << (field.value_bits - 1);
            put_values(
                array.values().iter().copied(),
                array.validity(),
                field,
                |x| (x as u128 & value_mask) ^ sign_mask,
                put,
            );
        },
        PackedKind::Number => with_match_arrow_primitive_type!(array.dtype(), |$T| {
            let array = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
            put_values(
                array.values().iter().copied(),
                array.validity(),
                field,
                to_bits::<$T>,
                put,
            );
        }),
    }
}

fn encode_words<W: NativeType + BitOrAssign>(
    num_rows: usize,
    columns: &[ArrayRef],
    packing: &Packing,
) -> PrimitiveArray<W>
where
    u128: AsPrimitive<W>,
{
    let mut words = vec![W::default(); num_rows];
    for (column, field) in columns.iter().zip(&packing.fields) {
        for_each_field_bits(column.as_ref(), field, |i, mut bits, null_bit| {
            if null_bit {
                bits |= 1 << field.value_bits;
            }
            words[i] |= (bits << field.shift).as_();
        });
    }
    PrimitiveArray::from_vec(words)
}

pub(crate) fn encode(
    num_rows: usize,
    columns: &[ArrayRef],
    layout: &RowEncodingLayout,
) -> ArrayRef {
    use ArrowDataType as D;
    let packing = layout.packing().unwrap();
    match layout.dtype() {
        D::Boolean => {
            let words = encode_words::<u8>(num_rows, columns, packing);
            BooleanArray::new(
                D::Boolean,
                Bitmap::from_trusted_len_iter(words.values_iter().map(|w| *w != 0)),
                None,
            )
            .boxed()
        },
        D::UInt8 => encode_words::<u8>(num_rows, columns, packing).boxed(),
        D::UInt16 => encode_words::<u16>(num_rows, columns, packing).boxed(),
        D::UInt32 => encode_words::<u32>(num_rows, columns, packing).boxed(),
        D::UInt64 => encode_words::<u64>(num_rows, columns, packing).boxed(),
        D::UInt128 => encode_words::<u128>(num_rows, columns, packing).boxed(),
        _ => unreachable!(),
    }
}

fn decode_field(
    num_rows: usize,
    dtype: &ArrowDataType,
    field: &PackedField,
    read: &impl Fn(usize, u32, u32) -> u128,
) -> ArrayRef {
    let validity = if field.has_null_bit {
        let valid_bit = !field.nulls_last as u128;
        let mut validity = BitmapBuilder::with_capacity(num_rows);
        for i in 0..num_rows {
            validity.push(read(i, field.shift + field.value_bits, 1) == valid_bit);
        }
        validity.into_opt_validity()
    } else {
        None
    };
    let invert = if field.descending {
        mask(field.value_bits)
    } else {
        0
    };
    let value = |i: usize| read(i, field.shift, field.value_bits) ^ invert;

    match field.kind {
        PackedKind::Null => NullArray::new(ArrowDataType::Null, num_rows).boxed(),
        PackedKind::Boolean => BooleanArray::new(
            ArrowDataType::Boolean,
            Bitmap::from_trusted_len_iter((0..num_rows).map(|i| value(i) != 0)),
            validity,
        )
        .boxed(),
        PackedKind::Decimal => {
            let sign_mask = 1u128 << (field.value_bits - 1);
            let unused_bits = 128 - field.value_bits;
            let values: Vec<i128> = (0..num_rows)
                .map(|i| (((value(i) ^ sign_mask) << unused_bits) as i128) >> unused_bits)
                .collect();
            PrimitiveArray::new(ArrowDataType::Int128, values.into(), validity).boxed()
        },
        PackedKind::Number => with_match_arrow_primitive_type!(dtype, |$T| {
            let values: Vec<$T> = (0..num_rows)
                .map(|i| from_bits::<$T>(value(i)))
                .collect();
            PrimitiveArray::<$T>::new(dtype.clone(), values.into(), validity).boxed()
        }),
    }
}

fn decode_words<W: NativeType + Into<u128>>(
    words: &[W],
    layout: &RowEncodingLayout,
) -> Vec<ArrayRef> {
    let packing = layout.packing().unwrap();
    let read = |i: usize, shift: u32, num_bits: u32| {
        let word: u128 = words[i].into();
        (word >> shift) & mask(num_bits)
    };
    layout
        .fields()
        .iter()
        .zip(&packing.fields)
        .map(|(f, p)| decode_field(words.len(), &f.dtype, p, &read))
        .collect()
}

fn primitive_values<W: NativeType>(rows: &dyn Array) -> &[W] {
    rows.as_any()
        .downcast_ref::<PrimitiveArray<W>>()
        .unwrap()
        .values()
}

pub(crate) fn decode(rows: &dyn Array, layout: &RowEncodingLayout) -> Vec<ArrayRef> {
    use ArrowDataType as D;
    match layout.dtype() {
        D::Boolean => {
            let rows = rows.as_any().downcast_ref::<BooleanArray>().unwrap();
            let words: Vec<u8> = rows.values().iter().map(|b| b as u8).collect();
            decode_words(&words, layout)
        },
        D::UInt8 => decode_words(primitive_values::<u8>(rows), layout),
        D::UInt16 => decode_words(primitive_values::<u16>(rows), layout),
        D::UInt32 => decode_words(primitive_values::<u32>(rows), layout),
        D::UInt64 => decode_words(primitive_values::<u64>(rows), layout),
        D::UInt128 => decode_words(primitive_values::<u128>(rows), layout),
        _ => unreachable!(),
    }
}
