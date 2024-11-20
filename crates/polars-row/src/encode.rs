use std::mem::MaybeUninit;

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeListArray,
    ListArray, MutableBinaryArray, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;

use crate::fixed::FixedLengthEncoding;
use crate::row::{EncodingField, RowsEncoded};
use crate::{with_match_arrow_primitive_type, ArrayRef};

pub fn convert_columns(
    num_rows: usize,
    columns: &[ArrayRef],
    fields: &[EncodingField],
) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized(num_rows, columns, fields, &mut rows);
    rows
}

pub fn convert_columns_no_order(num_rows: usize, columns: &[ArrayRef]) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized_no_order(num_rows, columns, &mut rows);
    rows
}

pub fn convert_columns_amortized_no_order(
    num_rows: usize,
    columns: &[ArrayRef],
    rows: &mut RowsEncoded,
) {
    convert_columns_amortized(
        num_rows,
        columns,
        std::iter::repeat(&EncodingField::default()).take(columns.len()),
        rows,
    );
}

pub fn convert_columns_amortized<'a, I: IntoIterator<Item = &'a EncodingField>>(
    num_rows: usize,
    columns: &'a [ArrayRef],
    fields: I,
    rows: &mut RowsEncoded,
) {
    let mut total_num_bytes = 0;
    let mut row_widths = RowWidths::new(num_rows);

    // Convert all string types to their binary counterparts.
    use ArrowDataType as D;
    let columns: Vec<ArrayRef> = columns
        .iter()
        .map(|c| match c.dtype() {
            D::Utf8 => c
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap()
                .to_binary()
                .boxed(),
            D::LargeUtf8 => c
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .unwrap()
                .to_binary()
                .boxed(),
            D::Utf8View => c
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap()
                .to_binview()
                .boxed(),
            _ => c.clone(),
        })
        .collect();

    let mut encoders = Vec::with_capacity(columns.len());
    for (column, field) in columns.iter().zip(fields.clone()) {
        let (encoder, num_bytes) = num_column_bytes(column.as_ref(), field, &mut row_widths);
        encoders.push(encoder);
        total_num_bytes += num_bytes;
    }

    let mut offsets = row_widths.to_offsets();
    let mut out = Vec::<u8>::with_capacity(total_num_bytes);
    let buffer = &mut out.spare_capacity_mut()[..total_num_bytes];

    for (encoder, field) in encoders.iter_mut().zip(fields) {
        unsafe { encode_array(buffer, encoder, field, &mut offsets) };
    }

    unsafe {
        out.set_len(total_num_bytes);
    }

    let mut offsets = Vec::with_capacity(num_rows + 1);
    row_widths.extend_with_offsets(&mut offsets);
    offsets.push(total_num_bytes);

    *rows = RowsEncoded {
        values: out,
        offsets,
    };
}

#[derive(Clone)]
pub(crate) enum RowWidths {
    Constant { num_rows: usize, width: usize },
    Variable(Vec<usize>),
}

impl Default for RowWidths {
    fn default() -> Self {
        Self::Constant {
            num_rows: 0,
            width: 0,
        }
    }
}

impl RowWidths {
    fn new(num_rows: usize) -> Self {
        Self::Constant { num_rows, width: 0 }
    }

    fn push_constant(&mut self, constant: usize) {
        match self {
            Self::Constant { width, .. } => *width += constant,
            Self::Variable(v) => {
                v.iter_mut().for_each(|v| *v += constant);
            },
        }
    }
    fn push(&mut self, other: Self) {
        debug_assert_eq!(self.num_rows(), other.num_rows());

        match (std::mem::take(self), other) {
            (
                Self::Constant {
                    width: lhs,
                    num_rows,
                },
                Self::Constant { width: rhs, .. },
            ) => {
                *self = Self::Constant {
                    num_rows,
                    width: lhs + rhs,
                }
            },
            (Self::Constant { width, .. }, Self::Variable(v))
            | (Self::Variable(v), Self::Constant { width, .. }) => {
                if width == 0 {
                    *self = Self::Variable(v);
                } else {
                    *self = Self::Variable(v.into_iter().map(|v| v + width).collect());
                }
            },
            (Self::Variable(lhs), Self::Variable(rhs)) => {
                *self = Self::Variable(lhs.into_iter().zip(rhs).map(|(l, r)| l + r).collect());
            },
        }
    }

    fn collapse_chunks(&self, chunk_size: usize) -> RowWidths {
        if chunk_size == 0 {
            assert_eq!(self.num_rows(), 0);
            return RowWidths::default();
        }

        assert_eq!(self.num_rows() % chunk_size, 0);
        match self {
            Self::Constant { num_rows, width } => Self::Constant {
                num_rows: num_rows / chunk_size,
                width: width * chunk_size,
            },
            Self::Variable(v) => Self::Variable(
                v.chunks_exact(chunk_size)
                    .map(|chunk| chunk.iter().copied().sum())
                    .collect(),
            ),
        }
    }

    fn to_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.num_rows());
        self.extend_with_offsets(&mut offsets);
        offsets
    }

    fn extend_with_offsets(&self, out: &mut Vec<usize>) {
        match self {
            RowWidths::Constant { num_rows, width } => {
                out.extend((0..*num_rows).map(|i| i * width));
            },
            RowWidths::Variable(widths) => {
                let mut next = 0;
                out.extend(widths.into_iter().map(|w| {
                    let current = next;
                    next += w;
                    current
                }));
            },
        }
    }

    fn num_rows(&self) -> usize {
        match self {
            Self::Constant { num_rows, .. } => *num_rows,
            Self::Variable(widths) => widths.len(),
        }
    }

    fn get(&self, index: usize) -> usize {
        assert!(index < self.num_rows());
        match self {
            Self::Constant { width, .. } => *width,
            Self::Variable(v) => v[index],
        }
    }

    fn sum(&self) -> usize {
        match self {
            Self::Constant { num_rows, width } => *num_rows * *width,
            Self::Variable(v) => v.iter().copied().sum(),
        }
    }
}

fn biniter_num_column_bytes<'a>(
    iter: impl ExactSizeIterator<Item = usize>,
    validity: Option<&Bitmap>,
    field: &EncodingField,
) -> (RowWidths, usize) {
    let mut num_bytes = 0;
    let widths = match validity {
        None => iter
            .map(|v| {
                let n = crate::variable::encoded_len_from_len(Some(v), field);
                num_bytes += n;
                n
            })
            .collect(),
        Some(validity) => iter
            .zip(validity.iter())
            .map(|(v, is_valid)| {
                let n = crate::variable::encoded_len_from_len(is_valid.then_some(v), field);
                num_bytes += n;
                n
            })
            .collect(),
    };
    (RowWidths::Variable(widths), num_bytes)
}

/// Determine the total number of bytes needed to encode the given column.
fn num_column_bytes(
    array: &dyn Array,
    field: &EncodingField,
    row_widths: &mut RowWidths,
) -> (Encoder2, usize) {
    use ArrowDataType as D;
    let dtype = array.dtype();

    // Fast path: column has a fixed size encoding
    if let Some(size) = fixed_size(dtype) {
        let widths = RowWidths::Constant {
            num_rows: array.len(),
            width: size,
        };
        row_widths.push_constant(size);
        let state = match dtype {
            D::FixedSizeList(_, width) => {
                let arr = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

                debug_assert_eq!(arr.values().len(), arr.len() * width);
                let (nested_encoder, _) = num_column_bytes(
                    arr.values().as_ref(),
                    field,
                    &mut RowWidths::new(arr.values().len()),
                );
                EncoderState::FixedSizeList(Box::new(nested_encoder), *width)
            },
            D::Struct(_) => {
                let arr = array.as_any().downcast_ref::<StructArray>().unwrap();

                let mut nested_encoders = Vec::with_capacity(arr.values().len());
                for array in arr.values() {
                    let (encoder, _) =
                        num_column_bytes(array.as_ref(), field, &mut RowWidths::new(arr.len()));
                    nested_encoders.push(encoder);
                }
                EncoderState::Struct(nested_encoders)
            },
            _ => EncoderState::Stateless,
        };
        let encoder = Encoder2 {
            widths,
            array: array.to_boxed(),
            state,
        };
        return (encoder, array.len() * size);
    }

    match dtype {
        D::FixedSizeList(_, width) => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

            debug_assert_eq!(array.values().len(), array.len() * width);
            let mut nested_row_widths = RowWidths::new(array.values().len());
            let (nested_encoder, num_bytes) =
                num_column_bytes(array.values().as_ref(), field, &mut nested_row_widths);

            let fsl_row_widths = nested_row_widths.collapse_chunks(*width);

            row_widths.push(fsl_row_widths.clone());
            let encoder = Encoder2 {
                widths: fsl_row_widths,
                array: array.to_boxed(),
                state: EncoderState::FixedSizeList(Box::new(nested_encoder), *width),
            };
            (encoder, num_bytes)
        },
        D::Struct(_) => {
            let arr = array.as_any().downcast_ref::<StructArray>().unwrap();

            let mut struct_row_widths = RowWidths::new(arr.len());
            let mut nested_encoders = Vec::with_capacity(arr.values().len());
            let mut total_num_bytes = 0;
            for array in arr.values() {
                let (encoder, num_bytes) =
                    num_column_bytes(array.as_ref(), field, &mut struct_row_widths);
                nested_encoders.push(encoder);
                total_num_bytes += num_bytes;
            }
            row_widths.push(struct_row_widths.clone());
            (
                Encoder2 {
                    widths: struct_row_widths,
                    array: array.to_boxed(),
                    state: EncoderState::Struct(nested_encoders),
                },
                total_num_bytes,
            )
        },

        D::LargeList(_) => {
            let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let array = array.trim_to_normalized_offsets_recursive();
            let values = array.values();

            let mut list_row_widths = RowWidths::new(values.len());
            let (encoder, num_bytes) =
                num_column_bytes(values.as_ref(), field, &mut list_row_widths);

            // @TODO: make specialized implementation for list_row_widths is RowWidths::Constant
            let mut offsets = Vec::with_capacity(list_row_widths.num_rows() + 1);
            list_row_widths.extend_with_offsets(&mut offsets);
            offsets.push(num_bytes);

            let mut total_num_bytes = 0;
            let mut widths = Vec::with_capacity(array.len());
            match array.validity() {
                None => {
                    widths.extend(array.offsets().offset_and_length_iter().map(
                        |(offset, length)| {
                            let length = crate::variable::encoded_len_from_len(
                                Some(offsets[offset + length] - offsets[offset]),
                                field,
                            );
                            total_num_bytes += length;
                            length
                        },
                    ));
                },
                Some(validity) => {
                    widths.extend(
                        array
                            .offsets()
                            .offset_and_length_iter()
                            .zip(validity.iter())
                            .map(|((offset, length), is_valid)| {
                                let length = crate::variable::encoded_len_from_len(
                                    is_valid.then_some(offsets[offset + length] - offsets[offset]),
                                    field,
                                );
                                total_num_bytes += length;
                                length
                            }),
                    );
                },
            }

            let widths = RowWidths::Variable(widths);
            row_widths.push(widths.clone());
            (
                Encoder2 {
                    widths,
                    array: array.boxed(),
                    state: EncoderState::List(Box::new(encoder)),
                },
                total_num_bytes,
            )
        },
        D::List(_) => {
            let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let array = array.trim_to_normalized_offsets_recursive();
            let values = array.values();
            todo!();
        },

        D::BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            let (widths, num_bytes) = biniter_num_column_bytes(
                array.views().iter().map(|v| v.length as usize),
                array.validity(),
                field,
            );
            let encoder = Encoder2 {
                widths,
                array: array.to_boxed(),
                state: EncoderState::Stateless,
            };
            (encoder, num_bytes)
        },
        D::Binary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
            let (widths, num_bytes) = biniter_num_column_bytes(
                array
                    .offsets()
                    .windows(2)
                    .map(|vs| (vs[1] - vs[0]) as usize),
                array.validity(),
                field,
            );
            let encoder = Encoder2 {
                widths,
                array: array.to_boxed(),
                state: EncoderState::Stateless,
            };
            (encoder, num_bytes)
        },
        D::LargeBinary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            let (widths, num_bytes) = biniter_num_column_bytes(
                array
                    .offsets()
                    .windows(2)
                    .map(|vs| (vs[1] - vs[0]) as usize),
                array.validity(),
                field,
            );
            let encoder = Encoder2 {
                widths,
                array: array.to_boxed(),
                state: EncoderState::Stateless,
            };
            (encoder, num_bytes)
        },

        // Should have become the Binary equivalent physical type.
        D::Utf8View | D::Utf8 | D::LargeUtf8 => unreachable!(),

        D::Dictionary(_, _, _) => todo!(),
        //     let array = array
        //         .as_any()
        //         .downcast_ref::<DictionaryArray<u32>>()
        //         .unwrap();
        //     let iter = array
        //         .iter_typed::<Utf8ViewArray>()
        //         .unwrap()
        //         .map(|opt_s| opt_s.map(|s| s.as_bytes()));
        //     let encoder = Encoder2::Stateless(row_widths.to_offsets(), array);
        //     biniter_num_column_bytes(iter, field, row_widths)
        // },
        D::Union(_, _, _) => todo!(),
        D::Map(_, _) => todo!(),
        D::Decimal(_, _) => todo!(),
        D::Decimal256(_, _) => todo!(),
        D::Extension(_, _, _) => todo!(),
        D::Unknown => todo!(),

        // All non-physical types
        D::Timestamp(_, _)
        | D::Date32
        | D::Date64
        | D::Time32(_)
        | D::Time64(_)
        | D::Duration(_)
        | D::Interval(_) => unreachable!(),

        // Should be fixed size type
        _ => unreachable!(),
    }
}

pub struct Encoder2 {
    widths: RowWidths,
    array: Box<dyn Array>,
    state: EncoderState,
}

pub enum EncoderState {
    Stateless,
    List(Box<Encoder2>),
    Dictionary(Box<Encoder2>),
    FixedSizeList(Box<Encoder2>, usize),
    Struct(Vec<Encoder2>),
}

unsafe fn encode_flat_array(
    buffer: &mut [MaybeUninit<u8>],
    array: &dyn Array,
    field: &EncodingField,
    offsets: &mut [usize],
) {
    use ArrowDataType as D;
    match array.dtype() {
        D::Null => return,
        D::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            crate::fixed::encode_iter(buffer, array.iter(), field, offsets);
        },
        dt if dt.is_numeric() => with_match_arrow_primitive_type!(dt, |$T| {
            let array = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
            encode_primitive(buffer, array, field, offsets);
        }),

        D::Binary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
            crate::variable::encode_iter(buffer, array.iter(), field, offsets);
        },
        D::LargeBinary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            crate::variable::encode_iter(buffer, array.iter(), field, offsets);
        },
        D::BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            crate::variable::encode_iter(buffer, array.iter(), field, offsets);
        },

        D::FixedSizeBinary(_) => todo!(),
        D::Decimal(_, _) => todo!(),
        D::Decimal256(_, _) => todo!(),

        D::Union(_, _, _) => todo!(),
        D::Map(_, _) => todo!(),
        D::Extension(_, _, _) => todo!(),
        D::Unknown => todo!(),

        // These are converted to their binary counterpart before.
        D::Utf8View | D::Utf8 | D::LargeUtf8 => unreachable!(),

        // All are non-physical types.
        D::Timestamp(_, _)
        | D::Date32
        | D::Date64
        | D::Time32(_)
        | D::Time64(_)
        | D::Duration(_)
        | D::Interval(_) => unreachable!(),

        _ => unreachable!(),
    }
}

unsafe fn encode_array(
    buffer: &mut [MaybeUninit<u8>],
    encoder: &Encoder2,
    field: &EncodingField,
    offsets: &mut [usize],
) {
    match &encoder.state {
        EncoderState::Stateless => {
            encode_flat_array(buffer, encoder.array.as_ref(), field, offsets)
        },
        EncoderState::List(nested_encoder) => {
            // @TODO: make more general.
            let array = encoder
                .array
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap();

            // @TODO: into scratch
            let total_num_bytes = nested_encoder.widths.sum();
            let mut nested_buffer = Vec::<u8>::with_capacity(total_num_bytes);
            let nested_buffer = &mut nested_buffer.spare_capacity_mut()[..total_num_bytes];
            let mut nested_offsets = nested_encoder.widths.to_offsets();

            encode_array(nested_buffer, nested_encoder, field, &mut nested_offsets);

            let nested_buffer: &[u8] = unsafe { std::mem::transmute(nested_buffer) };

            // @TODO: make specialized implementation for nested_encoder.widths is RowWidths::Constant
            let mut nested_offsets = Vec::with_capacity(nested_encoder.widths.num_rows() + 1);
            nested_encoder
                .widths
                .extend_with_offsets(&mut nested_offsets);
            nested_offsets.push(total_num_bytes);

            // @TODO: Differentiate between empty values and empty array.
            match encoder.array.validity() {
                None => {
                    crate::variable::encode_iter(
                        buffer,
                        array
                            .offsets()
                            .offset_and_length_iter()
                            .map(|(offset, length)| {
                                Some(
                                    &nested_buffer
                                        [nested_offsets[offset]..nested_offsets[offset + length]],
                                )
                            }),
                        field,
                        offsets,
                    );
                },
                Some(validity) => {
                    crate::variable::encode_iter(
                        buffer,
                        array
                            .offsets()
                            .offset_and_length_iter()
                            .zip(validity.iter())
                            .map(|((offset, length), is_valid)| {
                                is_valid.then(|| {
                                    &nested_buffer
                                        [nested_offsets[offset]..nested_offsets[offset + length]]
                                })
                            }),
                        field,
                        offsets,
                    );
                },
            }
        },
        EncoderState::Dictionary(_) => todo!(),
        EncoderState::FixedSizeList(array, width) => {
            // @TODO: handle validity
            if *width == 0 {
                return;
            }

            let mut child_offsets = Vec::with_capacity(offsets.len() * width);
            for i in 0..offsets.len() {
                let mut offset = offsets[i];
                for j in 0..*width {
                    child_offsets.push(offset);
                    offset += array.widths.get((i * width) + j);
                }
            }
            encode_array(buffer, array.as_ref(), field, &mut child_offsets);
            for i in 0..offsets.len() {
                offsets[i] = child_offsets[(i + 1) * width - 1];
            }
        },
        EncoderState::Struct(arrays) => {
            if arrays.is_empty() {
                return;
            }

            for array in arrays {
                encode_array(buffer, array, field, offsets);
            }
        },
    }

    match &encoder.widths {
        RowWidths::Constant { width, .. } => offsets.iter_mut().for_each(|v| *v += *width),
        RowWidths::Variable(v) => offsets.iter_mut().zip(v.iter()).for_each(|(v, w)| *v += *w),
    }
}

unsafe fn encode_primitive<T: NativeType + FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    arr: &PrimitiveArray<T>,
    field: &EncodingField,
    row_starts: &mut [usize],
) {
    if arr.null_count() == 0 {
        crate::fixed::encode_slice(buffer, arr.values().as_slice(), field, row_starts)
    } else {
        crate::fixed::encode_iter(
            buffer,
            arr.into_iter().map(|v| v.copied()),
            field,
            row_starts,
        )
    }
}

pub fn fixed_size(dtype: &ArrowDataType) -> Option<usize> {
    use ArrowDataType::*;
    Some(match dtype {
        UInt8 => u8::ENCODED_LEN,
        UInt16 => u16::ENCODED_LEN,
        UInt32 => u32::ENCODED_LEN,
        UInt64 => u64::ENCODED_LEN,
        Int8 => i8::ENCODED_LEN,
        Int16 => i16::ENCODED_LEN,
        Int32 => i32::ENCODED_LEN,
        Int64 => i64::ENCODED_LEN,
        Decimal(_, _) => i128::ENCODED_LEN,
        Float32 => f32::ENCODED_LEN,
        Float64 => f64::ENCODED_LEN,
        Boolean => bool::ENCODED_LEN,
       FixedSizeList(f, width) => width * fixed_size(f.dtype())?,
        Struct(fs) => {
            let mut sum = 0;
            for f in fs {
                sum += fixed_size(f.dtype())?;
            }
            sum
        },
        Null => 0,
        _ => return None,
    })
}

#[cfg(test)]
mod test {
    use arrow::array::Int32Array;
    use arrow::legacy::prelude::LargeListArray;
    use arrow::offset::Offsets;

    use super::*;
    use crate::decode::decode_rows_from_binary;
    use crate::variable::{decode_binview, BLOCK_SIZE, EMPTY_SENTINEL, NON_EMPTY_SENTINEL};

    #[test]
    fn test_fixed_and_variable_encode() {
        let a = Int32Array::from_vec(vec![1, 2, 3]);
        let b = Int32Array::from_vec(vec![213, 12, 12]);
        let c = Utf8ViewArray::from_slice([Some("a"), Some(""), Some("meep")]);

        let encoded = convert_columns_no_order(a.len(), &[Box::new(a), Box::new(b), Box::new(c)]);
        assert_eq!(encoded.offsets, &[0, 44, 55, 99]);
        assert_eq!(encoded.values.len(), 99);
        assert!(encoded.values.ends_with(&[0, 0, 0, 4]));
        assert!(encoded.values.starts_with(&[1, 128, 0, 0, 1, 1, 128]));
    }

    #[test]
    fn test_str_encode() {
        let sentence = "The black cat walked under a ladder but forget it's milk so it ...";
        let arr =
            BinaryViewArray::from_slice([Some("a"), Some(""), Some("meep"), Some(sentence), None]);

        let field = EncodingField::new_sorted(false, false);
        let arr = arrow::compute::cast::cast(&arr, &ArrowDataType::BinaryView, Default::default())
            .unwrap();
        let rows_encoded = convert_columns(arr.len(), &[arr], &[field]);
        let row1 = rows_encoded.get(0);

        // + 2 for the start valid byte and for the continuation token
        assert_eq!(row1.len(), BLOCK_SIZE + 2);
        let mut expected = [0u8; BLOCK_SIZE + 2];
        expected[0] = NON_EMPTY_SENTINEL;
        expected[1] = b'a';
        *expected.last_mut().unwrap() = 1;
        assert_eq!(row1, expected);

        let row2 = rows_encoded.get(1);
        let expected = &[EMPTY_SENTINEL];
        assert_eq!(row2, expected);

        let row3 = rows_encoded.get(2);
        let mut expected = [0u8; BLOCK_SIZE + 2];
        expected[0] = NON_EMPTY_SENTINEL;
        *expected.last_mut().unwrap() = 4;
        expected[1..5].copy_from_slice(b"meep");
        assert_eq!(row3, expected);

        let row4 = rows_encoded.get(3);
        let expected = [
            2, 84, 104, 101, 32, 98, 108, 97, 99, 107, 32, 99, 97, 116, 32, 119, 97, 108, 107, 101,
            100, 32, 117, 110, 100, 101, 114, 32, 97, 32, 108, 97, 100, 255, 100, 101, 114, 32, 98,
            117, 116, 32, 102, 111, 114, 103, 101, 116, 32, 105, 116, 39, 115, 32, 109, 105, 108,
            107, 32, 115, 111, 32, 105, 116, 32, 46, 255, 46, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
        ];
        assert_eq!(row4, expected);
        let row5 = rows_encoded.get(4);
        let expected = &[0u8];
        assert_eq!(row5, expected);
    }

    #[test]
    fn test_str_encode_block_size() {
        // create a key of exactly block size
        // and check the round trip
        let mut val = String::new();
        for i in 0..BLOCK_SIZE {
            val.push(char::from_u32(i as u32).unwrap())
        }

        let a = [val.as_str(), val.as_str(), val.as_str()];

        let field = EncodingField::new_sorted(false, false);
        let arr = BinaryViewArray::from_slice_values(a);
        let rows_encoded = convert_columns_no_order(arr.len(), &[arr.clone().boxed()]);

        let mut rows = rows_encoded.iter().collect::<Vec<_>>();
        let decoded = unsafe { decode_binview(&mut rows, &field) };
        assert_eq!(decoded, arr);
    }

    #[test]
    fn test_reverse_variable() {
        let a = Utf8ViewArray::from_slice_values(["one", "two", "three", "four", "five", "six"]);

        let fields = &[EncodingField::new_sorted(true, false)];

        let dtypes = [ArrowDataType::Utf8View];

        unsafe {
            let encoded = convert_columns(a.len(), &[Box::new(a.clone())], fields);
            let out = decode_rows_from_binary(&encoded.into_array(), fields, &dtypes, &mut vec![]);

            let arr = &out[0];
            let decoded = arr.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            assert_eq!(decoded, &a);
        }
    }

    #[test]
    fn test_list_encode() {
        let values = Utf8ViewArray::from_slice_values([
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        ]);
        let dtype = LargeListArray::default_datatype(values.dtype().clone());
        let array = LargeListArray::new(
            dtype,
            Offsets::<i64>::try_from(vec![0i64, 1, 4, 7, 7, 9, 10])
                .unwrap()
                .into(),
            values.boxed(),
            None,
        );
        let fields = &[EncodingField::new_sorted(true, false)];

        let out = convert_columns(array.len(), &[array.boxed()], fields);
        let out = out.into_array();
        assert_eq!(
            out.values().iter().map(|v| *v as usize).sum::<usize>(),
            42774
        );
    }
}
