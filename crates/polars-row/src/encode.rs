use std::mem::MaybeUninit;

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeListArray,
    ListArray, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::{NativeType, Offset};

use crate::fixed::numeric::FixedLengthEncoding;
use crate::row::{RowsEncoded, RowEncodingOptions};
use crate::widths::RowWidths;
use crate::{with_match_arrow_primitive_type, ArrayRef};

pub fn convert_columns(
    num_rows: usize,
    columns: &[ArrayRef],
    fields: &[RowEncodingOptions],
) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized(num_rows, columns, fields.iter().copied(), &mut rows);
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
        std::iter::repeat_n(RowEncodingOptions::default(), columns.len()),
        rows,
    );
}

pub fn convert_columns_amortized<'a>(
    num_rows: usize,
    columns: &[ArrayRef],
    fields: impl IntoIterator<Item = RowEncodingOptions> + Clone,
    rows: &mut RowsEncoded,
) {
    let mut row_widths = RowWidths::new(num_rows);
    let mut encoders = columns
        .iter()
        .zip(fields.clone())
        .map(|(column, field)| get_encoder(column.as_ref(), field, &mut row_widths))
        .collect::<Vec<_>>();

    // Create an offsets array, we append 0 at the beginning here so it can serve as the final
    // offset array.
    let mut offsets = Vec::with_capacity(num_rows + 1);
    offsets.push(0);
    row_widths.extend_with_offsets(&mut offsets);

    // Create a buffer without initializing everything to zero.
    let total_num_bytes = row_widths.sum();
    let mut out = Vec::<u8>::with_capacity(total_num_bytes);
    let buffer = &mut out.spare_capacity_mut()[..total_num_bytes];

    let mut scratches = EncodeScratches::default();
    for (encoder, field) in encoders.iter_mut().zip(fields) {
        unsafe { encode_array(buffer, encoder, field, &mut offsets[1..], &mut scratches) };
    }
    // SAFETY: All the bytes in out up to total_num_bytes should now be initialized.
    unsafe {
        out.set_len(total_num_bytes);
    }

    *rows = RowsEncoded {
        values: out,
        offsets,
    };
}

fn list_num_column_bytes<O: Offset>(
    array: &dyn Array,
    opt: RowEncodingOptions,
    row_widths: &mut RowWidths,
) -> Encoder {
    let array = array.as_any().downcast_ref::<ListArray<O>>().unwrap();
    let array = array.trim_to_normalized_offsets_recursive();
    let values = array.values();

    let mut list_row_widths = RowWidths::new(values.len());
    let encoder = get_encoder(values.as_ref(), opt, &mut list_row_widths);

    let widths = match array.validity() {
        None => row_widths.append_iter(array.offsets().offset_and_length_iter().map(
            |(offset, length)| {
                let mut sum = 0;
                for i in offset..offset + length {
                    sum += list_row_widths.get(i);
                }
                1 + length + sum
            },
        )),
        Some(validity) => row_widths.append_iter(
            array
                .offsets()
                .offset_and_length_iter()
                .zip(validity.iter())
                .map(|((offset, length), is_valid)| {
                    if !is_valid {
                        return 1;
                    }

                    let mut sum = 0;
                    for i in offset..offset + length {
                        sum += list_row_widths.get(i);
                    }
                    1 + length + sum
                }),
        ),
    };

    Encoder {
        widths,
        array: array.boxed(),
        state: EncoderState::List(Box::new(encoder)),
    }
}

fn biniter_num_column_bytes(
    array: &dyn Array,
    iter: impl ExactSizeIterator<Item = usize>,
    validity: Option<&Bitmap>,
    opt: RowEncodingOptions,
    row_widths: &mut RowWidths,
) -> Encoder {
    let widths = if opt.contains(RowEncodingOptions::NO_ORDER) {
        match validity {
            None => row_widths.append_iter(
                iter.map(|v| crate::variable::no_order::len_from_item(Some(v), opt)),
            ),
            Some(validity) => {
                row_widths.append_iter(iter.zip(validity.iter()).map(|(v, is_valid)| {
                    crate::variable::no_order::len_from_item(is_valid.then_some(v), opt)
                }))
            },
        }
    } else {
        match validity {
            None => row_widths.append_iter(
                iter.map(|v| crate::variable::binary::encoded_len_from_len(Some(v), opt)),
            ),
            Some(validity) => {
                row_widths.append_iter(iter.zip(validity.iter()).map(|(v, is_valid)| {
                    crate::variable::binary::encoded_len_from_len(is_valid.then_some(v), opt)
                }))
            },
        }
    };

    Encoder {
        widths,
        array: array.to_boxed(),
        state: EncoderState::Stateless,
    }
}

fn striter_num_column_bytes(
    array: &dyn Array,
    iter: impl ExactSizeIterator<Item = usize>,
    validity: Option<&Bitmap>,
    opt: RowEncodingOptions,
    row_widths: &mut RowWidths,
) -> Encoder {
    let widths = if opt.contains(RowEncodingOptions::NO_ORDER) {
        match validity {
            None => row_widths.append_iter(
                iter.map(|v| crate::variable::no_order::len_from_item(Some(v), opt)),
            ),
            Some(validity) => {
                row_widths.append_iter(iter.zip(validity.iter()).map(|(v, is_valid)| {
                    crate::variable::no_order::len_from_item(is_valid.then_some(v), opt)
                }))
            },
        }
    } else {
        match validity {
            None => row_widths
                .append_iter(iter.map(|v| crate::variable::utf8::len_from_item(Some(v), opt))),
            Some(validity) => {
                row_widths.append_iter(iter.zip(validity.iter()).map(|(v, is_valid)| {
                    crate::variable::utf8::len_from_item(is_valid.then_some(v), opt)
                }))
            },
        }
    };

    Encoder {
        widths,
        array: array.to_boxed(),
        state: EncoderState::Stateless,
    }
}

/// Get the encoder for a specific array.
fn get_encoder(array: &dyn Array, opt: RowEncodingOptions, row_widths: &mut RowWidths) -> Encoder {
    use ArrowDataType as D;
    let dtype = array.dtype();

    // Fast path: column has a fixed size encoding
    if let Some(size) = fixed_size(dtype) {
        row_widths.push_constant(size);
        let state = match dtype {
            D::FixedSizeList(_, width) => {
                let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                let array = array.propagate_nulls();

                debug_assert_eq!(array.values().len(), array.len() * width);
                let nested_encoder = get_encoder(
                    array.values().as_ref(),
                    opt,
                    &mut RowWidths::new(array.values().len()),
                );
                EncoderState::FixedSizeList(Box::new(nested_encoder), *width)
            },
            D::Struct(_) => {
                let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
                let struct_array = struct_array.propagate_nulls();
                EncoderState::Struct(
                    struct_array
                        .values()
                        .iter()
                        .map(|array| {
                            get_encoder(
                                array.as_ref(),
                                opt,
                                &mut RowWidths::new(struct_array.len()),
                            )
                        })
                        .collect(),
                )
            },
            _ => EncoderState::Stateless,
        };
        return Encoder {
            widths: RowWidths::Constant {
                num_rows: array.len(),
                width: size,
            },
            array: array.to_boxed(),
            state,
        };
    }

    match dtype {
        D::FixedSizeList(_, width) => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let array = array.propagate_nulls();

            debug_assert_eq!(array.values().len(), array.len() * width);
            let mut nested_row_widths = RowWidths::new(array.values().len());
            let nested_encoder =
                get_encoder(array.values().as_ref(), opt, &mut nested_row_widths);

            let mut fsl_row_widths = nested_row_widths.collapse_chunks(*width, array.len());
            fsl_row_widths.push_constant(1); // validity byte

            row_widths.push(&fsl_row_widths);
            Encoder {
                widths: fsl_row_widths,
                array: array.to_boxed(),
                state: EncoderState::FixedSizeList(Box::new(nested_encoder), *width),
            }
        },
        D::Struct(_) => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let array = array.propagate_nulls();

            let mut struct_row_widths = RowWidths::new(array.len());
            let mut nested_encoders = Vec::with_capacity(array.values().len());
            struct_row_widths.push_constant(1); // validity byte
            for array in array.values() {
                let encoder = get_encoder(array.as_ref(), opt, &mut struct_row_widths);
                nested_encoders.push(encoder);
            }
            row_widths.push(&struct_row_widths);
            Encoder {
                widths: struct_row_widths,
                array: array.to_boxed(),
                state: EncoderState::Struct(nested_encoders),
            }
        },

        D::List(_) => list_num_column_bytes::<i32>(array, opt, row_widths),
        D::LargeList(_) => list_num_column_bytes::<i64>(array, opt, row_widths),

        D::BinaryView => {
            let dc_array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            biniter_num_column_bytes(
                array,
                dc_array.views().iter().map(|v| v.length as usize),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },
        D::Binary => {
            let dc_array = array.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
            biniter_num_column_bytes(
                array,
                dc_array.offsets().lengths(),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },
        D::LargeBinary => {
            let dc_array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            biniter_num_column_bytes(
                array,
                dc_array.offsets().lengths(),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },

        D::Utf8View => {
            let dc_array = array.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            striter_num_column_bytes(
                array,
                dc_array.views().iter().map(|v| v.length as usize),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },
        D::Utf8 => {
            let dc_array = array.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
            striter_num_column_bytes(
                array,
                dc_array.offsets().lengths(),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },
        D::LargeUtf8 => {
            let dc_array = array.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            striter_num_column_bytes(
                array,
                dc_array.offsets().lengths(),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },

        D::Dictionary(_, _, _) => {
            let dc_array = array
                .as_any()
                .downcast_ref::<DictionaryArray<u32>>()
                .unwrap();
            let iter = dc_array
                .iter_typed::<Utf8ViewArray>()
                .unwrap()
                .map(|opt_s| opt_s.map_or(0, |s| s.len()));
            // @TODO: Do a better job here. This is just plainly incorrect.
            biniter_num_column_bytes(array, iter, dc_array.validity(), opt, row_widths)
        },
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

pub struct Encoder {
    widths: RowWidths,
    array: Box<dyn Array>,
    state: EncoderState,
}

pub enum EncoderState {
    Stateless,
    List(Box<Encoder>),
    Dictionary(Box<Encoder>),
    FixedSizeList(Box<Encoder>, usize),
    Struct(Vec<Encoder>),
}

unsafe fn encode_strs<'a>(
    buffer: &mut [MaybeUninit<u8>],
    iter: impl Iterator<Item = Option<&'a str>>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    if opt.contains(RowEncodingOptions::NO_ORDER) {
        crate::variable::no_order::encode_variable_no_order(
            buffer,
            iter.map(|v| v.map(str::as_bytes)),
            opt,
            offsets,
        );
    } else {
        crate::variable::utf8::encode_str(buffer, iter, opt, offsets);
    }
}

unsafe fn encode_bins<'a>(
    buffer: &mut [MaybeUninit<u8>],
    iter: impl Iterator<Item = Option<&'a [u8]>>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    if opt.contains(RowEncodingOptions::NO_ORDER) {
        crate::variable::no_order::encode_variable_no_order(buffer, iter, opt, offsets);
    } else {
        crate::variable::binary::encode_iter(buffer, iter, opt, offsets);
    }
}

unsafe fn encode_flat_array(
    buffer: &mut [MaybeUninit<u8>],
    array: &dyn Array,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    use ArrowDataType as D;
    match array.dtype() {
        D::Null => {},
        D::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            crate::fixed::boolean::encode_bool(buffer, array.iter(), opt, offsets);
        },
        dt if dt.is_numeric() => with_match_arrow_primitive_type!(dt, |$T| {
            let array = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
            encode_primitive(buffer, array, opt, offsets);
        }),

        D::Binary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
            encode_bins(buffer, array.iter(), opt, offsets);
        },
        D::LargeBinary => {
            let array = array.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
            encode_bins(buffer, array.iter(), opt, offsets);
        },
        D::BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            encode_bins(buffer, array.iter(), opt, offsets);
        },
        D::Utf8 => {
            let array = array.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
            encode_strs(buffer, array.iter(), opt, offsets);
        },
        D::LargeUtf8 => {
            let array = array.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            encode_strs(buffer, array.iter(), opt, offsets);
        },
        D::Utf8View => {
            let array = array.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            encode_strs(buffer, array.iter(), opt, offsets);
        },

        D::Dictionary(_, _, _) => {
            let dc_array = array
                .as_any()
                .downcast_ref::<DictionaryArray<u32>>()
                .unwrap();
            let iter = dc_array
                .iter_typed::<Utf8ViewArray>()
                .unwrap()
                .map(|opt_s| opt_s.map(|s| s.as_bytes()));
            crate::variable::binary::encode_iter(buffer, iter, opt, offsets);
        },

        D::FixedSizeBinary(_) => todo!(),
        D::Decimal(_, _) => todo!(),
        D::Decimal256(_, _) => todo!(),

        D::Union(_, _, _) => todo!(),
        D::Map(_, _) => todo!(),
        D::Extension(_, _, _) => todo!(),
        D::Unknown => todo!(),

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

#[derive(Default)]
struct EncodeScratches {
    nested_offsets: Vec<usize>,
    nested_buffer: Vec<u8>,
}

impl EncodeScratches {
    fn clear(&mut self) {
        self.nested_offsets.clear();
        self.nested_buffer.clear();
    }
}

unsafe fn encode_array(
    buffer: &mut [MaybeUninit<u8>],
    encoder: &Encoder,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
    scratches: &mut EncodeScratches,
) {
    match &encoder.state {
        EncoderState::Stateless => {
            encode_flat_array(buffer, encoder.array.as_ref(), opt, offsets)
        },
        EncoderState::List(nested_encoder) => {
            // @TODO: make more general.
            let array = encoder
                .array
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap();

            scratches.clear();

            scratches
                .nested_offsets
                .reserve(nested_encoder.widths.num_rows());
            let nested_offsets = &mut scratches.nested_offsets;

            let list_null_sentinel = opt.list_null_sentinel();
            let list_continuation_token = opt.list_continuation_token();
            let list_termination_token = opt.list_termination_token();

            match array.validity() {
                None => {
                    for (i, (offset, length)) in
                        array.offsets().offset_and_length_iter().enumerate()
                    {
                        for j in offset..offset + length {
                            buffer[offsets[i]] = MaybeUninit::new(list_continuation_token);
                            offsets[i] += 1;

                            nested_offsets.push(offsets[i]);
                            offsets[i] += nested_encoder.widths.get(j);
                        }
                        buffer[offsets[i]] = MaybeUninit::new(list_termination_token);
                        offsets[i] += 1;
                    }
                },
                Some(validity) => {
                    for (i, ((offset, length), is_valid)) in array
                        .offsets()
                        .offset_and_length_iter()
                        .zip(validity.iter())
                        .enumerate()
                    {
                        if !is_valid {
                            buffer[offsets[i]] = MaybeUninit::new(list_null_sentinel);
                            offsets[i] += 1;
                            continue;
                        }

                        for j in offset..offset + length {
                            buffer[offsets[i]] = MaybeUninit::new(list_continuation_token);
                            offsets[i] += 1;

                            nested_offsets.push(offsets[i]);
                            offsets[i] += nested_encoder.widths.get(j);
                        }
                        buffer[offsets[i]] = MaybeUninit::new(list_termination_token);
                        offsets[i] += 1;
                    }
                },
            }

            // Lists have the row encoding of the elements again encoded by the variable encoding.
            // This is not ideal ([["a", "b"]] produces 100 bytes), but this is sort of how
            // arrow-row works and is good enough for now.
            unsafe {
                encode_array(
                    buffer,
                    nested_encoder,
                    opt,
                    nested_offsets,
                    &mut EncodeScratches::default(),
                )
            };
        },
        EncoderState::Dictionary(_) => todo!(),
        EncoderState::FixedSizeList(array, width) => {
            encode_validity(buffer, encoder.array.validity(), opt, offsets);

            if *width == 0 {
                return;
            }

            let mut child_offsets = Vec::with_capacity(offsets.len() * width);
            for (i, offset) in offsets.iter_mut().enumerate() {
                for j in 0..*width {
                    child_offsets.push(*offset);
                    *offset += array.widths.get((i * width) + j);
                }
            }
            encode_array(buffer, array.as_ref(), opt, &mut child_offsets, scratches);
            for (i, offset) in offsets.iter_mut().enumerate() {
                *offset = child_offsets[(i + 1) * width - 1];
            }
        },
        EncoderState::Struct(arrays) => {
            encode_validity(buffer, encoder.array.validity(), opt, offsets);

            for array in arrays {
                encode_array(buffer, array, opt, offsets, scratches);
            }
        },
    }
}

unsafe fn encode_validity(
    buffer: &mut [MaybeUninit<u8>],
    validity: Option<&Bitmap>,
    opt: RowEncodingOptions,
    row_starts: &mut [usize],
) {
    let null_sentinel = opt.null_sentinel();
    match validity {
        None => {
            for row_start in row_starts.iter_mut() {
                buffer[*row_start] = MaybeUninit::new(1);
                *row_start += 1;
            }
        },
        Some(validity) => {
            for (row_start, is_valid) in row_starts.iter_mut().zip(validity.iter()) {
                let v = if is_valid {
                    MaybeUninit::new(1)
                } else {
                    MaybeUninit::new(null_sentinel)
                };
                buffer[*row_start] = v;
                *row_start += 1;
            }
        },
    }
}

unsafe fn encode_primitive<T: NativeType + FixedLengthEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    arr: &PrimitiveArray<T>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    if arr.null_count() == 0 {
        crate::fixed::numeric::encode_slice(buffer, arr.values().as_slice(), opt, offsets)
    } else {
        crate::fixed::numeric::encode_iter(
            buffer,
            arr.into_iter().map(|v| v.copied()),
            opt,
            offsets,
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
        Boolean => 1,
        FixedSizeList(f, width) => 1 + width * fixed_size(f.dtype())?,
        Struct(fs) => {
            let mut sum = 0;
            for f in fs {
                sum += fixed_size(f.dtype())?;
            }
            1 + sum
        },
        Null => 0,
        _ => return None,
    })
}
