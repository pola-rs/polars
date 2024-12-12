use std::mem::MaybeUninit;

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeListArray,
    ListArray, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::Offset;

use crate::fixed::{boolean, decimal, numeric, packed_u32};
use crate::row::{RowEncodingOptions, RowsEncoded};
use crate::variable::{binary, no_order, utf8};
use crate::widths::RowWidths;
use crate::{with_match_arrow_primitive_type, ArrayRef, RowEncodingCatOrder};

pub fn convert_columns(
    num_rows: usize,
    columns: &[ArrayRef],
    opts: &[RowEncodingOptions],
    dicts: &[Option<RowEncodingCatOrder>],
) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized(
        num_rows,
        columns,
        opts.iter().copied().zip(dicts.iter().map(|v| v.as_ref())),
        &mut rows,
    );
    rows
}

pub fn convert_columns_no_order(
    num_rows: usize,
    columns: &[ArrayRef],
    dicts: &[Option<RowEncodingCatOrder>],
) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized_no_order(num_rows, columns, dicts, &mut rows);
    rows
}

pub fn convert_columns_amortized_no_order(
    num_rows: usize,
    columns: &[ArrayRef],
    dicts: &[Option<RowEncodingCatOrder>],
    rows: &mut RowsEncoded,
) {
    convert_columns_amortized(
        num_rows,
        columns,
        std::iter::repeat_n(RowEncodingOptions::default(), columns.len())
            .zip(dicts.iter().map(|v| v.as_ref())),
        rows,
    );
}

pub fn convert_columns_amortized<'a>(
    num_rows: usize,
    columns: &[ArrayRef],
    fields: impl IntoIterator<Item = (RowEncodingOptions, Option<&'a RowEncodingCatOrder>)> + Clone,
    rows: &mut RowsEncoded,
) {
    let mut masked_out_max_length = 0;
    let mut row_widths = RowWidths::new(num_rows);
    let mut encoders = columns
        .iter()
        .zip(fields.clone())
        .map(|(column, (opt, dicts))| {
            get_encoder(
                column.as_ref(),
                opt,
                dicts,
                &mut row_widths,
                &mut masked_out_max_length,
            )
        })
        .collect::<Vec<_>>();

    // Create an offsets array, we append 0 at the beginning here so it can serve as the final
    // offset array.
    let mut offsets = Vec::with_capacity(num_rows + 1);
    offsets.push(0);
    row_widths.extend_with_offsets(&mut offsets);

    // Create a buffer without initializing everything to zero.
    let total_num_bytes = row_widths.sum();
    let mut out = Vec::<u8>::with_capacity(total_num_bytes + masked_out_max_length);
    let buffer = &mut out.spare_capacity_mut()[..total_num_bytes + masked_out_max_length];

    let masked_out_write_offset = total_num_bytes;
    let mut scratches = EncodeScratches::default();
    for (encoder, (opt, dict)) in encoders.iter_mut().zip(fields) {
        unsafe {
            encode_array(
                buffer,
                encoder,
                opt,
                dict,
                &mut offsets[1..],
                masked_out_write_offset,
                &mut scratches,
            )
        };
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
    dicts: Option<&RowEncodingCatOrder>,
    row_widths: &mut RowWidths,
    masked_out_max_width: &mut usize,
) -> Encoder {
    let array = array.as_any().downcast_ref::<ListArray<O>>().unwrap();
    let array = array.trim_to_normalized_offsets_recursive();
    let values = array.values();

    let mut list_row_widths = RowWidths::new(values.len());
    let encoder = get_encoder(
        values.as_ref(),
        opt,
        dicts,
        &mut list_row_widths,
        masked_out_max_width,
    );

    match array.validity() {
        None => row_widths.push_iter(array.offsets().offset_and_length_iter().map(
            |(offset, length)| {
                let mut sum = 0;
                for i in offset..offset + length {
                    sum += list_row_widths.get(i);
                }
                1 + length + sum
            },
        )),
        Some(validity) => row_widths.push_iter(
            array
                .offsets()
                .offset_and_length_iter()
                .zip(validity.iter())
                .map(|((offset, length), is_valid)| {
                    if !is_valid {
                        if length > 0 {
                            for i in offset..offset + length {
                                *masked_out_max_width =
                                    (*masked_out_max_width).max(list_row_widths.get(i));
                            }
                        }
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
        array: array.boxed(),
        state: Some(Box::new(EncoderState::List(
            Box::new(encoder),
            list_row_widths,
        ))),
    }
}

fn biniter_num_column_bytes(
    array: &dyn Array,
    iter: impl ExactSizeIterator<Item = usize>,
    validity: Option<&Bitmap>,
    opt: RowEncodingOptions,
    row_widths: &mut RowWidths,
) -> Encoder {
    if opt.contains(RowEncodingOptions::NO_ORDER) {
        match validity {
            None => row_widths.push_iter(iter.map(|v| no_order::len_from_item(Some(v), opt))),
            Some(validity) => row_widths.push_iter(
                iter.zip(validity.iter())
                    .map(|(v, is_valid)| no_order::len_from_item(is_valid.then_some(v), opt)),
            ),
        }
    } else {
        match validity {
            None => row_widths.push_iter(
                iter.map(|v| crate::variable::binary::encoded_len_from_len(Some(v), opt)),
            ),
            Some(validity) => row_widths.push_iter(
                iter.zip(validity.iter())
                    .map(|(v, is_valid)| binary::encoded_len_from_len(is_valid.then_some(v), opt)),
            ),
        }
    };

    Encoder {
        array: array.to_boxed(),
        state: None,
    }
}

fn striter_num_column_bytes(
    array: &dyn Array,
    iter: impl ExactSizeIterator<Item = usize>,
    validity: Option<&Bitmap>,
    opt: RowEncodingOptions,
    row_widths: &mut RowWidths,
) -> Encoder {
    if opt.contains(RowEncodingOptions::NO_ORDER) {
        match validity {
            None => row_widths.push_iter(iter.map(|v| no_order::len_from_item(Some(v), opt))),
            Some(validity) => row_widths.push_iter(
                iter.zip(validity.iter())
                    .map(|(v, is_valid)| no_order::len_from_item(is_valid.then_some(v), opt)),
            ),
        }
    } else {
        match validity {
            None => row_widths
                .push_iter(iter.map(|v| crate::variable::utf8::len_from_item(Some(v), opt))),
            Some(validity) => row_widths.push_iter(
                iter.zip(validity.iter())
                    .map(|(v, is_valid)| utf8::len_from_item(is_valid.then_some(v), opt)),
            ),
        }
    };

    Encoder {
        array: array.to_boxed(),
        state: None,
    }
}

fn lexical_cat_num_column_bytes(
    keys: &PrimitiveArray<u32>,
    values: &Utf8ViewArray,
    _opt: RowEncodingOptions,
    row_widths: &mut RowWidths,
) -> Encoder {
    // If there are no values, that means that the dictionary array can only ever be `None`
    // so we don't encode that and just.
    if values.is_empty() {
        return Encoder {
            array: keys.to_boxed(),
            state: Some(Box::new(EncoderState::LexicalCategorical(Vec::new()))),
        };
    }

    let num_bits = values.len().next_power_of_two().trailing_zeros() as usize + 1;
    let idx_width = packed_u32::len_from_num_bits(num_bits);

    let values: Vec<&str> = values.values_iter().collect();
    let mut sort_idxs = (0..values.len() as u32).collect::<Vec<_>>();

    sort_idxs.sort_unstable_by_key(|&i| values[i as usize]);

    let mut offset = 0;
    for i in 1..values.len() {
        offset += u32::from(values[i - 1] == values[i]);
        sort_idxs[i] -= offset;
    }

    row_widths.push_constant(idx_width * 2);
    Encoder {
        array: keys.to_boxed(),
        state: Some(Box::new(EncoderState::LexicalCategorical(sort_idxs))),
    }
}

/// Get the encoder for a specific array.
fn get_encoder(
    array: &dyn Array,
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingCatOrder>,
    row_widths: &mut RowWidths,
    masked_out_max_width: &mut usize,
) -> Encoder {
    use ArrowDataType as D;
    let dtype = array.dtype();

    // Fast path: column has a fixed size encoding
    if let Some(size) = fixed_size(dtype, dict) {
        row_widths.push_constant(size);
        let state = match dtype {
            D::FixedSizeList(_, width) => {
                let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                let array = array.propagate_nulls();

                debug_assert_eq!(array.values().len(), array.len() * width);
                let mut nested_row_widths = RowWidths::new(array.values().len());
                let nested_encoder = get_encoder(
                    array.values().as_ref(),
                    opt,
                    dict,
                    &mut nested_row_widths,
                    masked_out_max_width,
                );
                Some(EncoderState::FixedSizeList(
                    Box::new(nested_encoder),
                    *width,
                    nested_row_widths,
                ))
            },
            D::Struct(_) => {
                let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
                let struct_array = struct_array.propagate_nulls();

                Some(EncoderState::Struct(match dict {
                    None => struct_array
                        .values()
                        .iter()
                        .map(|array| {
                            get_encoder(
                                array.as_ref(),
                                opt,
                                None,
                                &mut RowWidths::new(row_widths.num_rows()),
                                masked_out_max_width,
                            )
                        })
                        .collect(),
                    Some(RowEncodingCatOrder::Struct(dicts)) => struct_array
                        .values()
                        .iter()
                        .zip(dicts)
                        .map(|(array, dict)| {
                            get_encoder(
                                array.as_ref(),
                                opt,
                                dict.as_ref(),
                                &mut RowWidths::new(row_widths.num_rows()),
                                masked_out_max_width,
                            )
                        })
                        .collect(),
                    _ => unreachable!(),
                }))
            },
            _ => None,
        };

        let state = state.map(Box::new);
        return Encoder {
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
            let nested_encoder = get_encoder(
                array.values().as_ref(),
                opt,
                dict,
                &mut nested_row_widths,
                masked_out_max_width,
            );

            let mut fsl_row_widths = nested_row_widths.collapse_chunks(*width, array.len());
            fsl_row_widths.push_constant(1); // validity byte

            row_widths.push(&fsl_row_widths);
            Encoder {
                array: array.to_boxed(),
                state: Some(Box::new(EncoderState::FixedSizeList(
                    Box::new(nested_encoder),
                    *width,
                    nested_row_widths,
                ))),
            }
        },
        D::Struct(_) => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let array = array.propagate_nulls();

            let mut nested_encoders = Vec::with_capacity(array.values().len());
            row_widths.push_constant(1); // validity byte
            match dict {
                None => {
                    for array in array.values() {
                        let encoder = get_encoder(
                            array.as_ref(),
                            opt,
                            None,
                            row_widths,
                            masked_out_max_width,
                        );
                        nested_encoders.push(encoder);
                    }
                },
                Some(RowEncodingCatOrder::Struct(dicts)) => {
                    for (array, dict) in array.values().iter().zip(dicts) {
                        let encoder = get_encoder(
                            array.as_ref(),
                            opt,
                            dict.as_ref(),
                            row_widths,
                            masked_out_max_width,
                        );
                        nested_encoders.push(encoder);
                    }
                },
                _ => unreachable!(),
            }
            Encoder {
                array: array.to_boxed(),
                state: Some(Box::new(EncoderState::Struct(nested_encoders))),
            }
        },

        D::List(_) => {
            list_num_column_bytes::<i32>(array, opt, dict, row_widths, masked_out_max_width)
        },
        D::LargeList(_) => {
            list_num_column_bytes::<i64>(array, opt, dict, row_widths, masked_out_max_width)
        },

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

        // For some reason, lexical ordered categoricals have two possible underlying types. This
        // way we properly handle that.
        D::Dictionary(_, _, _) => {
            let Some(RowEncodingCatOrder::Lexical(values)) = dict else {
                unreachable!();
            };

            let dc_array = array
                .as_any()
                .downcast_ref::<DictionaryArray<u32>>()
                .unwrap();

            lexical_cat_num_column_bytes(dc_array.keys(), values, opt, row_widths)
        },
        D::UInt32 => {
            let Some(RowEncodingCatOrder::Lexical(values)) = dict else {
                unreachable!();
            };

            let dc_array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<u32>>()
                .unwrap();

            lexical_cat_num_column_bytes(dc_array, values, opt, row_widths)
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

struct Encoder {
    array: Box<dyn Array>,

    /// State contains nested encoders and extra information needed to encode.
    state: Option<Box<EncoderState>>,
}

enum EncoderState {
    List(Box<Encoder>, RowWidths),
    FixedSizeList(Box<Encoder>, usize, RowWidths),
    Struct(Vec<Encoder>),
    LexicalCategorical(Vec<u32>),
}

unsafe fn encode_strs<'a>(
    buffer: &mut [MaybeUninit<u8>],
    iter: impl Iterator<Item = Option<&'a str>>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    if opt.contains(RowEncodingOptions::NO_ORDER) {
        no_order::encode_variable_no_order(
            buffer,
            iter.map(|v| v.map(str::as_bytes)),
            opt,
            offsets,
        );
    } else {
        utf8::encode_str(buffer, iter, opt, offsets);
    }
}

unsafe fn encode_bins<'a>(
    buffer: &mut [MaybeUninit<u8>],
    iter: impl Iterator<Item = Option<&'a [u8]>>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    if opt.contains(RowEncodingOptions::NO_ORDER) {
        no_order::encode_variable_no_order(buffer, iter, opt, offsets);
    } else {
        binary::encode_iter(buffer, iter, opt, offsets);
    }
}

unsafe fn encode_flat_array(
    buffer: &mut [MaybeUninit<u8>],
    array: &dyn Array,
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingCatOrder>,
    offsets: &mut [usize],
) {
    use ArrowDataType as D;

    match array.dtype() {
        D::Null => {},
        D::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            boolean::encode_bool(buffer, array.iter(), opt, offsets);
        },

        // Needs to happen before numeric arm.
        D::Decimal(precision, _) => decimal::encode(
            buffer,
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap(),
            opt,
            offsets,
            *precision,
        ),

        dt if dt.is_numeric() => {
            if matches!(dt, D::UInt32) {
                if let Some(dict) = dict {
                    let keys = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<u32>>()
                        .unwrap();

                    match dict {
                        RowEncodingCatOrder::Physical(num_bits) => {
                            packed_u32::encode(buffer, keys, opt, offsets, *num_bits)
                        },
                        _ => unreachable!(),
                    }
                    return;
                }
            }

            with_match_arrow_primitive_type!(dt, |$T| {
                let array = array.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                numeric::encode(buffer, array, opt, offsets);
            })
        },

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

        // Lexical ordered Categorical are casted to PrimitiveArray above.
        D::Dictionary(_, _, _) => todo!(),

        D::FixedSizeBinary(_) => todo!(),
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
    dict: Option<&RowEncodingCatOrder>,
    offsets: &mut [usize],
    masked_out_write_offset: usize, // Masked out values need to be written somewhere. We just
    // reserved space at the end and tell all values to write
    // there.
    scratches: &mut EncodeScratches,
) {
    let Some(state) = &encoder.state else {
        // This is actually the main path.
        //
        // If no nested types or special types are needed, this path is taken.
        return encode_flat_array(buffer, encoder.array.as_ref(), opt, dict, offsets);
    };

    match state.as_ref() {
        EncoderState::List(nested_encoder, nested_row_widths) => {
            // @TODO: make more general.
            let array = encoder
                .array
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap();

            scratches.clear();

            scratches
                .nested_offsets
                .reserve(nested_row_widths.num_rows());
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
                            offsets[i] += nested_row_widths.get(j);
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

                            // Values might have been masked out.
                            if length > 0 {
                                nested_offsets
                                    .extend(std::iter::repeat_n(masked_out_write_offset, length));
                            }

                            continue;
                        }

                        for j in offset..offset + length {
                            buffer[offsets[i]] = MaybeUninit::new(list_continuation_token);
                            offsets[i] += 1;

                            nested_offsets.push(offsets[i]);
                            offsets[i] += nested_row_widths.get(j);
                        }
                        buffer[offsets[i]] = MaybeUninit::new(list_termination_token);
                        offsets[i] += 1;
                    }
                },
            }

            unsafe {
                encode_array(
                    buffer,
                    nested_encoder,
                    opt,
                    dict,
                    nested_offsets,
                    masked_out_write_offset,
                    &mut EncodeScratches::default(),
                )
            };
        },
        EncoderState::FixedSizeList(array, width, nested_row_widths) => {
            encode_validity(buffer, encoder.array.validity(), opt, offsets);

            if *width == 0 {
                return;
            }

            let mut child_offsets = Vec::with_capacity(offsets.len() * width);
            for (i, offset) in offsets.iter_mut().enumerate() {
                for j in 0..*width {
                    child_offsets.push(*offset);
                    *offset += nested_row_widths.get((i * width) + j);
                }
            }
            encode_array(
                buffer,
                array.as_ref(),
                opt,
                dict,
                &mut child_offsets,
                masked_out_write_offset,
                scratches,
            );
            for (i, offset) in offsets.iter_mut().enumerate() {
                *offset = child_offsets[(i + 1) * width - 1];
            }
        },
        EncoderState::Struct(arrays) => {
            encode_validity(buffer, encoder.array.validity(), opt, offsets);

            match dict {
                None => {
                    for array in arrays {
                        encode_array(
                            buffer,
                            array,
                            opt,
                            None,
                            offsets,
                            masked_out_write_offset,
                            scratches,
                        );
                    }
                },
                Some(RowEncodingCatOrder::Struct(dicts)) => {
                    for (array, dict) in arrays.iter().zip(dicts) {
                        encode_array(
                            buffer,
                            array,
                            opt,
                            dict.as_ref(),
                            offsets,
                            masked_out_write_offset,
                            scratches,
                        );
                    }
                },
                _ => unreachable!(),
            }
        },
        EncoderState::LexicalCategorical(sort_idxs) => {
            // All values are null
            if sort_idxs.is_empty() {
                return;
            }

            let keys = encoder
                .array
                .as_any()
                .downcast_ref::<PrimitiveArray<u32>>()
                .unwrap();

            let num_bits = sort_idxs.len().next_power_of_two().trailing_zeros() as usize + 1;
            packed_u32::encode_iter(
                buffer,
                keys.iter().map(|k| k.map(|&k| sort_idxs[k as usize])),
                opt,
                offsets,
                num_bits,
            );
            packed_u32::encode_slice(buffer, keys.values(), opt, offsets, num_bits);
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

pub fn fixed_size(dtype: &ArrowDataType, dict: Option<&RowEncodingCatOrder>) -> Option<usize> {
    use numeric::FixedLengthEncoding;
    use ArrowDataType as D;
    Some(match dtype {
        D::Null => 0,
        D::Boolean => 1,

        D::UInt8 => u8::ENCODED_LEN,
        D::UInt16 => u16::ENCODED_LEN,
        D::UInt32 => match dict {
            None => u32::ENCODED_LEN,
            Some(RowEncodingCatOrder::Physical(num_bits)) => {
                packed_u32::len_from_num_bits(*num_bits)
            },
            _ => return None,
        },
        D::UInt64 => u64::ENCODED_LEN,

        D::Int8 => i8::ENCODED_LEN,
        D::Int16 => i16::ENCODED_LEN,
        D::Int32 => i32::ENCODED_LEN,
        D::Int64 => i64::ENCODED_LEN,

        D::Decimal(precision, _) => decimal::len_from_precision(*precision),
        D::Float32 => f32::ENCODED_LEN,
        D::Float64 => f64::ENCODED_LEN,
        D::FixedSizeList(f, width) => 1 + width * fixed_size(f.dtype(), dict)?,
        D::Struct(fs) => match dict {
            None => {
                let mut sum = 0;
                for f in fs {
                    sum += fixed_size(f.dtype(), None)?;
                }
                1 + sum
            },
            Some(RowEncodingCatOrder::Struct(dicts)) => {
                let mut sum = 0;
                for (f, dict) in fs.iter().zip(dicts) {
                    sum += fixed_size(f.dtype(), dict.as_ref())?;
                }
                1 + sum
            },
            _ => unreachable!(),
        },
        _ => return None,
    })
}
