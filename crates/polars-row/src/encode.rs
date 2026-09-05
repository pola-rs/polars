#![allow(unsafe_op_in_unsafe_fn)]
use std::mem::MaybeUninit;

use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::types::{NativeType, PrimitiveType};
use polars_array::{
    PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmapRef, PlBooleanArray,
    PlFixedSizeListArray, PlListArray, PlPrimitiveArray, PlStructArray, PlUtf8ViewArray,
};
use polars_dtype::categorical::CatNative;
use polars_utils::float16::pf16;

use crate::fixed::numeric::FixedLengthEncoding;
use crate::fixed::{boolean, decimal, numeric};
use crate::row::{RowEncodingOptions, RowsEncoded};
use crate::variable::{binary, no_order, utf8};
use crate::widths::RowWidths;
use crate::{RowEncodingCategoricalContext, RowEncodingContext, with_match_pl_primitive_type};

/// Downcasts an array whose [`PlArrayType`] has already been matched on.
///
/// # Panics
/// Panics if `array` is not an `A`, which its array type rules out.
#[inline]
fn downcast<A: PlArray>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type identifies the concrete array")
}

pub fn convert_columns(
    num_rows: usize,
    columns: &[Box<dyn PlArray>],
    opts: &[RowEncodingOptions],
    dicts: &[Option<RowEncodingContext>],
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
    columns: &[Box<dyn PlArray>],
    dicts: &[Option<RowEncodingContext>],
) -> RowsEncoded {
    let mut rows = RowsEncoded::new(vec![], vec![]);
    convert_columns_amortized_no_order(num_rows, columns, dicts, &mut rows);
    rows
}

pub fn convert_columns_amortized_no_order(
    num_rows: usize,
    columns: &[Box<dyn PlArray>],
    dicts: &[Option<RowEncodingContext>],
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
    columns: &[Box<dyn PlArray>],
    fields: impl IntoIterator<Item = (RowEncodingOptions, Option<&'a RowEncodingContext>)> + Clone,
    rows: &mut RowsEncoded,
) {
    let mut masked_out_max_length = 0;
    let mut row_widths = RowWidths::new(num_rows);
    let mut encoders = columns
        .iter()
        .zip(fields.clone())
        .map(|(column, (opt, dicts))| {
            assert_eq!(column.len(), num_rows);
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

/// The range of the values array every element of `array` covers.
///
/// Offsets that hold the single range every element covers are read as that one range, repeated.
fn value_ranges(array: &PlListArray) -> impl ExactSizeIterator<Item = std::ops::Range<usize>> {
    // SAFETY: every index is below the length the iterator counts up to.
    (0..array.len()).map(|i| unsafe { array.value_range_unchecked(i) })
}

fn list_num_column_bytes(
    array: &dyn PlArray,
    opt: RowEncodingOptions,
    dicts: Option<&RowEncodingContext>,
    row_widths: &mut RowWidths,
    masked_out_max_width: &mut usize,
) -> Encoder {
    let array = downcast::<PlListArray>(array);
    let values = array.values();

    let mut list_row_widths = RowWidths::new(values.len());
    let encoder = get_encoder(
        values,
        opt.into_nested(),
        dicts,
        &mut list_row_widths,
        masked_out_max_width,
    );

    match array.validity() {
        None => row_widths.push_iter(value_ranges(array).map(|range| {
            let length = range.len();
            let mut sum = 0;
            for i in range {
                sum += list_row_widths.get(i);
            }
            1 + length + sum
        })),
        Some(validity) => row_widths.push_iter(value_ranges(array).zip(validity.iter()).map(
            |(range, is_valid)| {
                let length = range.len();
                if !is_valid {
                    if length > 0 {
                        for i in range {
                            *masked_out_max_width =
                                (*masked_out_max_width).max(list_row_widths.get(i));
                        }
                    }
                    return 1;
                }

                let mut sum = 0;
                for i in range {
                    sum += list_row_widths.get(i);
                }
                1 + length + sum
            },
        )),
    };

    Encoder {
        array: array.to_boxed(),
        state: Some(Box::new(EncoderState::List(
            Box::new(encoder),
            list_row_widths,
        ))),
    }
}

fn biniter_num_column_bytes(
    array: &dyn PlArray,
    iter: impl ExactSizeIterator<Item = usize>,
    validity: Option<PlBitmapRef<'_>>,
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
    array: &dyn PlArray,
    iter: impl ExactSizeIterator<Item = usize>,
    validity: Option<PlBitmapRef<'_>>,
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

/// The array written out, if it holds one child that every one of its elements shares.
///
/// The leaf encoders read a scalar buffer as the one value it stands for, without writing it out.
/// A nested array's child is indexed per row, though, and so is [`RowWidths`]: a [`PlListArray`]
/// whose offsets hold the single range every element covers, or a [`PlFixedSizeListArray`] whose
/// values hold the single list every element is, has to be written out before it is encoded.
// TODO(polars-array): read a shared child in place instead, the way the leaves read a scalar
// buffer. Until then this is what the Arrow export used to do for every array, not just these.
fn write_out_shared_child(array: &dyn PlArray) -> Option<Box<dyn PlArray>> {
    match array.array_type() {
        PlArrayType::List => {
            let array = downcast::<PlListArray>(array);
            (!array.offsets_are_flat())
                .then(|| Box::new(array.to_flat().into_owned().into_array()) as Box<dyn PlArray>)
        },
        PlArrayType::FixedSizeList => {
            let array = downcast::<PlFixedSizeListArray>(array);
            (!array.values_are_flat())
                .then(|| Box::new(array.to_flat().into_owned().into_array()) as Box<dyn PlArray>)
        },
        // Every field of a struct array holds one element per row whatever its mask looks like, and
        // the remaining array types have no child.
        _ => None,
    }
}

/// Get the encoder for a specific array.
///
/// The array carries no logical type of its own, so this dispatches on its [`PlArrayType`] and
/// reads the widths and children off the array itself; `dict` carries what the physical
/// representation does not say — a decimal's precision, a categorical's mapping.
fn get_encoder(
    array: &dyn PlArray,
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingContext>,
    row_widths: &mut RowWidths,
    masked_out_max_width: &mut usize,
) -> Encoder {
    use PlArrayType as A;

    if let Some(array) = write_out_shared_child(array) {
        return get_encoder(&*array, opt, dict, row_widths, masked_out_max_width);
    }

    let array_type = array.array_type();

    // Fast path: column has a fixed size encoding
    if let Some(size) = fixed_size_of_array(array, opt, dict) {
        row_widths.push_constant(size);
        let state = match array_type {
            A::FixedSizeList => {
                let dc_array = downcast::<PlFixedSizeListArray>(array);
                let width = dc_array.width();

                debug_assert_eq!(dc_array.values().len(), dc_array.len() * width);
                let mut nested_row_widths = RowWidths::new(dc_array.values().len());
                let nested_encoder = get_encoder(
                    dc_array.values(),
                    opt.into_nested(),
                    dict,
                    &mut nested_row_widths,
                    masked_out_max_width,
                );
                Some(EncoderState::FixedSizeList(
                    Box::new(nested_encoder),
                    width,
                    nested_row_widths,
                ))
            },
            A::Struct => {
                let struct_array = downcast::<PlStructArray>(array);

                Some(EncoderState::Struct(match dict {
                    None => struct_array
                        .fields()
                        .iter()
                        .map(|array| {
                            get_encoder(
                                &**array,
                                opt.into_nested(),
                                None,
                                &mut RowWidths::new(row_widths.num_rows()),
                                masked_out_max_width,
                            )
                        })
                        .collect(),
                    Some(RowEncodingContext::Struct(dicts)) => struct_array
                        .fields()
                        .iter()
                        .zip(dicts)
                        .map(|(array, dict)| {
                            get_encoder(
                                &**array,
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

    // Non-fixed-size categorical path.
    if let Some(RowEncodingContext::Categorical(ctx)) = dict {
        /// The width of the string each category key stands for, which is what is encoded.
        macro_rules! cat_str_lengths {
            ($T:ty) => {{
                assert!(opt.is_ordered() && !ctx.is_enum);
                let dc_array = downcast::<PlPrimitiveArray<$T>>(array);
                return striter_num_column_bytes(
                    array,
                    dc_array.values_iter().map(|cat| {
                        ctx.mapping
                            .cat_to_str(cat.as_cat())
                            .map(|s| s.len())
                            .unwrap_or(0)
                    }),
                    dc_array.validity(),
                    opt,
                    row_widths,
                );
            }};
        }

        match array_type {
            A::Primitive(PrimitiveType::UInt8) => cat_str_lengths!(u8),
            A::Primitive(PrimitiveType::UInt16) => cat_str_lengths!(u16),
            A::Primitive(PrimitiveType::UInt32) => cat_str_lengths!(u32),
            _ => {
                // Fall through to below, should be nested type containing categorical.
                debug_assert!(matches!(array_type, A::Struct | A::List | A::FixedSizeList))
            },
        }
    }

    match array_type {
        A::FixedSizeList => {
            let array = downcast::<PlFixedSizeListArray>(array);
            let width = array.width();

            debug_assert_eq!(array.values().len(), array.len() * width);
            let mut nested_row_widths = RowWidths::new(array.values().len());
            let nested_encoder = get_encoder(
                array.values(),
                opt.into_nested(),
                dict,
                &mut nested_row_widths,
                masked_out_max_width,
            );

            let mut fsl_row_widths = nested_row_widths.collapse_chunks(width, array.len());
            fsl_row_widths.push_constant(1); // validity byte

            row_widths.push(&fsl_row_widths);
            Encoder {
                array: array.to_boxed(),
                state: Some(Box::new(EncoderState::FixedSizeList(
                    Box::new(nested_encoder),
                    width,
                    nested_row_widths,
                ))),
            }
        },
        A::Struct => {
            let array = downcast::<PlStructArray>(array);

            let mut nested_encoders = Vec::with_capacity(array.fields().len());
            row_widths.push_constant(1); // validity byte
            match dict {
                None => {
                    for array in array.fields() {
                        let encoder = get_encoder(
                            &**array,
                            opt.into_nested(),
                            None,
                            row_widths,
                            masked_out_max_width,
                        );
                        nested_encoders.push(encoder);
                    }
                },
                Some(RowEncodingContext::Struct(dicts)) => {
                    for (array, dict) in array.fields().iter().zip(dicts) {
                        let encoder = get_encoder(
                            &**array,
                            opt.into_nested(),
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

        A::List => list_num_column_bytes(array, opt, dict, row_widths, masked_out_max_width),

        A::BinaryView => {
            let dc_array = downcast::<PlBinaryViewArray>(array);
            biniter_num_column_bytes(
                array,
                view_lengths(dc_array),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },
        A::Binary => {
            let dc_array = downcast::<PlBinaryArray>(array);
            biniter_num_column_bytes(
                array,
                value_lengths(dc_array),
                dc_array.validity(),
                opt,
                row_widths,
            )
        },

        A::Utf8View => {
            let dc_array = downcast::<PlUtf8ViewArray>(array);
            striter_num_column_bytes(
                array,
                view_lengths(dc_array.as_binview()),
                dc_array.as_binview().validity(),
                opt,
                row_widths,
            )
        },

        A::FixedSizeBinary => unreachable!(),
        A::Object { .. } => unreachable!(),

        // Should be fixed size type
        A::Null | A::Boolean | A::Primitive(_) => unreachable!(),
    }
}

/// The number of bytes every element of `array` holds, read off the views.
fn view_lengths(array: &PlBinaryViewArray) -> impl ExactSizeIterator<Item = usize> {
    // SAFETY: every index is below the length the iterator counts up to.
    (0..array.len()).map(|i| unsafe { array.view_unchecked(i) }.length as usize)
}

/// The number of bytes every element of `array` holds, read off the offsets.
fn value_lengths(array: &PlBinaryArray) -> impl ExactSizeIterator<Item = usize> {
    // SAFETY: every index is below the length the iterator counts up to.
    (0..array.len()).map(|i| unsafe { array.value_length_unchecked(i) })
}

struct Encoder {
    array: Box<dyn PlArray>,

    /// State contains nested encoders and extra information needed to encode.
    state: Option<Box<EncoderState>>,
}

enum EncoderState {
    List(Box<Encoder>, RowWidths),
    FixedSizeList(Box<Encoder>, usize, RowWidths),
    Struct(Vec<Encoder>),
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

unsafe fn encode_cat_array<T: NativeType + FixedLengthEncoding + CatNative>(
    buffer: &mut [MaybeUninit<u8>],
    keys: &PlPrimitiveArray<T>,
    opt: RowEncodingOptions,
    ctx: &RowEncodingCategoricalContext,
    offsets: &mut [usize],
) {
    if ctx.is_enum || !opt.is_ordered() {
        numeric::encode(buffer, keys, opt, offsets);
    } else {
        utf8::encode_str(
            buffer,
            keys.iter()
                .map(|k| k.map(|cat| ctx.mapping.cat_to_str_unchecked(cat.as_cat()))),
            opt,
            offsets,
        );
    }
}

unsafe fn encode_flat_array(
    buffer: &mut [MaybeUninit<u8>],
    array: &dyn PlArray,
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingContext>,
    offsets: &mut [usize],
) {
    use PlArrayType as A;
    let array_type = array.array_type();

    if let Some(RowEncodingContext::Categorical(ctx)) = dict {
        match array_type {
            A::Primitive(PrimitiveType::UInt8) => encode_cat_array(
                buffer,
                downcast::<PlPrimitiveArray<u8>>(array),
                opt,
                ctx,
                offsets,
            ),
            A::Primitive(PrimitiveType::UInt16) => encode_cat_array(
                buffer,
                downcast::<PlPrimitiveArray<u16>>(array),
                opt,
                ctx,
                offsets,
            ),
            A::Primitive(PrimitiveType::UInt32) => encode_cat_array(
                buffer,
                downcast::<PlPrimitiveArray<u32>>(array),
                opt,
                ctx,
                offsets,
            ),
            _ => unreachable!(),
        };
        return;
    }

    match array_type {
        A::Null => {},
        A::Boolean => {
            let array = downcast::<PlBooleanArray>(array);
            boolean::encode_bool(buffer, array.iter(), opt, offsets);
        },

        A::Primitive(primitive) => {
            if primitive == PrimitiveType::Int128 {
                if let Some(RowEncodingContext::Decimal(precision)) = dict {
                    decimal::encode(
                        buffer,
                        downcast::<PlPrimitiveArray<i128>>(array),
                        opt,
                        offsets,
                        *precision,
                    );
                    return;
                }
            }

            with_match_pl_primitive_type!(primitive, |$T| {
                numeric::encode(buffer, downcast::<PlPrimitiveArray<$T>>(array), opt, offsets);
            })
        },

        A::Binary => {
            let array = downcast::<PlBinaryArray>(array);
            encode_bins(buffer, array.iter(), opt, offsets);
        },
        A::BinaryView => {
            let array = downcast::<PlBinaryViewArray>(array);
            encode_bins(buffer, array.iter(), opt, offsets);
        },
        A::Utf8View => {
            let array = downcast::<PlUtf8ViewArray>(array);
            encode_strs(buffer, array.iter(), opt, offsets);
        },

        A::FixedSizeBinary => todo!(),
        A::Object { .. } => todo!(),

        // Handled by the encoder's state, which holds the nested encoders.
        A::Struct | A::List | A::FixedSizeList => unreachable!(),
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
    dict: Option<&RowEncodingContext>,
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
            let array = downcast::<PlListArray>(encoder.array.as_ref());

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
                    for (i, range) in value_ranges(array).enumerate() {
                        for j in range {
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
                    for (i, (range, is_valid)) in
                        value_ranges(array).zip(validity.iter()).enumerate()
                    {
                        if !is_valid {
                            buffer[offsets[i]] = MaybeUninit::new(list_null_sentinel);
                            offsets[i] += 1;

                            // Values might have been masked out.
                            if !range.is_empty() {
                                nested_offsets.extend(std::iter::repeat_n(
                                    masked_out_write_offset,
                                    range.len(),
                                ));
                            }

                            continue;
                        }

                        for j in range {
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
                    opt.into_nested(),
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
                opt.into_nested(),
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
                            opt.into_nested(),
                            None,
                            offsets,
                            masked_out_write_offset,
                            scratches,
                        );
                    }
                },
                Some(RowEncodingContext::Struct(dicts)) => {
                    for (array, dict) in arrays.iter().zip(dicts) {
                        encode_array(
                            buffer,
                            array,
                            opt.into_nested(),
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
    }
}

unsafe fn encode_validity(
    buffer: &mut [MaybeUninit<u8>],
    validity: Option<PlBitmapRef<'_>>,
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

/// The width the row encoding of one value of `primitive` takes, if it has one.
///
/// A decimal is the one case where `dict` decides the width: the precision bounds how many bytes a
/// value needs.
fn fixed_size_primitive(
    primitive: PrimitiveType,
    dict: Option<&RowEncodingContext>,
) -> Option<usize> {
    use PrimitiveType as P;
    use numeric::FixedLengthEncoding;

    Some(match primitive {
        P::UInt8 => u8::ENCODED_LEN,
        P::UInt16 => u16::ENCODED_LEN,
        P::UInt32 => u32::ENCODED_LEN,
        P::UInt64 => u64::ENCODED_LEN,
        P::UInt128 => u128::ENCODED_LEN,

        P::Int8 => i8::ENCODED_LEN,
        P::Int16 => i16::ENCODED_LEN,
        P::Int32 => i32::ENCODED_LEN,
        P::Int64 => i64::ENCODED_LEN,
        P::Int128 => match dict {
            None => i128::ENCODED_LEN,
            Some(RowEncodingContext::Decimal(precision)) => decimal::len_from_precision(*precision),
            _ => unreachable!(),
        },

        P::Float16 => pf16::ENCODED_LEN,
        P::Float32 => f32::ENCODED_LEN,
        P::Float64 => f64::ENCODED_LEN,

        P::Int256 | P::DaysMs | P::MonthDayNano | P::MonthDayMillis => return None,
    })
}

/// Whether `dict` makes the encoding variable-width whatever the representation says.
///
/// An ordered categorical that is not an enum encodes the string each key stands for, and those
/// have no common width.
fn dict_is_variable_width(opt: RowEncodingOptions, dict: Option<&RowEncodingContext>) -> bool {
    matches!(dict, Some(RowEncodingContext::Categorical(ctx)) if !ctx.is_enum && opt.is_ordered())
}

/// [`fixed_size`] for an array, whose children and width are read off the array itself.
fn fixed_size_of_array(
    array: &dyn PlArray,
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingContext>,
) -> Option<usize> {
    use PlArrayType as A;

    if dict_is_variable_width(opt, dict) {
        return None;
    }

    Some(match array.array_type() {
        A::Null => 0,
        A::Boolean => 1,
        A::Primitive(primitive) => fixed_size_primitive(primitive, dict)?,
        A::FixedSizeList => {
            let array = downcast::<PlFixedSizeListArray>(array);
            1 + array.width() * fixed_size_of_array(array.values(), opt, dict)?
        },
        A::Struct => {
            let fields = downcast::<PlStructArray>(array).fields();
            let mut sum = 0;
            match dict {
                None => {
                    for field in fields {
                        sum += fixed_size_of_array(&**field, opt, None)?;
                    }
                },
                Some(RowEncodingContext::Struct(dicts)) => {
                    for (field, dict) in fields.iter().zip(dicts) {
                        sum += fixed_size_of_array(&**field, opt, dict.as_ref())?;
                    }
                },
                _ => unreachable!(),
            }
            1 + sum
        },
        _ => return None,
    })
}

/// The width the row encoding of one value of `dtype` takes, if it has one.
///
/// This is the decoder's side of [`fixed_size_of_array`]: a decode is driven by the type it is
/// asked for, since the array it produces does not carry one.
pub fn fixed_size(
    dtype: &ArrowDataType,
    opt: RowEncodingOptions,
    dict: Option<&RowEncodingContext>,
) -> Option<usize> {
    use ArrowDataType as D;

    if dict_is_variable_width(opt, dict) {
        return None;
    }

    Some(match dtype {
        D::Null => 0,
        D::Boolean => 1,

        D::UInt8
        | D::UInt16
        | D::UInt32
        | D::UInt64
        | D::UInt128
        | D::Int8
        | D::Int16
        | D::Int32
        | D::Int64
        | D::Int128
        | D::Float16
        | D::Float32
        | D::Float64 => {
            let PhysicalType::Primitive(primitive) = dtype.to_physical_type() else {
                unreachable!("every arm above is a primitive")
            };
            fixed_size_primitive(primitive, dict)?
        },

        D::FixedSizeList(f, width) => 1 + width * fixed_size(f.dtype(), opt, dict)?,
        D::Struct(fs) => match dict {
            None => {
                let mut sum = 0;
                for f in fs {
                    sum += fixed_size(f.dtype(), opt, None)?;
                }
                1 + sum
            },
            Some(RowEncodingContext::Struct(dicts)) => {
                let mut sum = 0;
                for (f, dict) in fs.iter().zip(dicts) {
                    sum += fixed_size(f.dtype(), opt, dict.as_ref())?;
                }
                1 + sum
            },
            _ => unreachable!(),
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;
    use arrow::array::proptest::{
        ArrayArbitraryOptions, ArrowDataTypeArbitraryOptions, ArrowDataTypeArbitrarySelection,
        array_with_options,
    };
    use polars_array::PlNullArray;

    use super::*;

    proptest::prop_compose! {
        fn arrays
            ()
            (length in 0..100usize)
            (arrays in proptest::collection::vec(array_with_options(length, ArrayArbitraryOptions {
                dtype: ArrowDataTypeArbitraryOptions {
                    allowed_dtypes: ArrowDataTypeArbitrarySelection::all() & !ArrowDataTypeArbitrarySelection::BINARY,
                    ..Default::default()
                }
            }), 1..3))
        -> Vec<Box<dyn Array>> {
            arrays
        }
    }

    proptest::proptest! {
        /// The arrays are generated as Arrow ones because that is where the generator lives; the
        /// import is a buffer handover, so what the encoder sees is the same data.
        #[test]
        fn test_encode_arrays
            (arrays in arrays())
         {
            let arrays = arrays
                .iter()
                .map(|array| polars_array::arrow::import::from_arrow(&**array))
                .collect::<Vec<_>>();
            let dicts: Vec<Option<RowEncodingContext>> = (0..arrays.len()).map(|_| None).collect();
            convert_columns_no_order(arrays[0].len(), &arrays, &dicts);
        }
    }

    /// The rows one column of `array` encodes to, under both sort orders.
    fn rows_of(array: Box<dyn PlArray>) -> Vec<Vec<u8>> {
        let length = array.len();
        let columns = [array];
        let dicts = [None];

        [
            RowEncodingOptions::new_unsorted(),
            RowEncodingOptions::new_sorted(false, false),
            RowEncodingOptions::new_sorted(true, true),
        ]
        .into_iter()
        .flat_map(|opt| {
            convert_columns(length, &columns, &[opt], &dicts)
                .iter()
                .map(<[u8]>::to_vec)
                .collect::<Vec<_>>()
        })
        .collect()
    }

    /// A logical array and the same one written out, which have to encode to the same rows.
    #[track_caller]
    fn assert_representations_agree(scalar: Box<dyn PlArray>, flat: Box<dyn PlArray>) {
        assert_eq!(scalar.len(), flat.len());
        assert!(
            scalar.eq_dyn(&*flat),
            "the two forms must hold the same elements"
        );
        assert_eq!(rows_of(scalar), rows_of(flat));
    }

    /// A scalar buffer is read as the one value it stands for, so it has to encode to the same
    /// rows as the array that holds that value once per element.
    #[test]
    fn scalar_and_flat_arrays_encode_alike() {
        const LENGTH: usize = 5;

        assert_representations_agree(
            Box::new(PlPrimitiveArray::new_scalar(7i64, LENGTH)),
            Box::new(PlPrimitiveArray::from_vec(vec![7i64; LENGTH])),
        );
        assert_representations_agree(
            Box::new(PlBooleanArray::new_scalar(true, LENGTH)),
            Box::new(PlBooleanArray::from_values(
                std::iter::repeat_n(true, LENGTH).collect(),
            )),
        );
        assert_representations_agree(
            Box::new(PlUtf8ViewArray::new_scalar("scalar", LENGTH)),
            Box::new(std::iter::repeat_n(Some("scalar"), LENGTH).collect::<PlUtf8ViewArray>()),
        );
        assert_representations_agree(
            Box::new(PlBinaryArray::new_scalar(b"scalar", LENGTH)),
            Box::new(PlBinaryArray::from_values_iter(std::iter::repeat_n(
                b"scalar", LENGTH,
            ))),
        );

        // A scalar nested array shares one child, which is written out before it is encoded.
        let element = || Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])) as Box<dyn PlArray>;
        assert_representations_agree(
            Box::new(PlFixedSizeListArray::new_scalar(element(), LENGTH)),
            Box::new(PlFixedSizeListArray::new(
                Box::new(PlPrimitiveArray::from_vec([1i32, 2].repeat(LENGTH))),
                2,
                LENGTH,
                None,
            )),
        );
        assert_representations_agree(
            Box::new(PlListArray::new_scalar(element(), LENGTH)),
            Box::new(PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec([1i32, 2].repeat(LENGTH))),
                (0..=LENGTH as u64).map(|i| i * 2).collect(),
            )),
        );

        // A struct array's own buffer is only its mask, and every field holds one element per row.
        assert_representations_agree(
            Box::new(PlStructArray::new(
                vec![Box::new(PlPrimitiveArray::new_scalar(7i64, LENGTH))],
                LENGTH,
                None,
            )),
            Box::new(PlStructArray::new(
                vec![Box::new(PlPrimitiveArray::from_vec(vec![7i64; LENGTH]))],
                LENGTH,
                None,
            )),
        );
    }

    /// The mask of a scalar array stands for the bit every element shares.
    #[test]
    fn scalar_and_flat_masks_encode_alike() {
        const LENGTH: usize = 4;

        assert_representations_agree(
            Box::new(PlPrimitiveArray::<i64>::new_full_null(LENGTH)),
            Box::new(std::iter::repeat_n(None, LENGTH).collect::<PlPrimitiveArray<i64>>()),
        );
        assert_representations_agree(
            Box::new(PlUtf8ViewArray::new_full_null(LENGTH)),
            Box::new(std::iter::repeat_n(None, LENGTH).collect::<PlUtf8ViewArray>()),
        );
        assert_representations_agree(
            Box::new(PlNullArray::new(LENGTH)),
            Box::new(PlNullArray::new(LENGTH)),
        );
    }
}
