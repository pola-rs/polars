use arrow::array::{Array, BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray, View};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::{ArrowDataType, PhysicalType};

use super::dictionary_encoded::{append_validity, constrain_page_validity};
use super::utils::{
    dict_indices_decoder, filter_from_range, freeze_validity, unspecialized_decode,
};
use super::{dictionary_encoded, Filter, PredicateFilter};
use crate::parquet::encoding::{delta_byte_array, delta_length_byte_array, hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::{self};

mod optional;
mod optional_masked;
mod predicate;
mod required;
mod required_masked;

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, BitmapBuilder);

impl<'a> utils::StateTranslation<'a, BinViewDecoder> for StateTranslation<'a> {
    type PlainDecoder = BinaryIter<'a>;

    fn new(
        _decoder: &BinViewDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinViewDecoder as utils::Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                let values = BinaryIter::new(values, page.num_values());

                Ok(Self::Plain(values))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(_)) => {
                let values =
                    dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))?;
                Ok(Self::Dictionary(values))
            },
            (Encoding::DeltaLengthByteArray, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::DeltaLengthByteArray(
                    delta_length_byte_array::Decoder::try_new(values)?,
                    Vec::new(),
                ))
            },
            (Encoding::DeltaByteArray, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::DeltaBytes(delta_byte_array::Decoder::try_new(
                    values,
                )?))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn num_rows(&self) -> usize {
        match self {
            StateTranslation::Plain(i) => i.max_num_values,
            StateTranslation::Dictionary(i) => i.len(),
            StateTranslation::DeltaLengthByteArray(i, _) => i.len(),
            StateTranslation::DeltaBytes(i) => i.len(),
        }
    }
}

pub(crate) struct BinViewDecoder {
    pub is_string: bool,
}

impl BinViewDecoder {
    pub fn new_string() -> Self {
        Self { is_string: true }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    DeltaLengthByteArray(delta_length_byte_array::Decoder<'a>, Vec<u32>),
    DeltaBytes(delta_byte_array::Decoder<'a>),
}

impl utils::Decoded for DecodedStateTuple {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn extend_nulls(&mut self, n: usize) {
        self.0.extend_constant(n, Some(&[]));
        self.1.extend_constant(n, false);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_plain(
    values: &[u8],
    max_num_values: usize,
    target: &mut MutableBinaryViewArray<[u8]>,

    is_optional: bool,
    validity: &mut BitmapBuilder,

    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,

    pred_true_mask: &mut BitmapBuilder,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if is_optional {
        append_validity(page_validity, filter.as_ref(), validity, max_num_values);
    }
    let page_validity = constrain_page_validity(max_num_values, page_validity, filter.as_ref());

    match (filter, page_validity) {
        (None, None) => required::decode(max_num_values, values, None, target, verify_utf8),
        (Some(Filter::Range(rng)), None) if rng.start == 0 => {
            required::decode(max_num_values, values, Some(rng.end), target, verify_utf8)
        },
        (None, Some(page_validity)) => optional::decode(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            verify_utf8,
        ),
        (Some(Filter::Range(rng)), Some(page_validity)) if rng.start == 0 => optional::decode(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            verify_utf8,
        ),
        (Some(Filter::Mask(mask)), None) => {
            required_masked::decode(max_num_values, values, target, &mask, verify_utf8)
        },
        (Some(Filter::Mask(mask)), Some(page_validity)) => optional_masked::decode(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            &mask,
            verify_utf8,
        ),
        (Some(Filter::Range(rng)), None) => required_masked::decode(
            max_num_values,
            values,
            target,
            &filter_from_range(rng.clone()),
            verify_utf8,
        ),
        (Some(Filter::Range(rng)), Some(page_validity)) => optional_masked::decode(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            &filter_from_range(rng.clone()),
            verify_utf8,
        ),
        (Some(Filter::Predicate(p)), page_validity) => {
            let Some(needle) = p.predicate.to_equals_scalar() else {
                unreachable!();
            };

            if needle.is_null() || page_validity.is_some() {
                todo!();
            }

            let needle = if verify_utf8 {
                needle.as_str().unwrap().as_bytes()
            } else {
                needle.as_binary().unwrap()
            };

            let start_pred_true_num = pred_true_mask.set_bits();
            predicate::decode_equals(max_num_values, values, needle, pred_true_mask)?;

            if p.include_values {
                let pred_true_num = pred_true_mask.set_bits() - start_pred_true_num;

                if pred_true_num > 0 {
                    let new_target_len = target.len() + pred_true_num;
                    let new_total_bytes_len =
                        target.total_bytes_len() + pred_true_num * needle.len();

                    target.push_value(needle);
                    let view = *target.views().last().unwrap();

                    // SAFETY: We know that the view is valid since we added it safely and we
                    // update the total_bytes_len afterwards. The total_buffer_len is not affected.
                    unsafe {
                        target.views_mut().resize(new_target_len, view);
                        target.set_total_bytes_len(new_total_bytes_len);
                    }
                }
            }

            Ok(())
        },
    }?;

    Ok(())
}

#[cold]
fn invalid_input_err() -> ParquetError {
    ParquetError::oos("String data does not match given length")
}

#[cold]
fn invalid_utf8_err() -> ParquetError {
    ParquetError::oos("String data contained invalid UTF-8")
}

pub fn decode_plain_generic(
    values: &[u8],
    target: &mut MutableBinaryViewArray<[u8]>,

    num_rows: usize,
    mut next: impl FnMut() -> Option<(bool, bool)>,

    verify_utf8: bool,
) -> ParquetResult<()> {
    // Since the offset in the buffer is decided by the interleaved lengths, every value has to be
    // walked no matter what. This makes decoding rather inefficient in general.
    //
    // There are three cases:
    // 1. All inlinable values
    //    - Most time is spend in decoding
    //    - No additional buffer has to be formed
    //    - Possible UTF-8 verification is fast because the len_below_128 trick
    // 2. All non-inlinable values
    //    - Little time is spend in decoding
    //    - Most time is spend in buffer memcopying (we remove the interleaved lengths)
    //    - Possible UTF-8 verification is fast because the continuation byte trick
    // 3. Mixed inlinable and non-inlinable values
    //    - Time shared between decoding and buffer forming
    //    - UTF-8 verification might still use len_below_128 trick, but might need to fall back to
    //      slow path.

    target.finish_in_progress();
    unsafe { target.views_mut() }.reserve(num_rows);

    let buffer_idx = target.completed_buffers().len() as u32;
    let mut buffer = Vec::with_capacity(values.len() + 1);
    let mut none_starting_with_continuation_byte = true; // Whether the transition from between strings is valid
                                                         // UTF-8
    let mut all_len_below_128 = true; // Whether all the lengths of the values are below 128, this
                                      // allows us to make UTF-8 verification a lot faster.

    let mut total_bytes_len = 0;
    let mut num_seen = 0;
    let mut num_inlined = 0;

    let mut mvalues = values;
    while let Some((is_valid, is_selected)) = next() {
        if !is_valid {
            if is_selected {
                unsafe { target.views_mut() }.push(unsafe { View::new_inline_unchecked(&[]) });
            }
            continue;
        }

        if mvalues.len() < 4 {
            return Err(invalid_input_err());
        }

        let length;
        (length, mvalues) = mvalues.split_at(4);
        let length: &[u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(*length);

        if mvalues.len() < length as usize {
            return Err(invalid_input_err());
        }

        let value;
        (value, mvalues) = mvalues.split_at(length as usize);

        num_seen += 1;
        all_len_below_128 &= value.len() < 128;
        // Everything starting with 10.. .... is a continuation byte.
        none_starting_with_continuation_byte &=
            value.is_empty() || value[0] & 0b1100_0000 != 0b1000_0000;

        if !is_selected {
            continue;
        }

        let offset = buffer.len() as u32;

        if value.len() <= View::MAX_INLINE_SIZE as usize {
            unsafe { target.views_mut() }.push(unsafe { View::new_inline_unchecked(value) });
            num_inlined += 1;
        } else {
            buffer.extend_from_slice(value);
            unsafe { target.views_mut() }
                .push(unsafe { View::new_noninline_unchecked(value, buffer_idx, offset) });
        }

        total_bytes_len += value.len();
    }

    unsafe {
        target.set_total_bytes_len(target.total_bytes_len() + total_bytes_len);
    }

    if verify_utf8 {
        // This is a trick that allows us to check the resulting buffer which allows to batch the
        // UTF-8 verification.
        //
        // This is allowed if none of the strings start with a UTF-8 continuation byte, so we keep
        // track of that during the decoding.
        if num_inlined == 0 {
            if !none_starting_with_continuation_byte || simdutf8::basic::from_utf8(&buffer).is_err()
            {
                return Err(invalid_utf8_err());
            }

        // This is a small trick that allows us to check the Parquet buffer instead of the view
        // buffer. Batching the UTF-8 verification is more performant. For this to be allowed,
        // all the interleaved lengths need to be valid UTF-8.
        //
        // Every strings prepended by 4 bytes (L, 0, 0, 0), since we check here L < 128. L is
        // only a valid first byte of a UTF-8 code-point and (L, 0, 0, 0) is valid UTF-8.
        // Consequently, it is valid to just check the whole buffer.
        } else if all_len_below_128 {
            if simdutf8::basic::from_utf8(&values[..values.len() - mvalues.len()]).is_err() {
                return Err(invalid_utf8_err());
            }
        } else {
            // We check all the non-inlined values here.
            if !none_starting_with_continuation_byte || simdutf8::basic::from_utf8(&buffer).is_err()
            {
                return Err(invalid_utf8_err());
            }

            let mut all_inlined_are_ascii = true;

            // @NOTE: This is only valid because we initialize our inline View's to be zeroes on
            // non-included bytes.
            for view in &target.views()[target.len() - num_seen..] {
                all_inlined_are_ascii &= (view.length > View::MAX_INLINE_SIZE)
                    | (view.as_u128() & 0x0000_0000_8080_8080_8080_8080_8080_8080 == 0);
            }

            // This is the very slow path.
            if !all_inlined_are_ascii {
                let mut is_valid = true;
                for view in &target.views()[target.len() - num_seen..] {
                    if view.length <= View::MAX_INLINE_SIZE {
                        is_valid &=
                            std::str::from_utf8(unsafe { view.get_inlined_slice_unchecked() })
                                .is_ok();
                    }
                }

                if !is_valid {
                    return Err(invalid_utf8_err());
                }
            }
        }
    }

    target.push_buffer(buffer.into());

    Ok(())
}

impl utils::Decoder for BinViewDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = BinaryViewArray;
    type DecodedState = DecodedStateTuple;
    type Output = Box<dyn Array>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBinaryViewArray::with_capacity(capacity),
            BitmapBuilder::with_capacity(capacity),
        )
    }

    fn evaluate_dict_predicate(
        &self,
        dict: &Self::Dict,
        predicate: &PredicateFilter,
    ) -> ParquetResult<Bitmap> {
        let utf8_array;
        let mut dict_arr = dict as &dyn Array;

        if self.is_string {
            utf8_array = unsafe { dict.to_utf8view_unchecked() };
            dict_arr = &utf8_array
        }

        Ok(predicate.predicate.evaluate(dict_arr))
    }

    fn has_predicate_specialization(
        &self,
        state: &utils::State<'_, Self>,
        predicate: &PredicateFilter,
    ) -> ParquetResult<bool> {
        let mut has_predicate_specialization = false;

        has_predicate_specialization |=
            matches!(state.translation, StateTranslation::Dictionary(_));
        has_predicate_specialization |= matches!(state.translation, StateTranslation::Plain(_))
            && predicate.predicate.to_equals_scalar().is_some();

        // @TODO: This should be implemented
        has_predicate_specialization &= state.page_validity.is_none();

        Ok(has_predicate_specialization)
    }

    fn apply_dictionary(
        &mut self,
        (values, _): &mut Self::DecodedState,
        dict: &Self::Dict,
    ) -> ParquetResult<()> {
        if values.completed_buffers().len() < dict.data_buffers().len() {
            for buffer in dict.data_buffers().as_ref() {
                values.push_buffer(buffer.clone());
            }
        }

        assert!(values.completed_buffers().len() == dict.data_buffers().len());

        Ok(())
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let values = &page.buffer;
        let num_values = page.num_values;

        let mut arr = MutableBinaryViewArray::new();
        required::decode(num_values, values, None, &mut arr, self.is_string)?;

        Ok(arr.freeze())
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn Array,
        is_optional: bool,
    ) -> ParquetResult<()> {
        let is_utf8 = self.is_string;
        if is_utf8 {
            let array = additional.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            let mut array = array.to_binview();

            if let Some(validity) = array.take_validity() {
                decoded.0.extend_from_array(&array);
                decoded.1.extend_from_bitmap(&validity);
            } else {
                decoded.0.extend_from_array(&array);
                if is_optional {
                    decoded.1.extend_constant(array.len(), true);
                }
            }
        } else {
            let array = additional
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap();
            let mut array = array.clone();

            if let Some(validity) = array.take_validity() {
                decoded.0.extend_from_array(&array);
                decoded.1.extend_from_bitmap(&validity);
            } else {
                decoded.0.extend_from_array(&array);
                if is_optional {
                    decoded.1.extend_constant(array.len(), true);
                }
            }
        }

        Ok(())
    }

    fn extend_filtered_with_state(
        &mut self,
        mut state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
        filter: Option<super::Filter>,
    ) -> ParquetResult<()> {
        match state.translation {
            StateTranslation::Plain(iter) => decode_plain(
                iter.values,
                iter.max_num_values,
                &mut decoded.0,
                state.is_optional,
                &mut decoded.1,
                state.page_validity.as_ref(),
                filter,
                pred_true_mask,
                self.is_string,
            ),
            StateTranslation::Dictionary(ref mut indexes) => {
                let dict = state.dict.unwrap();

                let start_length = decoded.0.views().len();

                dictionary_encoded::decode_dict(
                    indexes.clone(),
                    dict.views().as_slice(),
                    state.dict_mask,
                    state.is_optional,
                    state.page_validity.as_ref(),
                    filter,
                    &mut decoded.1,
                    unsafe { decoded.0.views_mut() },
                    pred_true_mask,
                )?;

                let total_length: usize = decoded
                    .0
                    .views()
                    .iter()
                    .skip(start_length)
                    .map(|view| view.length as usize)
                    .sum();
                unsafe {
                    decoded
                        .0
                        .set_total_bytes_len(decoded.0.total_bytes_len() + total_length);
                }

                Ok(())
            },
            StateTranslation::DeltaLengthByteArray(decoder, _vec) => {
                let values = decoder.values;
                let lengths = decoder.lengths.collect::<Vec<i64>>()?;

                if self.is_string {
                    let mut none_starting_with_continuation_byte = true;
                    let mut offset = 0;
                    for length in &lengths {
                        none_starting_with_continuation_byte &=
                            *length == 0 || values[offset] & 0xC0 != 0x80;
                        offset += *length as usize;
                    }

                    if !none_starting_with_continuation_byte {
                        return Err(invalid_utf8_err());
                    }

                    if simdutf8::basic::from_utf8(&values[..offset]).is_err() {
                        return Err(invalid_utf8_err());
                    }
                }

                let mut i = 0;
                let mut offset = 0;
                unspecialized_decode(
                    lengths.len(),
                    || {
                        let length = lengths[i] as usize;

                        let value = &values[offset..offset + length];

                        i += 1;
                        offset += length;

                        Ok(value)
                    },
                    filter,
                    state.page_validity,
                    state.is_optional,
                    &mut decoded.1,
                    &mut decoded.0,
                )
            },
            StateTranslation::DeltaBytes(mut decoder) => {
                let check_utf8 = self.is_string;

                unspecialized_decode(
                    decoder.len(),
                    || {
                        let value = decoder.next().unwrap()?;

                        if check_utf8 && simdutf8::basic::from_utf8(&value[..]).is_err() {
                            return Err(invalid_utf8_err());
                        }

                        Ok(value)
                    },
                    filter,
                    state.page_validity,
                    state.is_optional,
                    &mut decoded.1,
                    &mut decoded.0,
                )
            },
        }
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        let mut array: BinaryViewArray = values.freeze();

        let validity = freeze_validity(validity);
        array = array.with_validity(validity);

        match dtype.to_physical_type() {
            PhysicalType::BinaryView => Ok(array.boxed()),
            PhysicalType::Utf8View => {
                // SAFETY: we already checked utf8
                unsafe {
                    Ok(Utf8ViewArray::new_unchecked(
                        dtype,
                        array.views().clone(),
                        array.data_buffers().clone(),
                        array.validity().cloned(),
                        array.total_bytes_len(),
                        array.total_buffer_len(),
                    )
                    .boxed())
                }
            },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct BinaryIter<'a> {
    values: &'a [u8],

    /// A maximum number of items that this [`BinaryIter`] may produce.
    ///
    /// This equal the length of the iterator i.f.f. the data encoded by the [`BinaryIter`] is not
    /// nullable.
    max_num_values: usize,
}

impl<'a> BinaryIter<'a> {
    pub fn new(values: &'a [u8], max_num_values: usize) -> Self {
        Self {
            values,
            max_num_values,
        }
    }
}

impl<'a> Iterator for BinaryIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.max_num_values == 0 {
            assert!(self.values.is_empty());
            return None;
        }

        let (length, remaining) = self.values.split_at(4);
        let length: [u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(length) as usize;
        let (result, remaining) = remaining.split_at(length);
        self.max_num_values -= 1;
        self.values = remaining;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.max_num_values))
    }
}
