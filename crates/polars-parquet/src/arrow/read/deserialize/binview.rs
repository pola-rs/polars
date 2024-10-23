use std::sync::atomic::{AtomicBool, Ordering};

use arrow::array::{
    Array, BinaryViewArray, DictionaryArray, DictionaryKey, MutableBinaryViewArray, PrimitiveArray,
    Utf8ViewArray, View,
};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::Buffer;
use arrow::datatypes::{ArrowDataType, PhysicalType};

use super::utils::dict_encoded::{append_validity, constrain_page_validity};
use super::utils::{
    dict_indices_decoder, filter_from_range, freeze_validity, unspecialized_decode,
};
use super::Filter;
use crate::parquet::encoding::{delta_byte_array, delta_length_byte_array, hybrid_rle, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::read::deserialize::utils::{self};
use crate::read::PrimitiveLogicalType;

type DecodedStateTuple = (MutableBinaryViewArray<[u8]>, MutableBitmap);

impl<'a> utils::StateTranslation<'a, BinViewDecoder> for StateTranslation<'a> {
    type PlainDecoder = BinaryIter<'a>;

    fn new(
        decoder: &BinViewDecoder,
        page: &'a DataPage,
        dict: Option<&'a <BinViewDecoder as utils::Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        decoder.check_utf8.store(is_string, Ordering::Relaxed);
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
}

#[derive(Default)]
pub(crate) struct BinViewDecoder {
    check_utf8: AtomicBool,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(BinaryIter<'a>),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    DeltaLengthByteArray(delta_length_byte_array::Decoder<'a>, Vec<u32>),
    DeltaBytes(delta_byte_array::Decoder<'a>),
}

impl utils::ExactSize for DecodedStateTuple {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl utils::ExactSize for (Vec<View>, Vec<Buffer<u8>>) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_plain(
    values: &[u8],
    max_num_values: usize,
    target: &mut MutableBinaryViewArray<[u8]>,

    is_optional: bool,
    validity: &mut MutableBitmap,

    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if is_optional {
        append_validity(page_validity, filter.as_ref(), validity, max_num_values);
    }
    let page_validity = constrain_page_validity(max_num_values, page_validity, filter.as_ref());

    match (filter, page_validity) {
        (None, None) => decode_required_plain(max_num_values, values, None, target, verify_utf8),
        (Some(Filter::Range(rng)), None) if rng.start == 0 => {
            decode_required_plain(max_num_values, values, Some(rng.end), target, verify_utf8)
        },
        (None, Some(page_validity)) => decode_optional_plain(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            verify_utf8,
        ),
        (Some(Filter::Range(rng)), Some(page_validity)) if rng.start == 0 => decode_optional_plain(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            verify_utf8,
        ),
        (Some(Filter::Mask(mask)), None) => {
            decode_masked_required_plain(max_num_values, values, target, &mask, verify_utf8)
        },
        (Some(Filter::Mask(mask)), Some(page_validity)) => decode_masked_optional_plain(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            &mask,
            verify_utf8,
        ),
        (Some(Filter::Range(rng)), None) => decode_masked_required_plain(
            max_num_values,
            values,
            target,
            &filter_from_range(rng.clone()),
            verify_utf8,
        ),
        (Some(Filter::Range(rng)), Some(page_validity)) => decode_masked_optional_plain(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            &filter_from_range(rng.clone()),
            verify_utf8,
        ),
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

fn decode_required_plain(
    num_expected_values: usize,
    values: &[u8],
    limit: Option<usize>,
    target: &mut MutableBinaryViewArray<[u8]>,

    verify_utf8: bool,
) -> ParquetResult<()> {
    let limit = limit.unwrap_or(num_expected_values);

    let mut idx = 0;
    decode_plain_generic(
        values,
        target,
        limit,
        || {
            if idx >= limit {
                return None;
            }

            idx += 1;

            Some((true, true))
        },
        verify_utf8,
    )
}

fn decode_optional_plain(
    num_expected_values: usize,
    values: &[u8],
    target: &mut MutableBinaryViewArray<[u8]>,
    page_validity: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if page_validity.unset_bits() == 0 {
        return decode_required_plain(
            num_expected_values,
            values,
            Some(page_validity.len()),
            target,
            verify_utf8,
        );
    }

    let mut validity_iter = page_validity.iter();
    decode_plain_generic(
        values,
        target,
        page_validity.len(),
        || Some((validity_iter.next()?, true)),
        verify_utf8,
    )
}

fn decode_masked_required_plain(
    num_expected_values: usize,
    values: &[u8],
    target: &mut MutableBinaryViewArray<[u8]>,

    mask: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    if mask.unset_bits() == 0 {
        return decode_required_plain(
            num_expected_values,
            values,
            Some(mask.len()),
            target,
            verify_utf8,
        );
    }

    let mut mask_iter = mask.iter();
    decode_plain_generic(
        values,
        target,
        mask.set_bits(),
        || Some((true, mask_iter.next()?)),
        verify_utf8,
    )
}

fn decode_masked_optional_plain(
    num_expected_values: usize,
    values: &[u8],
    target: &mut MutableBinaryViewArray<[u8]>,

    page_validity: &Bitmap,
    mask: &Bitmap,

    verify_utf8: bool,
) -> ParquetResult<()> {
    assert_eq!(page_validity.len(), mask.len());

    if mask.unset_bits() == 0 {
        return decode_optional_plain(
            num_expected_values,
            values,
            target,
            page_validity,
            verify_utf8,
        );
    }

    if page_validity.unset_bits() == 0 {
        return decode_masked_required_plain(
            num_expected_values,
            values,
            target,
            page_validity,
            verify_utf8,
        );
    }

    let mut validity_iter = page_validity.iter();
    let mut mask_iter = mask.iter();
    decode_plain_generic(
        values,
        target,
        mask.set_bits(),
        || Some((validity_iter.next()?, mask_iter.next()?)),
        verify_utf8,
    )
}

pub fn decode_plain_generic(
    values: &[u8],
    target: &mut MutableBinaryViewArray<[u8]>,

    num_rows: usize,
    mut next: impl FnMut() -> Option<(bool, bool)>,

    verify_utf8: bool,
) -> ParquetResult<()> {
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
        if num_inlined == 0 {
            if !none_starting_with_continuation_byte || simdutf8::basic::from_utf8(&buffer).is_err()
            {
                return Err(invalid_utf8_err());
            }
        } else if all_len_below_128 {
            if simdutf8::basic::from_utf8(values).is_err() {
                return Err(invalid_utf8_err());
            }
        } else {
            if !none_starting_with_continuation_byte || simdutf8::basic::from_utf8(&buffer).is_err()
            {
                return Err(invalid_utf8_err());
            }

            let mut all_inlined_are_ascii = true;

            const MASK: [u128; 2] = [
                0x0000_0000_8080_8080_8080_8080_8080_8080,
                0x0000_0000_0000_0000_0000_0000_0000_0000,
            ];

            for view in &target.views()[target.len() - num_seen..] {
                all_inlined_are_ascii &=
                    view.as_u128() & MASK[usize::from(view.length > View::MAX_INLINE_SIZE)] == 0;
            }

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
    type Dict = (Vec<View>, Vec<Buffer<u8>>);
    type DecodedState = DecodedStateTuple;
    type Output = Box<dyn Array>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBinaryViewArray::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn apply_dictionary(
        &mut self,
        (values, _): &mut Self::DecodedState,
        dict: &Self::Dict,
    ) -> ParquetResult<()> {
        if values.completed_buffers().len() < dict.1.len() {
            for buffer in &dict.1 {
                values.push_buffer(buffer.clone());
            }
        }

        assert!(values.completed_buffers().len() == dict.1.len());

        Ok(())
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let values = &page.buffer;
        let num_values = page.num_values;

        let mut arr = MutableBinaryViewArray::new();
        decode_required_plain(
            num_values,
            values,
            None,
            &mut arr,
            self.check_utf8.load(Ordering::Relaxed),
        )?;

        let (views, buffers) = arr.take();

        Ok((views, buffers))
    }

    fn extend_filtered_with_state(
        &mut self,
        mut state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
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
                self.check_utf8.load(Ordering::Relaxed),
            ),
            StateTranslation::Dictionary(ref mut indexes) => {
                let (dict, _) = state.dict.unwrap();

                let start_length = decoded.0.views().len();

                utils::dict_encoded::decode_dict(
                    indexes.clone(),
                    dict,
                    state.is_optional,
                    state.page_validity.as_ref(),
                    filter,
                    &mut decoded.1,
                    unsafe { decoded.0.views_mut() },
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

                if self.check_utf8.load(Ordering::Relaxed) {
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
                let check_utf8 = self.check_utf8.load(Ordering::Relaxed);

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

impl utils::DictDecodable for BinViewDecoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        dtype: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_dtype = match &dtype {
            ArrowDataType::Dictionary(_, values, _) => values.as_ref().clone(),
            _ => dtype.clone(),
        };

        let mut view_dict = MutableBinaryViewArray::with_capacity(dict.0.len());
        for buffer in dict.1 {
            view_dict.push_buffer(buffer);
        }
        unsafe { view_dict.views_mut().extend(dict.0.iter()) };
        unsafe { view_dict.set_total_bytes_len(dict.0.iter().map(|v| v.length as usize).sum()) };
        let view_dict = view_dict.freeze();

        let dict = match value_dtype.to_physical_type() {
            PhysicalType::Utf8View => view_dict.to_utf8view().unwrap().boxed(),
            PhysicalType::BinaryView => view_dict.boxed(),
            _ => unreachable!(),
        };

        Ok(DictionaryArray::try_new(dtype, keys, dict).unwrap())
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
