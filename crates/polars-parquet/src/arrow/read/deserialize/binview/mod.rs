use arrow::array::{Array, BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray, View};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use polars_utils::aliases::PlIndexSet;

use super::dictionary_encoded::{append_validity, constrain_page_validity};
use super::utils::{
    dict_indices_decoder, filter_from_range, freeze_validity, unspecialized_decode,
};
use super::{Filter, PredicateFilter, dictionary_encoded};
use crate::parquet::encoding::{Encoding, delta_byte_array, delta_length_byte_array, hybrid_rle};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, DictPage, split_buffer};
use crate::read::deserialize::utils::{self, Decoded};
use crate::read::expr::{ParquetScalar, SpecializedParquetColumnExpr};

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

enum EqualsOneOfState {
    Empty,
    Inlinable([View; 4]),
    Set(PlIndexSet<Box<[u8]>>, Vec<View>),
}

pub(crate) struct BinViewDecoder {
    is_string: bool,
    equals_one_of_state: Option<Box<EqualsOneOfState>>,
}

impl BinViewDecoder {
    pub fn new(is_string: bool) -> Self {
        Self {
            is_string,
            equals_one_of_state: None,
        }
    }

    pub fn new_string() -> Self {
        Self::new(true)
    }

    fn initialize_predicate_equals_one_of_state(
        &mut self,
        needles: &[ParquetScalar],
    ) -> &EqualsOneOfState {
        self.equals_one_of_state.get_or_insert_with(|| {
            if needles.is_empty() {
                return Box::new(EqualsOneOfState::Empty);
            }

            let is_inlinable = needles.len() <= 4
                && needles.iter().all(|needle| {
                    let needle = if self.is_string {
                        needle.as_str().unwrap().as_bytes()
                    } else {
                        needle.as_binary().unwrap()
                    };
                    needle.len() < View::MAX_INLINE_SIZE as usize
                });

            Box::new(if is_inlinable {
                let mut views = [View::default(); 4];
                for (i, needle) in needles.iter().enumerate() {
                    let needle = if self.is_string {
                        needle.as_str().unwrap().as_bytes()
                    } else {
                        needle.as_binary().unwrap()
                    };
                    views[i] = View::new_inline(needle);
                }
                for i in needles.len()..4 {
                    views[i] = views[0];
                }
                EqualsOneOfState::Inlinable(views)
            } else {
                let mut needle_set = PlIndexSet::<Box<_>>::default();
                needle_set.extend(needles.iter().map(|needle| {
                    assert!(!needle.is_null());
                    let needle = if self.is_string {
                        needle.as_str().unwrap().as_bytes()
                    } else {
                        needle.as_binary().unwrap()
                    };
                    needle.into()
                }));
                EqualsOneOfState::Set(needle_set, Vec::new())
            })
        })
    }

    fn initialize_decode_equals_one_of_state(
        &mut self,
        target: &mut MutableBinaryViewArray<[u8]>,
    ) -> Option<&EqualsOneOfState> {
        if let Some(equals_one_of_state) = self.equals_one_of_state.as_deref_mut() {
            match equals_one_of_state {
                EqualsOneOfState::Set(needles, views) => {
                    if needles.len() != views.len() {
                        assert!(views.is_empty());
                        views.extend(
                            needles
                                .iter()
                                .map(|needle| target.push_value_into_buffer(needle)),
                        );
                    }
                },
                EqualsOneOfState::Empty | EqualsOneOfState::Inlinable(_) => {},
            }
        }
        self.equals_one_of_state.as_deref()
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

    fn remaining_capacity(&self) -> usize {
        (self.0.capacity() - self.0.len()).min(self.1.capacity() - self.1.len())
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_plain(
    values: &[u8],
    max_num_values: usize,
    target: &mut MutableBinaryViewArray<[u8]>,

    is_optional: bool,
    validity: &mut BitmapBuilder,

    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,

    equals_one_of_state: Option<&EqualsOneOfState>,
    verify_utf8: bool,
) -> ParquetResult<()> {
    if is_optional {
        append_validity(page_validity, filter.as_ref(), validity, max_num_values);
    }

    if let Some(equals_one_of_state) = equals_one_of_state {
        assert!(page_validity.is_none());

        let mut total_bytes_len = 0;
        match equals_one_of_state {
            EqualsOneOfState::Empty => {},
            EqualsOneOfState::Inlinable(views) => {
                predicate::decode_is_in_inlinable(
                    max_num_values,
                    values,
                    views,
                    unsafe { target.views_mut() },
                    &mut total_bytes_len,
                )?;
            },
            EqualsOneOfState::Set(needles, views) => {
                predicate::decode_is_in_non_inlinable(
                    max_num_values,
                    values,
                    needles,
                    views,
                    unsafe { target.views_mut() },
                    &mut total_bytes_len,
                )?;
            },
        }

        let new_total_bytes_len = target.total_bytes_len() + total_bytes_len;

        // SAFETY: We know that the view is valid since we added it safely and we
        // update the total_bytes_len afterwards. The total_buffer_len is not affected.
        unsafe {
            target.set_total_bytes_len(new_total_bytes_len);
        }

        return Ok(());
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
            &filter_from_range(rng),
            verify_utf8,
        ),
        (Some(Filter::Range(rng)), Some(page_validity)) => optional_masked::decode(
            page_validity.set_bits(),
            values,
            target,
            &page_validity,
            &filter_from_range(rng),
            verify_utf8,
        ),
        (Some(Filter::Predicate(_)), _) => unreachable!(),
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

    let start_target_length = target.len();

    let buffer_idx = target.completed_buffers().len() as u32;
    let mut buffer = Vec::with_capacity(values.len() + 1);
    let mut none_starting_with_continuation_byte = true; // Whether the transition from between strings is valid
    // UTF-8
    let mut all_len_below_128 = true; // Whether all the lengths of the values are below 128, this
    // allows us to make UTF-8 verification a lot faster.

    let mut total_bytes_len = 0;
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
            for view in &target.views()[start_target_length..] {
                all_inlined_are_ascii &= (view.length > View::MAX_INLINE_SIZE)
                    | (view.as_u128() & 0x0000_0000_8080_8080_8080_8080_8080_8080 == 0);
            }

            // This is the very slow path.
            if !all_inlined_are_ascii {
                let mut is_valid = true;
                for view in &target.views()[start_target_length..] {
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

    fn evaluate_predicate(
        &mut self,
        state: &utils::State<'_, Self>,
        predicate: Option<&SpecializedParquetColumnExpr>,
        pred_true_mask: &mut BitmapBuilder,
        dict_mask: Option<&Bitmap>,
    ) -> ParquetResult<bool> {
        if state.page_validity.is_some() {
            // @Performance. This should be implemented.
            return Ok(false);
        }

        if let StateTranslation::Dictionary(values) = &state.translation {
            let dict_mask = dict_mask.unwrap();
            super::dictionary_encoded::predicate::decode(
                values.clone(),
                dict_mask,
                pred_true_mask,
            )?;
            return Ok(true);
        }

        let Some(predicate) = predicate else {
            return Ok(false);
        };

        use {SpecializedParquetColumnExpr as Spce, StateTranslation as St};
        match (&state.translation, predicate) {
            (St::Plain(iter), Spce::Equal(needle)) => {
                assert!(!needle.is_null());

                let needle = if self.is_string {
                    needle.as_str().unwrap().as_bytes()
                } else {
                    needle.as_binary().unwrap()
                };
                predicate::decode_equals(iter.max_num_values, iter.values, needle, pred_true_mask)?;
            },
            (St::Plain(iter), Spce::EqualOneOf(needles)) => {
                let e = self.initialize_predicate_equals_one_of_state(needles);

                match e {
                    EqualsOneOfState::Empty => {
                        pred_true_mask.extend_constant(iter.max_num_values, false)
                    },
                    EqualsOneOfState::Inlinable(views) => {
                        predicate::decode_is_in_no_values_inlinable(
                            iter.max_num_values,
                            iter.values,
                            views,
                            pred_true_mask,
                        )?;
                    },
                    EqualsOneOfState::Set(needle_set, _) => {
                        predicate::decode_is_in_no_values_non_inlinable(
                            iter.max_num_values,
                            iter.values,
                            needle_set,
                            pred_true_mask,
                        )?;
                    },
                }
            },
            _ => return Ok(false),
        }

        Ok(true)
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
        filter: Option<super::Filter>,
    ) -> ParquetResult<()> {
        let is_string = self.is_string;
        let equals_one_of_state = self.initialize_decode_equals_one_of_state(&mut decoded.0);
        match state.translation {
            StateTranslation::Plain(iter) => decode_plain(
                iter.values,
                iter.max_num_values,
                &mut decoded.0,
                state.is_optional,
                &mut decoded.1,
                state.page_validity.as_ref(),
                filter,
                equals_one_of_state,
                is_string,
            ),
            StateTranslation::Dictionary(ref mut indexes) => {
                let dict = state.dict.unwrap();

                let start_length = decoded.0.views().len();

                dictionary_encoded::decode_dict(
                    indexes.clone(),
                    dict.views().as_slice(),
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

    fn extend_constant(
        &mut self,
        decoded: &mut Self::DecodedState,
        length: usize,
        value: &ParquetScalar,
    ) -> ParquetResult<()> {
        if value.is_null() {
            decoded.extend_nulls(length);
            return Ok(());
        }

        let value = match value {
            ParquetScalar::String(v) => v.as_bytes(),
            ParquetScalar::Binary(v) => v.as_ref(),
            _ => unreachable!(),
        };

        decoded.0.extend_constant(length, Some(value));
        decoded.1.extend_constant(length, true);

        Ok(())
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
