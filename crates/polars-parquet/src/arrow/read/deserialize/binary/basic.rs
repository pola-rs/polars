use std::cell::Cell;
use std::collections::VecDeque;
use std::default::Default;

use arrow::array::specification::try_check_utf8;
use arrow::array::{Array, BinaryArray, MutableBinaryValuesArray, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::Offset;
use polars_error::PolarsResult;

use super::super::utils::{
    extend_from_decoder, get_selected_rows, next, DecodedState, FilteredOptionalPageValidity,
    MaybeNext, OptionalPageValidity,
};
use super::super::{utils, PagesIter};
use super::utils::*;
use crate::parquet::deserialize::SliceFilteredIter;
use crate::parquet::encoding::{delta_bitpacked, delta_length_byte_array, hybrid_rle, Encoding};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;
use crate::read::{ParquetError, PrimitiveLogicalType};

#[derive(Debug)]
pub(super) struct Required<'a> {
    pub values: std::iter::Take<BinaryIter<'a>>,
}

impl<'a> Required<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let (_, _, values) = split_buffer(page)?;
        let values = BinaryIter::new(values).take(page.num_values());

        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct Delta<'a> {
    pub lengths: std::vec::IntoIter<usize>,
    pub values: &'a [u8],
}

impl<'a> Delta<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let (_, _, values) = split_buffer(page)?;

        let mut lengths_iter = delta_length_byte_array::Decoder::try_new(values)?;

        #[allow(clippy::needless_collect)] // we need to consume it to get the values
        let lengths = lengths_iter
            .by_ref()
            .map(|x| x.map(|x| x as usize))
            .collect::<Result<Vec<_>, ParquetError>>()?;

        let values = lengths_iter.into_values();
        Ok(Self {
            lengths: lengths.into_iter(),
            values,
        })
    }

    pub fn len(&self) -> usize {
        self.lengths.size_hint().0
    }
}

impl<'a> Iterator for Delta<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let length = self.lengths.next()?;
        let (item, remaining) = self.values.split_at(length);
        self.values = remaining;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.lengths.size_hint()
    }
}

#[derive(Debug)]
pub(super) struct DeltaBytes<'a> {
    prefix: std::vec::IntoIter<i32>,
    suffix: std::vec::IntoIter<i32>,
    data: &'a [u8],
    data_offset: usize,
    last_value: Vec<u8>,
}

impl<'a> DeltaBytes<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let (_, _, values) = split_buffer(page)?;
        let mut decoder = delta_bitpacked::Decoder::try_new(values)?;
        let prefix = (&mut decoder)
            .take(page.num_values())
            .map(|r| r.map(|v| v as i32).unwrap())
            .collect::<Vec<_>>();

        let mut data_offset = decoder.consumed_bytes();
        let mut decoder = delta_bitpacked::Decoder::try_new(&values[decoder.consumed_bytes()..])?;
        let suffix = (&mut decoder)
            .map(|r| r.map(|v| v as i32).unwrap())
            .collect::<Vec<_>>();
        data_offset += decoder.consumed_bytes();

        Ok(Self {
            prefix: prefix.into_iter(),
            suffix: suffix.into_iter(),
            data: values,
            data_offset,
            last_value: vec![],
        })
    }
}

impl<'a> Iterator for DeltaBytes<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let prefix_len = self.prefix.next()? as usize;
        let suffix_len = self.suffix.next()? as usize;

        self.last_value.truncate(prefix_len);
        self.last_value
            .extend_from_slice(&self.data[self.data_offset..self.data_offset + suffix_len]);
        self.data_offset += suffix_len;

        // SAFETY: the consumer will only keep one value around per iteration.
        // We need a different API for this to work with safe code.
        let extend_lifetime =
            unsafe { std::mem::transmute::<&[u8], &'a [u8]>(self.last_value.as_slice()) };
        Some(extend_lifetime)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.prefix.size_hint()
    }
}

#[derive(Debug)]
pub(super) struct FilteredRequired<'a> {
    pub values: SliceFilteredIter<std::iter::Take<BinaryIter<'a>>>,
}

impl<'a> FilteredRequired<'a> {
    pub fn new(page: &'a DataPage) -> Self {
        let values = BinaryIter::new(page.buffer()).take(page.num_values());

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct FilteredDelta<'a> {
    pub values: SliceFilteredIter<Delta<'a>>,
}

impl<'a> FilteredDelta<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let values = Delta::try_new(page)?;

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

pub(super) type Dict = BinaryArray<i64>;

#[derive(Debug)]
pub(super) struct RequiredDictionary<'a> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a Dict,
}

impl<'a> RequiredDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct FilteredRequiredDictionary<'a> {
    pub values: SliceFilteredIter<hybrid_rle::HybridRleDecoder<'a>>,
    pub dict: &'a Dict,
}

impl<'a> FilteredRequiredDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Ok(Self { values, dict })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct ValuesDictionary<'a> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a Dict,
}

impl<'a> ValuesDictionary<'a> {
    pub fn try_new(page: &'a DataPage, dict: &'a Dict) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
enum State<'a> {
    Optional(OptionalPageValidity<'a>, BinaryIter<'a>),
    Required(Required<'a>),
    RequiredDictionary(RequiredDictionary<'a>),
    OptionalDictionary(OptionalPageValidity<'a>, ValuesDictionary<'a>),
    Delta(Delta<'a>),
    OptionalDelta(OptionalPageValidity<'a>, Delta<'a>),
    OptionalDeltaByteArray(OptionalPageValidity<'a>, DeltaBytes<'a>),
    FilteredRequired(FilteredRequired<'a>),
    FilteredDelta(FilteredDelta<'a>),
    FilteredOptionalDelta(FilteredOptionalPageValidity<'a>, Delta<'a>),
    FilteredOptional(FilteredOptionalPageValidity<'a>, BinaryIter<'a>),
    FilteredRequiredDictionary(FilteredRequiredDictionary<'a>),
    FilteredOptionalDictionary(FilteredOptionalPageValidity<'a>, ValuesDictionary<'a>),
}

impl<'a> utils::PageState<'a> for State<'a> {
    fn len(&self) -> usize {
        match self {
            State::Optional(validity, _) => validity.len(),
            State::Required(state) => state.len(),
            State::Delta(state) => state.len(),
            State::OptionalDelta(state, _) => state.len(),
            State::RequiredDictionary(values) => values.len(),
            State::OptionalDictionary(optional, _) => optional.len(),
            State::FilteredRequired(state) => state.len(),
            State::FilteredOptional(validity, _) => validity.len(),
            State::FilteredDelta(state) => state.len(),
            State::FilteredOptionalDelta(state, _) => state.len(),
            State::FilteredRequiredDictionary(values) => values.len(),
            State::FilteredOptionalDictionary(optional, _) => optional.len(),
            State::OptionalDeltaByteArray(optional, _) => optional.len(),
        }
    }
}

impl<O: Offset> DecodedState for (Binary<O>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Default)]
struct BinaryDecoder<O: Offset> {
    phantom_o: std::marker::PhantomData<O>,
    check_utf8: Cell<bool>,
}

impl<'a, O: Offset> utils::Decoder<'a> for BinaryDecoder<O> {
    type State = State<'a>;
    type Dict = Dict;
    type DecodedState = (Binary<O>, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        let is_string = matches!(
            page.descriptor.primitive_type.logical_type,
            Some(PrimitiveLogicalType::String)
        );
        self.check_utf8.set(is_string);

        match (page.encoding(), dict, is_optional, is_filtered) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, false) => {
                if is_string {
                    try_check_utf8(dict.offsets(), dict.values())?;
                }
                Ok(State::RequiredDictionary(RequiredDictionary::try_new(
                    page, dict,
                )?))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, false) => {
                if is_string {
                    try_check_utf8(dict.offsets(), dict.values())?;
                }
                Ok(State::OptionalDictionary(
                    OptionalPageValidity::try_new(page)?,
                    ValuesDictionary::try_new(page, dict)?,
                ))
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, true) => {
                if is_string {
                    try_check_utf8(dict.offsets(), dict.values())?;
                }
                FilteredRequiredDictionary::try_new(page, dict)
                    .map(State::FilteredRequiredDictionary)
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, true) => {
                if is_string {
                    try_check_utf8(dict.offsets(), dict.values())?;
                }
                Ok(State::FilteredOptionalDictionary(
                    FilteredOptionalPageValidity::try_new(page)?,
                    ValuesDictionary::try_new(page, dict)?,
                ))
            },
            (Encoding::Plain, _, true, false) => {
                let (_, _, values) = split_buffer(page)?;

                let values = BinaryIter::new(values);

                Ok(State::Optional(
                    OptionalPageValidity::try_new(page)?,
                    values,
                ))
            },
            (Encoding::Plain, _, false, false) => Ok(State::Required(Required::try_new(page)?)),
            (Encoding::Plain, _, false, true) => {
                Ok(State::FilteredRequired(FilteredRequired::new(page)))
            },
            (Encoding::Plain, _, true, true) => {
                let (_, _, values) = split_buffer(page)?;

                Ok(State::FilteredOptional(
                    FilteredOptionalPageValidity::try_new(page)?,
                    BinaryIter::new(values),
                ))
            },
            (Encoding::DeltaLengthByteArray, _, false, false) => {
                Delta::try_new(page).map(State::Delta)
            },
            (Encoding::DeltaLengthByteArray, _, true, false) => Ok(State::OptionalDelta(
                OptionalPageValidity::try_new(page)?,
                Delta::try_new(page)?,
            )),
            (Encoding::DeltaLengthByteArray, _, false, true) => {
                FilteredDelta::try_new(page).map(State::FilteredDelta)
            },
            (Encoding::DeltaLengthByteArray, _, true, true) => Ok(State::FilteredOptionalDelta(
                FilteredOptionalPageValidity::try_new(page)?,
                Delta::try_new(page)?,
            )),
            (Encoding::DeltaByteArray, _, true, false) => Ok(State::OptionalDeltaByteArray(
                OptionalPageValidity::try_new(page)?,
                DeltaBytes::try_new(page)?,
            )),
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Binary::<O>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn extend_from_state(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        additional: usize,
    ) -> PolarsResult<()> {
        let (values, validity) = decoded;
        let mut validate_utf8 = self.check_utf8.take();
        let len_before = values.offsets.len();
        match state {
            State::Optional(page_validity, page_values) => extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                page_values,
            ),
            State::Required(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            State::Delta(page) => {
                values.extend_lengths(page.lengths.by_ref().take(additional), &mut page.values);
            },
            State::OptionalDelta(page_validity, page_values) => {
                let Binary {
                    offsets,
                    values: values_,
                } = values;

                let last_offset = *offsets.last();
                extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    offsets,
                    page_values.lengths.by_ref(),
                );

                let length = *offsets.last() - last_offset;

                let (consumed, remaining) = page_values.values.split_at(length.to_usize());
                page_values.values = remaining;
                values_.extend_from_slice(consumed);
            },
            State::FilteredRequired(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            State::FilteredDelta(page) => {
                for x in page.values.by_ref().take(additional) {
                    values.push(x)
                }
            },
            State::OptionalDictionary(page_validity, page_values) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page_values.dict;
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page_values
                        .values
                        .by_ref()
                        .map(|index| page_dict.value(index.unwrap() as usize)),
                )
            },
            State::RequiredDictionary(page) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page.dict;

                for x in page
                    .values
                    .by_ref()
                    .map(|index| page_dict.value(index.unwrap() as usize))
                    .take(additional)
                {
                    values.push(x)
                }
            },
            State::FilteredOptional(page_validity, page_values) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values.by_ref(),
                );
            },
            State::FilteredOptionalDelta(page_validity, page_values) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values.by_ref(),
                );
            },
            State::FilteredRequiredDictionary(page) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page.dict;
                for x in page
                    .values
                    .by_ref()
                    .map(|index| page_dict.value(index.unwrap() as usize))
                    .take(additional)
                {
                    values.push(x)
                }
            },
            State::FilteredOptionalDictionary(page_validity, page_values) => {
                // Already done on the dict.
                validate_utf8 = false;
                let page_dict = &page_values.dict;
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page_values
                        .values
                        .by_ref()
                        .map(|index| page_dict.value(index.unwrap() as usize)),
                )
            },
            State::OptionalDeltaByteArray(page_validity, page_values) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    page_values,
                )
            },
        }

        if validate_utf8 {
            let offsets = &values.offsets.as_slice()[len_before..];
            try_check_utf8(offsets, &values.values)
        } else {
            Ok(())
        }
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, page.num_values)
    }
}

pub(super) fn finish<O: Offset>(
    data_type: &ArrowDataType,
    mut values: Binary<O>,
    mut validity: MutableBitmap,
) -> PolarsResult<Box<dyn Array>> {
    values.offsets.shrink_to_fit();
    values.values.shrink_to_fit();
    validity.shrink_to_fit();

    match data_type.to_physical_type() {
        PhysicalType::Binary | PhysicalType::LargeBinary => BinaryArray::<O>::try_new(
            data_type.clone(),
            values.offsets.into(),
            values.values.into(),
            validity.into(),
        )
        .map(|x| x.boxed()),
        PhysicalType::Utf8 | PhysicalType::LargeUtf8 => unsafe {
            Ok(Utf8Array::<O>::new_unchecked(
                data_type.clone(),
                values.offsets.into(),
                values.values.into(),
                validity.into(),
            )
            .boxed())
        },
        _ => unreachable!(),
    }
}

pub struct Iter<O: Offset, I: PagesIter> {
    iter: I,
    data_type: ArrowDataType,
    items: VecDeque<(Binary<O>, MutableBitmap)>,
    dict: Option<Dict>,
    chunk_size: Option<usize>,
    remaining: usize,
}

impl<O: Offset, I: PagesIter> Iter<O, I> {
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        chunk_size: Option<usize>,
        num_rows: usize,
    ) -> Self {
        Self {
            iter,
            data_type,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
        }
    }
}

impl<O: Offset, I: PagesIter> Iterator for Iter<O, I> {
    type Item = PolarsResult<Box<dyn Array>>;

    fn next(&mut self) -> Option<Self::Item> {
        let decoder = BinaryDecoder::<O>::default();
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                self.chunk_size,
                &decoder,
            );
            match maybe_state {
                MaybeNext::Some(Ok((values, validity))) => {
                    return Some(finish(&self.data_type, values, validity))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}

pub(super) fn deserialize_plain(values: &[u8], num_values: usize) -> Dict {
    let all = BinaryIter::new(values).take(num_values).collect::<Vec<_>>();
    let values_size = all.iter().map(|v| v.len()).sum::<usize>();
    let mut dict_values = MutableBinaryValuesArray::<i64>::with_capacities(all.len(), values_size);
    for v in all {
        dict_values.push(v)
    }
    dict_values.into()
}
