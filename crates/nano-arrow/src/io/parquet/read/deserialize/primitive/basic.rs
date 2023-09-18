use std::collections::VecDeque;

use parquet2::{
    deserialize::SliceFilteredIter,
    encoding::{hybrid_rle, Encoding},
    page::{split_buffer, DataPage, DictPage},
    schema::Repetition,
    types::decode,
    types::NativeType as ParquetNativeType,
};

use crate::{
    array::MutablePrimitiveArray, bitmap::MutableBitmap, datatypes::DataType, error::Result,
    types::NativeType,
};

use super::super::utils;
use super::super::utils::{get_selected_rows, FilteredOptionalPageValidity, OptionalPageValidity};
use super::super::Pages;

#[derive(Debug)]
pub(super) struct FilteredRequiredValues<'a> {
    values: SliceFilteredIter<std::slice::ChunksExact<'a, u8>>,
}

impl<'a> FilteredRequiredValues<'a> {
    pub fn try_new<P: ParquetNativeType>(page: &'a DataPage) -> Result<Self> {
        let (_, _, values) = split_buffer(page)?;
        assert_eq!(values.len() % std::mem::size_of::<P>(), 0);

        let values = values.chunks_exact(std::mem::size_of::<P>());

        let rows = get_selected_rows(page);
        let values = SliceFilteredIter::new(values, rows);

        Ok(Self { values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct Values<'a> {
    pub values: std::slice::ChunksExact<'a, u8>,
}

impl<'a> Values<'a> {
    pub fn try_new<P: ParquetNativeType>(page: &'a DataPage) -> Result<Self> {
        let (_, _, values) = split_buffer(page)?;
        assert_eq!(values.len() % std::mem::size_of::<P>(), 0);
        Ok(Self {
            values: values.chunks_exact(std::mem::size_of::<P>()),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct ValuesDictionary<'a, T>
where
    T: NativeType,
{
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a Vec<T>,
}

impl<'a, T> ValuesDictionary<'a, T>
where
    T: NativeType,
{
    pub fn try_new(page: &'a DataPage, dict: &'a Vec<T>) -> Result<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

// The state of a `DataPage` of `Primitive` parquet primitive type
#[derive(Debug)]
pub(super) enum State<'a, T>
where
    T: NativeType,
{
    Optional(OptionalPageValidity<'a>, Values<'a>),
    Required(Values<'a>),
    RequiredDictionary(ValuesDictionary<'a, T>),
    OptionalDictionary(OptionalPageValidity<'a>, ValuesDictionary<'a, T>),
    FilteredRequired(FilteredRequiredValues<'a>),
    FilteredOptional(FilteredOptionalPageValidity<'a>, Values<'a>),
}

impl<'a, T> utils::PageState<'a> for State<'a, T>
where
    T: NativeType,
{
    fn len(&self) -> usize {
        match self {
            State::Optional(optional, _) => optional.len(),
            State::Required(values) => values.len(),
            State::RequiredDictionary(values) => values.len(),
            State::OptionalDictionary(optional, _) => optional.len(),
            State::FilteredRequired(values) => values.len(),
            State::FilteredOptional(optional, _) => optional.len(),
        }
    }
}

#[derive(Debug)]
pub(super) struct PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    phantom: std::marker::PhantomData<T>,
    phantom_p: std::marker::PhantomData<P>,
    pub op: F,
}

impl<T, P, F> PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    #[inline]
    pub(super) fn new(op: F) -> Self {
        Self {
            phantom: std::marker::PhantomData,
            phantom_p: std::marker::PhantomData,
            op,
        }
    }
}

impl<T: std::fmt::Debug> utils::DecodedState for (Vec<T>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, T, P, F> utils::Decoder<'a> for PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type State = State<'a, T>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn build_state(&self, page: &'a DataPage, dict: Option<&'a Self::Dict>) -> Result<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        match (page.encoding(), dict, is_optional, is_filtered) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, false) => {
                ValuesDictionary::try_new(page, dict).map(State::RequiredDictionary)
            }
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, false) => {
                Ok(State::OptionalDictionary(
                    OptionalPageValidity::try_new(page)?,
                    ValuesDictionary::try_new(page, dict)?,
                ))
            }
            (Encoding::Plain, _, true, false) => {
                let validity = OptionalPageValidity::try_new(page)?;
                let values = Values::try_new::<P>(page)?;

                Ok(State::Optional(validity, values))
            }
            (Encoding::Plain, _, false, false) => Ok(State::Required(Values::try_new::<P>(page)?)),
            (Encoding::Plain, _, false, true) => {
                FilteredRequiredValues::try_new::<P>(page).map(State::FilteredRequired)
            }
            (Encoding::Plain, _, true, true) => Ok(State::FilteredOptional(
                FilteredOptionalPageValidity::try_new(page)?,
                Values::try_new::<P>(page)?,
            )),
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn extend_from_state(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        remaining: usize,
    ) {
        let (values, validity) = decoded;
        match state {
            State::Optional(page_validity, page_values) => utils::extend_from_decoder(
                validity,
                page_validity,
                Some(remaining),
                values,
                page_values.values.by_ref().map(decode).map(self.op),
            ),
            State::Required(page) => {
                values.extend(
                    page.values
                        .by_ref()
                        .map(decode)
                        .map(self.op)
                        .take(remaining),
                );
            }
            State::OptionalDictionary(page_validity, page_values) => {
                let op1 = |index: u32| page_values.dict[index as usize];
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(remaining),
                    values,
                    &mut page_values.values.by_ref().map(|x| x.unwrap()).map(op1),
                )
            }
            State::RequiredDictionary(page) => {
                let op1 = |index: u32| page.dict[index as usize];
                values.extend(
                    page.values
                        .by_ref()
                        .map(|x| x.unwrap())
                        .map(op1)
                        .take(remaining),
                );
            }
            State::FilteredRequired(page) => {
                values.extend(
                    page.values
                        .by_ref()
                        .map(decode)
                        .map(self.op)
                        .take(remaining),
                );
            }
            State::FilteredOptional(page_validity, page_values) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(remaining),
                    values,
                    page_values.values.by_ref().map(decode).map(self.op),
                );
            }
        }
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, self.op)
    }
}

pub(super) fn finish<T: NativeType>(
    data_type: &DataType,
    values: Vec<T>,
    validity: MutableBitmap,
) -> MutablePrimitiveArray<T> {
    let validity = if validity.is_empty() {
        None
    } else {
        Some(validity)
    };
    MutablePrimitiveArray::try_new(data_type.clone(), values, validity).unwrap()
}

/// An [`Iterator`] adapter over [`Pages`] assumed to be encoded as primitive arrays
#[derive(Debug)]
pub struct Iter<T, I, P, F>
where
    I: Pages,
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    iter: I,
    data_type: DataType,
    items: VecDeque<(Vec<T>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
    dict: Option<Vec<T>>,
    op: F,
    phantom: std::marker::PhantomData<P>,
}

impl<T, I, P, F> Iter<T, I, P, F>
where
    I: Pages,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    pub fn new(
        iter: I,
        data_type: DataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        op: F,
    ) -> Self {
        Self {
            iter,
            data_type,
            items: VecDeque::new(),
            dict: None,
            remaining: num_rows,
            chunk_size,
            op,
            phantom: Default::default(),
        }
    }
}

impl<T, I, P, F> Iterator for Iter<T, I, P, F>
where
    I: Pages,
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type Item = Result<MutablePrimitiveArray<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let maybe_state = utils::next(
            &mut self.iter,
            &mut self.items,
            &mut self.dict,
            &mut self.remaining,
            self.chunk_size,
            &PrimitiveDecoder::new(self.op),
        );
        match maybe_state {
            utils::MaybeNext::Some(Ok((values, validity))) => {
                Some(Ok(finish(&self.data_type, values, validity)))
            }
            utils::MaybeNext::Some(Err(e)) => Some(Err(e)),
            utils::MaybeNext::None => None,
            utils::MaybeNext::More => self.next(),
        }
    }
}

pub(super) fn deserialize_plain<T, P, F>(values: &[u8], op: F) -> Vec<T>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    values
        .chunks_exact(std::mem::size_of::<P>())
        .map(decode)
        .map(op)
        .collect::<Vec<_>>()
}
