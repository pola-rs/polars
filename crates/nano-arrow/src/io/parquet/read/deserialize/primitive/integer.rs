use std::collections::VecDeque;

use num_traits::AsPrimitive;
use parquet2::{
    deserialize::SliceFilteredIter,
    encoding::{delta_bitpacked::Decoder, Encoding},
    page::{split_buffer, DataPage, DictPage},
    schema::Repetition,
    types::NativeType as ParquetNativeType,
};

use crate::{
    array::MutablePrimitiveArray,
    bitmap::MutableBitmap,
    datatypes::DataType,
    error::{Error, Result},
    io::parquet::read::deserialize::utils::{
        get_selected_rows, FilteredOptionalPageValidity, OptionalPageValidity,
    },
    types::NativeType,
};

use super::super::utils;
use super::super::Pages;

use super::basic::{finish, PrimitiveDecoder, State as PrimitiveState};

/// The state of a [`DataPage`] of an integer parquet type (i32 or i64)
#[derive(Debug)]
enum State<'a, T>
where
    T: NativeType,
{
    Common(PrimitiveState<'a, T>),
    DeltaBinaryPackedRequired(Decoder<'a>),
    DeltaBinaryPackedOptional(OptionalPageValidity<'a>, Decoder<'a>),
    FilteredDeltaBinaryPackedRequired(SliceFilteredIter<Decoder<'a>>),
    FilteredDeltaBinaryPackedOptional(FilteredOptionalPageValidity<'a>, Decoder<'a>),
}

impl<'a, T> utils::PageState<'a> for State<'a, T>
where
    T: NativeType,
{
    fn len(&self) -> usize {
        match self {
            State::Common(state) => state.len(),
            State::DeltaBinaryPackedRequired(state) => state.size_hint().0,
            State::DeltaBinaryPackedOptional(state, _) => state.len(),
            State::FilteredDeltaBinaryPackedRequired(state) => state.size_hint().0,
            State::FilteredDeltaBinaryPackedOptional(state, _) => state.len(),
        }
    }
}

/// Decoder of integer parquet type
#[derive(Debug)]
struct IntDecoder<T, P, F>(PrimitiveDecoder<T, P, F>)
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    F: Fn(P) -> T;

impl<T, P, F> IntDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    F: Fn(P) -> T,
{
    #[inline]
    fn new(op: F) -> Self {
        Self(PrimitiveDecoder::new(op))
    }
}

impl<'a, T, P, F> utils::Decoder<'a> for IntDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
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
            (Encoding::DeltaBinaryPacked, _, false, false) => {
                let (_, _, values) = split_buffer(page)?;
                Decoder::try_new(values)
                    .map(State::DeltaBinaryPackedRequired)
                    .map_err(Error::from)
            }
            (Encoding::DeltaBinaryPacked, _, true, false) => {
                let (_, _, values) = split_buffer(page)?;
                Ok(State::DeltaBinaryPackedOptional(
                    OptionalPageValidity::try_new(page)?,
                    Decoder::try_new(values)?,
                ))
            }
            (Encoding::DeltaBinaryPacked, _, false, true) => {
                let (_, _, values) = split_buffer(page)?;
                let values = Decoder::try_new(values)?;

                let rows = get_selected_rows(page);
                let values = SliceFilteredIter::new(values, rows);

                Ok(State::FilteredDeltaBinaryPackedRequired(values))
            }
            (Encoding::DeltaBinaryPacked, _, true, true) => {
                let (_, _, values) = split_buffer(page)?;
                let values = Decoder::try_new(values)?;

                Ok(State::FilteredDeltaBinaryPackedOptional(
                    FilteredOptionalPageValidity::try_new(page)?,
                    values,
                ))
            }
            _ => self.0.build_state(page, dict).map(State::Common),
        }
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        self.0.with_capacity(capacity)
    }

    fn extend_from_state(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        remaining: usize,
    ) {
        let (values, validity) = decoded;
        match state {
            State::Common(state) => self.0.extend_from_state(state, decoded, remaining),
            State::DeltaBinaryPackedRequired(state) => {
                values.extend(
                    state
                        .by_ref()
                        .map(|x| x.unwrap().as_())
                        .map(self.0.op)
                        .take(remaining),
                );
            }
            State::DeltaBinaryPackedOptional(page_validity, page_values) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(remaining),
                    values,
                    page_values
                        .by_ref()
                        .map(|x| x.unwrap().as_())
                        .map(self.0.op),
                )
            }
            State::FilteredDeltaBinaryPackedRequired(page) => {
                values.extend(
                    page.by_ref()
                        .map(|x| x.unwrap().as_())
                        .map(self.0.op)
                        .take(remaining),
                );
            }
            State::FilteredDeltaBinaryPackedOptional(page_validity, page_values) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(remaining),
                    values,
                    page_values
                        .by_ref()
                        .map(|x| x.unwrap().as_())
                        .map(self.0.op),
                );
            }
        }
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        self.0.deserialize_dict(page)
    }
}

/// An [`Iterator`] adapter over [`Pages`] assumed to be encoded as primitive arrays
/// encoded as parquet integer types
#[derive(Debug)]
pub struct IntegerIter<T, I, P, F>
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

impl<T, I, P, F> IntegerIter<T, I, P, F>
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

impl<T, I, P, F> Iterator for IntegerIter<T, I, P, F>
where
    I: Pages,
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
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
            &IntDecoder::new(self.op),
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
