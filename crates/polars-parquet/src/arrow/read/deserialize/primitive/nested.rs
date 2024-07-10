use std::collections::VecDeque;

use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::nested_utils::*;
use super::super::utils::MaybeNext;
use super::super::{utils, PagesIter};
use super::basic::{deserialize_plain, Values, ValuesDictionary};
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;
use crate::parquet::types::{decode, NativeType as ParquetNativeType};

#[derive(Debug)]
struct State<'a, T: NativeType> {
    is_optional: bool,
    translation: StateTranslation<'a, T>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum StateTranslation<'a, T: NativeType> {
    Unit(Values<'a>),
    Dictionary(ValuesDictionary<'a, T>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
}

impl<'a, T: NativeType> utils::PageState<'a> for State<'a, T> {
    fn len(&self) -> usize {
        match &self.translation {
            StateTranslation::Unit(values) => values.len(),
            StateTranslation::Dictionary(values) => values.len(),
            StateTranslation::ByteStreamSplit(decoder) => decoder.len(),
        }
    }
}

#[derive(Debug)]
struct PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    phantom: std::marker::PhantomData<T>,
    phantom_p: std::marker::PhantomData<P>,
    op: F,
}

impl<T, P, F> PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    #[inline]
    fn new(op: F) -> Self {
        Self {
            phantom: std::marker::PhantomData,
            phantom_p: std::marker::PhantomData,
            op,
        }
    }
}

impl<'a, T, P, F> NestedDecoder<'a> for PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type State = State<'a, T>;
    type Dictionary = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dictionary>,
    ) -> PolarsResult<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        if is_filtered {
            return Err(utils::not_implemented(page));
        }

        let translation = match (page.encoding(), dict) {
            (Encoding::Plain, _) => StateTranslation::Unit(Values::try_new::<P>(page)?),
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                StateTranslation::Dictionary(ValuesDictionary::try_new(page, dict)?)
            },
            (Encoding::ByteStreamSplit, _) => {
                StateTranslation::ByteStreamSplit(byte_stream_split::Decoder::try_new(
                    split_buffer(page)?.values,
                    std::mem::size_of::<P>(),
                )?)
            },
            _ => return Err(utils::not_implemented(page)),
        };

        Ok(State {
            is_optional,
            translation,
        })
    }

    /// Initializes a new state
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn push_n_valid(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        if utils::PageState::len(state) < n {
            return Err(ParquetError::oos("No values left in page"));
        }

        match &mut state.translation {
            StateTranslation::Unit(page_values) => {
                for value in page_values.values.by_ref().take(n) {
                    values.push((self.op)(decode(value)));
                }
            },
            StateTranslation::Dictionary(page) => {
                let translator = DictionaryTranslator(page.dict);
                page.values
                    .translate_and_collect_n_into(values, n, &translator)?;
            },
            StateTranslation::ByteStreamSplit(decoder) => {
                for value in decoder.iter_converted(decode).by_ref().take(n) {
                    values.push((self.op)(value));
                }
            },
        }

        if state.is_optional {
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize) {
        let (values, validity) = decoded;
        values.resize(values.len() + n, T::default());
        validity.extend_constant(n, false);
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        deserialize_plain(&page.buffer, self.op)
    }
}

fn finish<T: NativeType>(
    data_type: &ArrowDataType,
    values: Vec<T>,
    validity: MutableBitmap,
) -> PrimitiveArray<T> {
    PrimitiveArray::new(data_type.clone(), values.into(), validity.into())
}

/// An iterator adapter over [`PagesIter`] assumed to be encoded as boolean arrays
#[derive(Debug)]
pub struct NestedIter<T, I, P, F>
where
    I: PagesIter,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    iter: I,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    items: VecDeque<(NestedState, (Vec<T>, MutableBitmap))>,
    dict: Option<Vec<T>>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: PrimitiveDecoder<T, P, F>,
}

impl<T, I, P, F> NestedIter<T, I, P, F>
where
    I: PagesIter,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        op: F,
    ) -> Self {
        Self {
            iter,
            init,
            data_type,
            items: VecDeque::new(),
            dict: None,
            chunk_size,
            remaining: num_rows,
            decoder: PrimitiveDecoder::new(op),
        }
    }
}

impl<T, I, P, F> Iterator for NestedIter<T, I, P, F>
where
    I: PagesIter,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type Item = PolarsResult<(NestedState, PrimitiveArray<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                &self.init,
                self.chunk_size,
                &self.decoder,
            );
            match maybe_state {
                MaybeNext::Some(Ok((nested, state))) => {
                    return Some(Ok((nested, finish(&self.data_type, state.0, state.1))))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
