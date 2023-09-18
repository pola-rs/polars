use std::collections::VecDeque;

use parquet2::{
    encoding::Encoding,
    page::{DataPage, DictPage},
    schema::Repetition,
    types::decode,
    types::NativeType as ParquetNativeType,
};

use crate::{
    array::PrimitiveArray, bitmap::MutableBitmap, datatypes::DataType, error::Result,
    types::NativeType,
};

use super::super::utils;
use super::super::Pages;
use super::basic::{Values, ValuesDictionary};
use super::{super::nested_utils::*, basic::deserialize_plain};

// The state of a `DataPage` of `Primitive` parquet primitive type
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum State<'a, T>
where
    T: NativeType,
{
    Optional(Values<'a>),
    Required(Values<'a>),
    RequiredDictionary(ValuesDictionary<'a, T>),
    OptionalDictionary(ValuesDictionary<'a, T>),
}

impl<'a, T> utils::PageState<'a> for State<'a, T>
where
    T: NativeType,
{
    fn len(&self) -> usize {
        match self {
            State::Optional(values) => values.len(),
            State::Required(values) => values.len(),
            State::RequiredDictionary(values) => values.len(),
            State::OptionalDictionary(values) => values.len(),
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
    ) -> Result<Self::State> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        match (page.encoding(), dict, is_optional, is_filtered) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false, false) => {
                ValuesDictionary::try_new(page, dict).map(State::RequiredDictionary)
            }
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true, false) => {
                ValuesDictionary::try_new(page, dict).map(State::OptionalDictionary)
            }
            (Encoding::Plain, _, true, false) => Values::try_new::<P>(page).map(State::Optional),
            (Encoding::Plain, _, false, false) => Values::try_new::<P>(page).map(State::Required),
            _ => Err(utils::not_implemented(page)),
        }
    }

    /// Initializes a new state
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn push_valid(&self, state: &mut Self::State, decoded: &mut Self::DecodedState) -> Result<()> {
        let (values, validity) = decoded;
        match state {
            State::Optional(page_values) => {
                let value = page_values.values.by_ref().next().map(decode).map(self.op);
                // convert unwrap to error
                values.push(value.unwrap_or_default());
                validity.push(true);
            }
            State::Required(page_values) => {
                let value = page_values.values.by_ref().next().map(decode).map(self.op);
                // convert unwrap to error
                values.push(value.unwrap_or_default());
            }
            State::RequiredDictionary(page) => {
                let value = page
                    .values
                    .next()
                    .map(|index| page.dict[index.unwrap() as usize]);

                values.push(value.unwrap_or_default());
            }
            State::OptionalDictionary(page) => {
                let value = page
                    .values
                    .next()
                    .map(|index| page.dict[index.unwrap() as usize]);

                values.push(value.unwrap_or_default());
                validity.push(true);
            }
        }
        Ok(())
    }

    fn push_null(&self, decoded: &mut Self::DecodedState) {
        let (values, validity) = decoded;
        values.push(T::default());
        validity.push(false)
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dictionary {
        deserialize_plain(&page.buffer, self.op)
    }
}

fn finish<T: NativeType>(
    data_type: &DataType,
    values: Vec<T>,
    validity: MutableBitmap,
) -> PrimitiveArray<T> {
    PrimitiveArray::new(data_type.clone(), values.into(), validity.into())
}

/// An iterator adapter over [`Pages`] assumed to be encoded as boolean arrays
#[derive(Debug)]
pub struct NestedIter<T, I, P, F>
where
    I: Pages,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    iter: I,
    init: Vec<InitNested>,
    data_type: DataType,
    items: VecDeque<(NestedState, (Vec<T>, MutableBitmap))>,
    dict: Option<Vec<T>>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: PrimitiveDecoder<T, P, F>,
}

impl<T, I, P, F> NestedIter<T, I, P, F>
where
    I: Pages,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: DataType,
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
    I: Pages,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type Item = Result<(NestedState, PrimitiveArray<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
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
            utils::MaybeNext::Some(Ok((nested, state))) => {
                Some(Ok((nested, finish(&self.data_type, state.0, state.1))))
            }
            utils::MaybeNext::Some(Err(e)) => Some(Err(e)),
            utils::MaybeNext::None => None,
            utils::MaybeNext::More => self.next(),
        }
    }
}
