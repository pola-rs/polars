use super::DeDataType;
use crate::prelude::*;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use std::cell::RefCell;

pub struct IterSer<I>
where
    I: IntoIterator,
    <I as IntoIterator>::Item: Serialize,
{
    iter: RefCell<Option<I>>,
}

impl<I> IterSer<I>
where
    I: IntoIterator,
    <I as IntoIterator>::Item: Serialize,
{
    fn new(iter: I) -> Self {
        IterSer {
            iter: RefCell::new(Some(iter)),
        }
    }
}

impl<I> Serialize for IterSer<I>
where
    I: IntoIterator,
    <I as IntoIterator>::Item: Serialize,
{
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let iter: I = self.iter.borrow_mut().take().unwrap();
        serializer.collect_seq(iter.into_iter())
    }
}

impl<T> Serialize for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Serialize,
{
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("series", 3)?;
        state.serialize_field("name", self.name())?;
        let dtype: DeDataType = self.dtype().into();
        state.serialize_field("datatype", &dtype)?;
        state.serialize_field("values", &IterSer::new(self.into_iter()))?;
        state.end()
    }
}

macro_rules! impl_serialize {
    ($ca: ident) => {
        impl Serialize for $ca {
            fn serialize<S>(
                &self,
                serializer: S,
            ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
            where
                S: Serializer,
            {
                let mut state = serializer.serialize_struct("series", 3)?;
                state.serialize_field("name", self.name())?;
                let dtype: DeDataType = self.dtype().into();
                state.serialize_field("datatype", &dtype)?;
                state.serialize_field("values", &IterSer::new(self.into_iter()))?;
                state.end()
            }
        }
    };
}

impl_serialize!(Utf8Chunked);
impl_serialize!(BooleanChunked);
impl_serialize!(ListChunked);

impl Serialize for CategoricalChunked {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let ca = self.cast::<Utf8Type>().unwrap();
        ca.serialize(serializer)
    }
}
