use std::cell::RefCell;

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use crate::chunked_array::Settings;
use crate::prelude::*;

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
        serializer.collect_seq(iter)
    }
}

fn serialize_impl<T, S>(
    serializer: S,
    name: &str,
    dtype: &DataType,
    bit_settings: Settings,
    ca: &ChunkedArray<T>,
) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
where
    T: PolarsNumericType,
    T::Native: Serialize,
    S: Serializer,
{
    let mut state = serializer.serialize_map(Some(4))?;
    state.serialize_entry("name", name)?;
    state.serialize_entry("datatype", dtype)?;
    state.serialize_entry("bit_settings", &bit_settings)?;
    state.serialize_entry("values", &IterSer::new(ca.into_iter()))?;
    state.end()
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
        serialize_impl(
            serializer,
            self.name(),
            self.dtype(),
            self.get_flags(),
            self,
        )
    }
}

impl<K: PolarsDataType, T: PolarsNumericType> Serialize for Logical<K, T>
where
    Self: LogicalType,
    ChunkedArray<T>: Serialize,
    T::Native: Serialize,
{
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        serialize_impl(
            serializer,
            self.name(),
            self.dtype(),
            self.get_flags(),
            self,
        )
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
                let mut state = serializer.serialize_map(Some(4))?;
                state.serialize_entry("name", self.name())?;
                state.serialize_entry("datatype", self.dtype())?;
                state.serialize_entry("bit_settings", &self.get_flags())?;
                state.serialize_entry("values", &IterSer::new(self.into_iter()))?;
                state.end()
            }
        }
    };
}

impl_serialize!(Utf8Chunked);
impl_serialize!(BooleanChunked);
impl_serialize!(ListChunked);
impl_serialize!(BinaryChunked);

#[cfg(feature = "dtype-categorical")]
impl Serialize for CategoricalChunked {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        {
            let mut state = serializer.serialize_map(Some(4))?;
            state.serialize_entry("name", self.name())?;
            state.serialize_entry("datatype", self.dtype())?;
            state.serialize_entry("bit_settings", &self.get_flags())?;
            state.serialize_entry("values", &IterSer::new(self.iter_str()))?;
            state.end()
        }
    }
}

#[cfg(feature = "dtype-struct")]
impl Serialize for StructChunked {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        {
            let mut state = serializer.serialize_map(Some(3))?;
            state.serialize_entry("name", self.name())?;
            state.serialize_entry("datatype", self.dtype())?;
            state.serialize_entry("values", self.fields())?;
            state.end()
        }
    }
}
