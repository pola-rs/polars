use super::DeDataType;
use crate::prelude::*;
use serde::ser::SerializeMap;
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

fn serialize_impl<T, S>(
    serializer: S,
    name: &str,
    dtype: &DataType,
    ca: &ChunkedArray<T>,
) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
where
    T: PolarsNumericType,
    T::Native: Serialize,
    S: Serializer,
{
    let mut state = serializer.serialize_map(Some(3))?;
    state.serialize_entry("name", name)?;
    let dtype: DeDataType = dtype.into();
    state.serialize_entry("datatype", &dtype)?;
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
        serialize_impl(serializer, self.name(), self.dtype(), self)
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
        serialize_impl(serializer, self.name(), self.dtype(), self)
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
                let mut state = serializer.serialize_map(Some(3))?;
                state.serialize_entry("name", self.name())?;
                let dtype: DeDataType = self.dtype().into();
                state.serialize_entry("datatype", &dtype)?;
                state.serialize_entry("values", &IterSer::new(self.into_iter()))?;
                state.end()
            }
        }
    };
}

impl_serialize!(Utf8Chunked);
impl_serialize!(BooleanChunked);
impl_serialize!(ListChunked);

#[cfg(feature = "dtype-categorical")]
impl Serialize for CategoricalChunked {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let ca = self.cast(&DataType::Utf8).unwrap();
        ca.serialize(serializer)
    }
}
