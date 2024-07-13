use std::cell::RefCell;

use serde::ser::{Error, SerializeMap};
use serde::{Serialize, Serializer};

use crate::chunked_array::metadata::MetadataFlags;
use crate::prelude::*;
use crate::series::implementations::null::NullChunked;

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
    bit_settings: MetadataFlags,
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
    state.serialize_entry("values", &IterSer::new(ca.iter()))?;
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

impl_serialize!(StringChunked);
impl_serialize!(BooleanChunked);
impl_serialize!(ListChunked);
impl_serialize!(BinaryChunked);
#[cfg(feature = "dtype-array")]
impl_serialize!(ArrayChunked);

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
            if self.null_count() > 0 {
                return Err(S::Error::custom(
                    "serializing struct with outer validity not yet supported",
                ));
            }

            let mut state = serializer.serialize_map(Some(3))?;
            state.serialize_entry("name", self.name())?;
            state.serialize_entry("datatype", self.dtype())?;
            state.serialize_entry("values", &self.fields_as_series())?;
            state.end()
        }
    }
}

impl Serialize for NullChunked {
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
            state.serialize_entry("values", &IterSer::new(std::iter::once(self.len())))?;
            state.end()
        }
    }
}
