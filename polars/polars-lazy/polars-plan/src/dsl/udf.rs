use std::fmt::{self, Debug};
use std::ops::Deref;
use std::sync::Arc;

#[cfg(feature = "serde")]
pub use erased_serde::{
    Deserializer as ErasedDeserializer, Error as ErasedError, Serialize as ErasedSerialize,
};
use polars_core::export::once_cell::sync::OnceCell;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
pub use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::prelude::*;

#[cfg(feature = "serde")]
pub trait UdfDeserializer: Send + Sync {
    fn deserialize_udf(
        &self,
        deserializer: &mut dyn ErasedDeserializer,
    ) -> Result<Arc<dyn SerializableUdf>, ErasedError>;
}

#[cfg(feature = "serde")]
static UDF_DESERIALIZER: OnceCell<Box<dyn UdfDeserializer>> = OnceCell::new();
#[cfg(feature = "serde")]
pub fn set_udf_deserializer(
    udf_serializer: Box<dyn UdfDeserializer>,
) -> Result<(), Box<dyn UdfDeserializer>> {
    UDF_DESERIALIZER.set(udf_serializer)
}

pub trait SerializableUdf: Send + Sync {
    /// Used for Expr::AnonymousFunction
    fn call_series_slice(&self, _s: &mut [Series]) -> PolarsResult<Series> {
        unimplemented!()
    }
    /// Used for Expr::AnonymousFunction
    fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        _fields: &[Field],
    ) -> PolarsResult<Field> {
        unimplemented!()
    }
    /// Used for Expr::RenameAlias
    fn map_name(&self, _name: &str) -> PolarsResult<String> {
        unimplemented!()
    }

    // upcasting methods

    #[cfg(feature = "serde")]
    fn as_serialize(&self) -> Option<&dyn ErasedSerialize> {
        None
    }

    fn as_debug(&self) -> &dyn Debug {
        &"<user-defined function>"
    }
}

/// Note that the resulting UDF will not be serializable.
pub(crate) fn make_series_udf<E, O>(eval: E, output: O) -> UdfWrapper
where
    E: Fn(&mut [Series]) -> PolarsResult<Series> + Send + Sync + 'static,
    O: Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'static,
{
    struct SeriesUdf<E, O>(E, O);
    impl<E, O> SerializableUdf for SeriesUdf<E, O>
    where
        E: Fn(&mut [Series]) -> PolarsResult<Series> + Send + Sync + 'static,
        O: Fn(&[Field]) -> PolarsResult<Field> + Send + Sync + 'static,
    {
        fn call_series_slice(&self, s: &mut [Series]) -> PolarsResult<Series> {
            self.0(s)
        }
        fn get_field(
            &self,
            _input_schema: &Schema,
            _cntxt: Context,
            f: &[Field],
        ) -> PolarsResult<Field> {
            self.1(f)
        }
    }
    UdfWrapper(Arc::new(SeriesUdf(eval, output)))
}

/// Note that the resulting UDF will not be serializable.
pub(crate) fn make_rename_alias_udf<E>(func: E) -> UdfWrapper
where
    E: Fn(&str) -> PolarsResult<String> + Send + Sync + 'static,
{
    struct RenameAliasUdf<E>(E);
    impl<E> SerializableUdf for RenameAliasUdf<E>
    where
        E: Fn(&str) -> PolarsResult<String> + Send + Sync + 'static,
    {
        fn map_name(&self, n: &str) -> PolarsResult<String> {
            self.0(n)
        }
    }
    UdfWrapper(Arc::new(RenameAliasUdf(func)))
}

#[derive(Clone)]
pub struct UdfWrapper(pub Arc<dyn SerializableUdf>);
impl Deref for UdfWrapper {
    type Target = dyn SerializableUdf;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl PartialEq for UdfWrapper {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Debug for UdfWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.as_debug().fmt(f)
    }
}

#[cfg(feature = "serde")]
impl Serialize for UdfWrapper {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0
            .as_serialize()
            .ok_or_else(|| {
                serde::ser::Error::custom("Cannot serialize this user-defined function")
            })?
            .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for UdfWrapper {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let udf_serde = UDF_DESERIALIZER.get().ok_or_else(|| {
            serde::de::Error::custom(
                "Cannot deserialize a user-defined function without a deserializer",
            )
        })?;

        let val = udf_serde
            .deserialize_udf(&mut <dyn ErasedDeserializer>::erase(deserializer))
            .map_err(serde::de::Error::custom)?;
        Ok(UdfWrapper(val))
    }
}
