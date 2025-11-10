use polars_error::{PolarsError, polars_bail};
use polars_utils::pl_str::PlSmallStr;
use serde::{Deserialize, Serialize};

use super::{LibrarySymbol, PluginV1, PluginV1Flags};

#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Serialize, Deserialize)]
pub struct PluginV2Serde {
    lib: PlSmallStr,
    symbol: PlSmallStr,
    data: Vec<u8>,
    function_name: PlSmallStr,
    flags: PluginV1Flags,
}

impl TryFrom<&PluginV1> for PluginV2Serde {
    type Error = PolarsError;

    fn try_from(value: &PluginV1) -> Result<Self, Self::Error> {
        let Some(LibrarySymbol {
            lib,
            symbol,
            library: _,
        }) = value.library.as_deref()
        else {
            polars_bail!(
                InvalidOperation:
                "serialization not supported for this 'opaque' function",
            );
        };

        let mut data = Vec::new();
        unsafe {
            value
                .vtable
                .serialize_data(value.data.ptr_clone(), &mut data)?
        };

        Ok(Self {
            lib: lib.clone(),
            symbol: symbol.clone(),
            data,
            function_name: value.function_name.clone(),
            flags: value.flags,
        })
    }
}

impl TryFrom<PluginV2Serde> for PluginV1 {
    type Error = PolarsError;
    fn try_from(value: PluginV2Serde) -> Result<Self, Self::Error> {
        let PluginV2Serde {
            lib,
            symbol,
            data,
            function_name,
            flags,
        } = value;

        let (library, vtable) = super::load_vtable(&lib, &symbol)?;
        let data_ptr = vtable.deserialize_data(&data)?;

        Ok(PluginV1 {
            flags,
            function_name,
            data: data_ptr,
            library: Some(Box::new(LibrarySymbol {
                lib,
                symbol,
                library,
            })),
            vtable,
        })
    }
}

impl Serialize for PluginV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        let slf = PluginV2Serde::try_from(self).map_err(S::Error::custom)?;
        slf.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PluginV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let slf = PluginV2Serde::deserialize(deserializer)?;
        slf.try_into().map_err(D::Error::custom)
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PluginV1 {
    fn schema_name() -> String {
        "StatefulUdf".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "StatefulUdf"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        PluginV2Serde::json_schema(generator)
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PluginV1Flags {
    fn schema_name() -> String {
        "UdfV2Flags".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "UdfV2Flags"))
    }

    fn json_schema(_generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use serde_json::{Map, Value};

        let name_to_bits: Map<String, Value> = Self::all()
            .iter_names()
            .map(|(name, flag)| (name.to_owned(), flag.bits().into()))
            .collect();

        schemars::schema::Schema::Object(schemars::schema::SchemaObject {
            instance_type: Some(schemars::schema::InstanceType::String.into()),
            format: Some("bitflags".to_owned()),
            extensions: schemars::Map::from_iter([
                // Add a map of flag names and bit patterns to detect schema changes
                ("bitflags".to_owned(), Value::Object(name_to_bits)),
            ]),
            ..Default::default()
        })
    }
}
