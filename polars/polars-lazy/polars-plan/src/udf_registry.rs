use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub use erased_serde::{
    Deserializer as ErasedDeserializer, Error as ErasedError, Serialize as ErasedSerialize,
};
use serde::de::{DeserializeSeed, Visitor};
use serde::ser::SerializeMap;
pub use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize_udf<S: Serializer>(
    ty: &str,
    obj: &dyn erased_serde::Serialize,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // { <type>: <value> }
    let mut map = serializer.serialize_map(Some(1))?;
    map.serialize_key(ty)?;
    map.serialize_value(obj)?;
    map.end()
}

mod deser {
    use super::*;

    struct DeserSeedWrapper<T> {
        f: DeserializeFn<T>,
    }

    impl<'de, T: 'static> DeserializeSeed<'de> for DeserSeedWrapper<T> {
        type Value = T;

        fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            let mut erased = <dyn erased_serde::Deserializer>::erase(deserializer);
            (self.f)(&mut erased).map_err(serde::de::Error::custom)
        }
    }
    pub(super) struct MapLookupVisitor<'a, T> {
        pub(super) registry: &'a Registry<T>,
    }

    impl<'de, 'a, T: 'static> Visitor<'de> for MapLookupVisitor<'a, T> {
        type Value = T;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "{{user defined function}}")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let k = map
                .next_key::<&str>()?
                .ok_or_else(|| serde::de::Error::missing_field("type"))?;

            let func = self
                .registry
                .map
                .get(k)
                .ok_or_else(|| serde::de::Error::unknown_variant(k, &[]))?;

            let val = map.next_value_seed(DeserSeedWrapper { f: *func })?;

            Ok(val)
        }
    }
}

pub fn deserialize_udf<'de, D: serde::de::Deserializer<'de>, T: 'static>(
    deser: D,
    registry: &Registry<T>,
) -> Result<T, D::Error> {
    deser.deserialize_map(deser::MapLookupVisitor { registry })
}

pub trait RegistryDeserializable<'de> {
    fn deserialize_with_registry<D: Deserializer<'de>>(
        deser: D,
        registry: &UdfSerializeRegistry,
    ) -> Result<Self, D::Error>
    where
        Self: Sized;
}

pub type DeserializeFn<T> =
    fn(&mut dyn erased_serde::Deserializer) -> Result<T, erased_serde::Error>;

pub struct Registry<T> {
    pub map: HashMap<String, DeserializeFn<T>>,
}

impl<T> Registry<T> {
    pub fn new(map: HashMap<String, DeserializeFn<T>>) -> Self {
        Self { map }
    }
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

#[derive(Default)]
pub struct UdfSerializeRegistry {
    pub expr_rename_alias: Registry<Arc<dyn crate::dsl::RenameAliasFn>>,
    pub expr_series_udf: Registry<Arc<dyn crate::dsl::SeriesUdf>>,
    pub expr_fn_output_field: Registry<Arc<dyn crate::dsl::FunctionOutputField>>,
}
