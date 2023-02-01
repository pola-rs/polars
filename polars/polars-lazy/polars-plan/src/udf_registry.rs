use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub use erased_serde::{
    Deserializer as ErasedDeserializer, Error as ErasedError, Serialize as ErasedSerialize,
};
use polars_core::export::once_cell::sync::OnceCell;
use serde::de::{DeserializeOwned, DeserializeSeed, Error, MapAccess, Visitor};
use serde::ser::SerializeMap;
pub use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Serialization

pub fn serialize_udf<S: Serializer>(
    ty: &str,
    obj: &dyn ErasedSerialize,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // { "type": <type>, "data": <data> }
    let mut map = serializer.serialize_map(Some(2))?;
    map.serialize_entry("type", ty)?;
    map.serialize_entry("data", obj)?;
    map.end()
}

// Deserialization

struct MapLookupVisitor<T: 'static> {
    registry: &'static Registry<T>,
}

impl<'de, T: 'static> Visitor<'de> for MapLookupVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "a user-defined function")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        // { "type": <type>, "data": <data> }
        let k = map
            .next_key::<&str>()?
            .ok_or_else(|| Error::missing_field("type"))?;
        if k != "type" {
            Err(Error::unknown_field(k, &["type", "data"]))?
        }

        let k = map.next_value::<&str>()?;

        let v = map
            .next_key::<&str>()?
            .ok_or_else(|| Error::missing_field("data"))?;
        if v != "data" {
            Err(Error::unknown_field(v, &["type", "data"]))?
        }

        let func = self
            .registry
            .map
            .get(k)
            .ok_or_else(|| Error::unknown_variant(k, &self.registry.variants))?;

        // Deserialize the data
        struct DeserSeedWrapper<'f, T> {
            f: &'f DeserializeFn<T>,
        }

        impl<'de, 'f, T: 'static> DeserializeSeed<'de> for DeserSeedWrapper<'f, T> {
            type Value = T;

            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                let mut erased = <dyn ErasedDeserializer>::erase(deserializer);
                (self.f)(&mut erased).map_err(Error::custom)
            }
        }
        let val = map.next_value_seed(DeserSeedWrapper { f: func })?;

        Ok(val)
    }
}

pub fn deserialize_udf<'de, D: Deserializer<'de>, T: 'static>(
    deser: D,
    registry: &'static Registry<T>,
) -> Result<T, D::Error> {
    deser.deserialize_map(MapLookupVisitor { registry })
}

// Deserialization: Registry

pub static UDF_DESERIALIZE_REGISTRY: OnceCell<UdfSerializeRegistry> = OnceCell::new();

pub type DeserializeFn<T> =
    Box<dyn Fn(&mut dyn ErasedDeserializer) -> Result<T, ErasedError> + Send + Sync>;

pub struct Registry<T> {
    map: HashMap<&'static str, DeserializeFn<T>>,
    variants: Vec<&'static str>,
}

impl<T> Registry<T> {
    pub fn new(map: HashMap<&'static str, DeserializeFn<T>>) -> Self {
        let variants = map.keys().copied().collect();
        Self { map, variants }
    }

    pub fn map(&self) -> &HashMap<&'static str, DeserializeFn<T>> {
        &self.map
    }

    pub fn map_mut(&mut self) -> &HashMap<&'static str, DeserializeFn<T>> {
        &mut self.map
    }

    pub fn with<D: DeserializeOwned>(
        mut self,
        key: &'static str,
        make_t: impl Fn(D) -> T + Send + Sync + 'static,
    ) -> Self {
        self.insert::<D>(key, make_t);
        self
    }

    pub fn insert<D: DeserializeOwned>(
        &mut self,
        key: &'static str,
        make_t: impl Fn(D) -> T + Send + Sync + 'static,
    ) -> &mut Self {
        self.map.insert(
            key,
            Box::new(move |deser: &mut dyn ErasedDeserializer| D::deserialize(deser).map(&make_t)),
        );

        self
    }
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            variants: Default::default(),
        }
    }
}

#[derive(Default)]
pub struct UdfSerializeRegistry {
    pub expr_rename_alias: Registry<Arc<dyn crate::dsl::RenameAliasFn>>,
    pub expr_series_udf: Registry<Arc<dyn crate::dsl::SeriesUdf>>,
    pub expr_fn_output_field: Registry<Arc<dyn crate::dsl::FunctionOutputField>>,
}
