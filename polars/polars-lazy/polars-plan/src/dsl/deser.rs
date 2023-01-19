use std::fmt;

use serde::de::Visitor;

use super::Expr;
use crate::udf_registry::{RegistryDeserializable, UdfSerializeRegistry};

impl<'de> RegistryDeserializable<'de> for Expr {
    fn deserialize_with_registry<D: serde::Deserializer<'de>>(
        deser: D,
        registry: &UdfSerializeRegistry,
    ) -> Result<Self, D::Error>
    where
        Self: Sized,
    {
        // here, we would have same thing as the code #[derive(Deserialize)]
        //  generates. This is fairly big, so we may want helpers/a derive macro
        todo!()
    }
}
