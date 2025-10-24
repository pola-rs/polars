use std::any::Any;
use std::sync::Arc;

use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_error::{PolarsResult, feature_gated, polars_bail};
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::SpecialEq;
use crate::dsl::LazySerde;

pub trait AnonymousStreamingAgg: Send + Sync {
    fn as_any(self: Arc<Self>) -> Box<dyn Any>;
    fn deep_clone(self: Arc<Self>) -> Arc<dyn AnonymousStreamingAgg>;

    fn get_field(&self, input_schema: &Schema, fields: &[Field]) -> PolarsResult<Field>;
}

pub type OpaqueStreamingAgg = LazySerde<SpecialEq<Arc<dyn AnonymousStreamingAgg>>>;

impl OpaqueStreamingAgg {
    pub fn materialize(self) -> PolarsResult<SpecialEq<Arc<dyn AnonymousStreamingAgg>>> {
        match self {
            Self::Deserialized(t) => Ok(t),
            Self::Named {
                name,
                payload,
                value,
            } => feature_gated!("serde", {
                use super::named_serde::NAMED_SERDE_REGISTRY_EXPR;
                match value {
                    Some(v) => Ok(v),
                    None => Ok(SpecialEq::new(
                        NAMED_SERDE_REGISTRY_EXPR
                            .read()
                            .unwrap()
                            .as_ref()
                            .expect("NAMED EXPR REGISTRY NOT SET")
                            .get_agg(&name, payload.unwrap().as_ref())
                            .expect("NAMED AGG NOT FOUND"),
                    )),
                }
            }),
            Self::Bytes(_b) => {
                unreachable!("not supported")
            },
        }
    }
}

#[cfg(feature = "ir_serde")]
impl Serialize for SpecialEq<Arc<dyn AnonymousStreamingAgg>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unreachable!("should not be hit")
    }
}

#[cfg(feature = "ir_serde")]
impl<'a> Deserialize<'a> for SpecialEq<Arc<dyn AnonymousStreamingAgg>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        unreachable!("should not be hit")
    }
}
