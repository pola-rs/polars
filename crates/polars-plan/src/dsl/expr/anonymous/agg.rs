use std::any::Any;
use std::sync::Arc;

use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_error::{PolarsResult, feature_gated};
#[cfg(feature = "ir_serde")]
use serde::{Serialize, Serializer};

use super::SpecialEq;
use crate::dsl::LazySerde;

pub trait AnonymousStreamingAgg: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn get_field(&self, input_schema: &Schema, fields: &[Field]) -> PolarsResult<Field>;
}

pub type OpaqueStreamingAgg = LazySerde<SpecialEq<Arc<dyn AnonymousStreamingAgg>>>;

impl OpaqueStreamingAgg {
    pub fn materialize(&self) -> PolarsResult<SpecialEq<Arc<dyn AnonymousStreamingAgg>>> {
        match self {
            Self::Deserialized(t) => Ok(t.clone()),
            Self::Named {
                name,
                payload,
                value,
            } => feature_gated!("serde", {
                use super::named_serde::NAMED_SERDE_REGISTRY_EXPR;
                match value {
                    Some(v) => Ok(v.clone()),
                    None => Ok(SpecialEq::new(
                        NAMED_SERDE_REGISTRY_EXPR
                            .read()
                            .unwrap()
                            .as_ref()
                            .expect("NAMED EXPR REGISTRY NOT SET")
                            .get_agg(name, payload.as_ref().unwrap())?
                            .expect("NAMED AGG NOT FOUND"),
                    )),
                }
            }),
            Self::Bytes(_b) => {
                feature_gated!("serde", {
                    use crate::dsl::anonymous::serde_expr;
                    serde_expr::deserialize_anon_agg(_b.as_ref()).map(SpecialEq::new)
                })
            },
        }
    }
}

#[cfg(feature = "ir_serde")]
impl Serialize for SpecialEq<Arc<dyn AnonymousStreamingAgg>> {
    fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unreachable!("should not be hit")
    }
}
