use std::any::Any;
use std::sync::Arc;

use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_error::{PolarsResult, feature_gated, polars_bail};

use super::SpecialEq;
use crate::dsl::LazySerde;

pub trait AnonymousStreamingAgg {
    fn as_any(self: Arc<Self>) -> Box<dyn Any>;
    fn deep_clone(self: Arc<Self>) -> Arc<dyn AnonymousStreamingAgg>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this 'opaque' function")
    }

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
