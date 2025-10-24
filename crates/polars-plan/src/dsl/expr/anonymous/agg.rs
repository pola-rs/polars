use std::any::Any;
use std::sync::Arc;

use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_error::{PolarsResult, polars_bail};

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
