use crate::dsl::{BinaryUdfOutputField, NoEq, SeriesBinaryUdf};
use crate::logical_plan::Context;
use polars_core::prelude::*;
use std::sync::Arc;

impl Default for NoEq<Arc<dyn SeriesBinaryUdf>> {
    fn default() -> Self {
        NoEq::new(Arc::new(|_, _| Err(PolarsError::ImplementationError)))
    }
}

impl Default for NoEq<Arc<dyn BinaryUdfOutputField>> {
    fn default() -> Self {
        let output_field = move |_: &Schema, _: Context, _: &Field, _: &Field| None;
        NoEq::new(Arc::new(output_field))
    }
}
