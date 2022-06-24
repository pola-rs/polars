use crate::dsl::{BinaryUdfOutputField, SeriesBinaryUdf, SpecialEq};
use crate::logical_plan::Context;
use polars_core::prelude::*;
use std::sync::Arc;

impl Default for SpecialEq<Arc<dyn SeriesBinaryUdf>> {
    fn default() -> Self {
        panic!("implementation error");
    }
}

impl Default for SpecialEq<Arc<dyn BinaryUdfOutputField>> {
    fn default() -> Self {
        let output_field = move |_: &Schema, _: Context, _: &Field, _: &Field| None;
        SpecialEq::new(Arc::new(output_field))
    }
}
