use crate::dsl::{BinaryUdfOutputField, NoEq, SeriesBinaryUdf};
use crate::logical_plan::Context;
use crate::prelude::*;
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

pub(crate) fn dummy_aexpr_binary_fn() -> AExpr {
    AExpr::BinaryFunction {
        input_a: Default::default(),
        input_b: Default::default(),
        function: Default::default(),
        output_field: Default::default(),
    }
}

pub(crate) fn dummy_aexpr_sort_by() -> AExpr {
    AExpr::SortBy {
        expr: Default::default(),
        by: Default::default(),
        reverse: Default::default(),
    }
}
pub(crate) fn dummy_aexpr_filter() -> AExpr {
    AExpr::Filter {
        input: Default::default(),
        by: Default::default(),
    }
}
