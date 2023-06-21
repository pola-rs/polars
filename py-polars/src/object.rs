use std::any::Any;
use std::sync::Arc;

use polars_core::chunked_array::object::builder::ObjectChunkedBuilder;
use polars_core::chunked_array::object::registry;
use polars_core::chunked_array::object::registry::AnonymousObjectBuilder;
use polars_core::error::PolarsError::ComputeError;
use polars_core::error::PolarsResult;
use polars_core::prelude::{AnyValue, Series};
use pyo3::prelude::*;

use crate::apply::lazy::{call_lambda_with_series, ToSeries};
use crate::prelude::{python_udf, ObjectValue};
use crate::py_modules::POLARS;
use crate::Wrap;

pub(crate) const OBJECT_NAME: &str = "object";

fn python_function_caller(s: Series, lambda: &PyObject) -> PolarsResult<Series> {
    Python::with_gil(|py| {
        let object = call_lambda_with_series(py, s.clone(), lambda)
            .map_err(|s| ComputeError(format!("{}", s).into()))?;
        object.to_series(py, &POLARS, s.name())
    })
}

#[pyfunction]
pub fn __register_startup_deps() {
    if !registry::is_object_builder_registered() {
        let object_builder = Box::new(|name: &str, capacity: usize| {
            Box::new(ObjectChunkedBuilder::<ObjectValue>::new(name, capacity))
                as Box<dyn AnonymousObjectBuilder>
        });

        let object_converter = Arc::new(|av: AnyValue| {
            let object = Python::with_gil(|py| ObjectValue {
                inner: Wrap(av).to_object(py),
            });
            Box::new(object) as Box<dyn Any>
        });

        registry::register_object_builder(object_builder, object_converter);
        unsafe { python_udf::CALL_LAMBDA = Some(python_function_caller) }
    }
}
