#![allow(unsafe_op_in_unsafe_fn)]
use std::any::Any;
use std::sync::OnceLock;

use polars::prelude::*;
use polars_core::chunked_array::object::builder::ObjectChunkedBuilder;
use polars_core::chunked_array::object::registry::AnonymousObjectBuilder;
use polars_core::chunked_array::object::{registry, set_polars_allow_extension};
use polars_error::PolarsWarning;
use polars_error::signals::register_polars_keyboard_interrupt_hook;
use polars_ffi::version_0::SeriesExport;
use polars_plan::plans::python_df_to_rust;
use polars_utils::python_convert_registry::{FromPythonConvertRegistry, PythonConvertRegistry};
use pyo3::prelude::*;
use pyo3::{IntoPyObjectExt, intern};

use crate::Wrap;
use crate::dataframe::PyDataFrame;
use crate::map::lazy::{ToSeries, call_lambda_with_series};
use crate::prelude::ObjectValue;
use crate::py_modules::{pl_df, pl_utils, polars, polars_rs};

fn python_function_caller_series(s: Column, lambda: &PyObject) -> PolarsResult<Column> {
    Python::with_gil(|py| {
        let object = call_lambda_with_series(py, s.as_materialized_series(), lambda)?;
        object.to_series(py, polars(py), s.name()).map(Column::from)
    })
}

fn python_function_caller_df(df: DataFrame, lambda: &PyObject) -> PolarsResult<DataFrame> {
    Python::with_gil(|py| {
        let pypolars = polars(py).bind(py);

        // create a PySeries struct/object for Python
        let mut pydf = PyDataFrame::new(df);
        // Wrap this PySeries object in the python side Series wrapper
        let mut python_df_wrapper = pypolars
            .getattr("wrap_df")
            .unwrap()
            .call1((pydf.clone(),))
            .unwrap();

        if !python_df_wrapper
            .getattr("_df")
            .unwrap()
            .is_instance(polars_rs(py).getattr(py, "PyDataFrame").unwrap().bind(py))
            .unwrap()
        {
            let pldf = pl_df(py).bind(py);
            let width = pydf.width();
            // Don't resize the Vec to avoid calling SeriesExport's Drop impl
            // The import takes ownership and is responsible for dropping
            let mut columns: Vec<SeriesExport> = Vec::with_capacity(width);
            unsafe {
                pydf._export_columns(columns.as_mut_ptr() as usize);
            }
            // Wrap this PyDataFrame object in the python side DataFrame wrapper
            python_df_wrapper = pldf
                .getattr("_import_columns")
                .unwrap()
                .call1((columns.as_mut_ptr() as usize, width))
                .unwrap();
        }
        // call the lambda and get a python side df wrapper
        let result_df_wrapper = lambda.call1(py, (python_df_wrapper,))?;

        // unpack the wrapper in a PyDataFrame
        let py_pydf = result_df_wrapper.getattr(py, "_df").map_err(|_| {
            let pytype = result_df_wrapper.bind(py).get_type();
            PolarsError::ComputeError(
                format!("Expected 'LazyFrame.map' to return a 'DataFrame', got a '{pytype}'",)
                    .into(),
            )
        })?;
        // Downcast to Rust
        match py_pydf.extract::<PyDataFrame>(py) {
            Ok(pydf) => Ok(pydf.df),
            Err(_) => python_df_to_rust(py, result_df_wrapper.into_bound(py)),
        }
    })
}

fn warning_function(msg: &str, warning: PolarsWarning) {
    Python::with_gil(|py| {
        let warn_fn = pl_utils(py)
            .bind(py)
            .getattr(intern!(py, "_polars_warn"))
            .unwrap();

        if let Err(e) = warn_fn.call1((msg, Wrap(warning).into_pyobject(py).unwrap())) {
            eprintln!("{e}")
        }
    });
}

static POLARS_REGISTRY_INIT_LOCK: OnceLock<()> = OnceLock::new();

/// # Safety
/// Caller must ensure that no other threads read the objects set by this registration.
pub unsafe fn register_startup_deps(catch_keyboard_interrupt: bool) {
    // TODO: should we throw an error if we try to initialize while already initialized?
    POLARS_REGISTRY_INIT_LOCK.get_or_init(|| {
        set_polars_allow_extension(true);

        // Stack frames can get really large in debug mode.
        #[cfg(debug_assertions)]
        {
            recursive::set_minimum_stack_size(1024 * 1024);
            recursive::set_stack_allocation_size(1024 * 1024 * 16);
        }

        // Register object type builder.
        let object_builder = Box::new(|name: PlSmallStr, capacity: usize| {
            Box::new(ObjectChunkedBuilder::<ObjectValue>::new(name, capacity))
                as Box<dyn AnonymousObjectBuilder>
        });

        let object_converter = Arc::new(|av: AnyValue| {
            let object = Python::with_gil(|py| ObjectValue {
                inner: Wrap(av).into_py_any(py).unwrap(),
            });
            Box::new(object) as Box<dyn Any>
        });
        let pyobject_converter = Arc::new(|av: AnyValue| {
            let object = Python::with_gil(|py| Wrap(av).into_py_any(py).unwrap());
            Box::new(object) as Box<dyn Any>
        });

        polars_utils::python_convert_registry::register_converters(PythonConvertRegistry {
            from_py: FromPythonConvertRegistry {
                sink_target: Arc::new(|py_f| {
                    Python::with_gil(|py| {
                        Ok(
                            Box::new(py_f.extract::<Wrap<polars_plan::dsl::SinkTarget>>(py)?.0)
                                as _,
                        )
                    })
                }),
            },
        });

        let object_size = size_of::<ObjectValue>();
        let physical_dtype = ArrowDataType::FixedSizeBinary(object_size);
        registry::register_object_builder(
            object_builder,
            object_converter,
            pyobject_converter,
            physical_dtype,
        );
        // Register SERIES UDF.
        python_dsl::CALL_COLUMNS_UDF_PYTHON = Some(python_function_caller_series);
        // Register DATAFRAME UDF.
        python_dsl::CALL_DF_UDF_PYTHON = Some(python_function_caller_df);
        // Register warning function for `polars_warn!`.
        polars_error::set_warning_function(warning_function);

        if catch_keyboard_interrupt {
            register_polars_keyboard_interrupt_hook();
        }
    });
}
