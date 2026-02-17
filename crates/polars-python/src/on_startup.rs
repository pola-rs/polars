#![allow(unsafe_op_in_unsafe_fn)]
use std::any::Any;
use std::sync::OnceLock;

use arrow::array::Array;
use polars::chunked_array::object::ObjectArray;
use polars::prelude::file_provider::FileProviderReturn;
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
use crate::lazyframe::PyLazyFrame;
use crate::map::lazy::call_lambda_with_series;
use crate::prelude::ObjectValue;
use crate::py_modules::{pl_df, pl_utils, polars, polars_rs};
use crate::series::PySeries;

fn python_function_caller_series(
    s: &[Column],
    output_dtype: Option<DataType>,
    lambda: &Py<PyAny>,
) -> PolarsResult<Column> {
    Python::attach(|py| call_lambda_with_series(py, s, output_dtype, lambda))
}

fn python_function_caller_df(df: DataFrame, lambda: &Py<PyAny>) -> PolarsResult<DataFrame> {
    Python::attach(|py| {
        let pypolars = polars(py).bind(py);

        // create a PySeries struct/object for Python
        let pydf = PyDataFrame::new(df);
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
            Ok(pydf) => Ok(pydf.df.into_inner()),
            Err(_) => python_df_to_rust(py, result_df_wrapper.into_bound(py)),
        }
    })
}

fn warning_function(msg: &str, warning: PolarsWarning) {
    Python::attach(|py| {
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

        #[cfg(feature = "backtrace_filter")]
        {
            use std::path::Path;
            use color_backtrace::{BacktracePrinter, default_output_stream, default_is_dependency_frame, Frame, ColorScheme};
            use color_backtrace::termcolor::{ColorSpec, Color};

            let polars_base_path = || {
                let on_startup = Path::new(file!()).canonicalize().ok()?;
                let src = on_startup.parent()?;
                let polars_python = src.parent()?;
                let crates = polars_python.parent()?;
                let root = crates.parent()?;
                Some(root.to_path_buf())
            };

            let mut btp = BacktracePrinter::default();
            if let Some(bp) = polars_base_path() {
                btp = btp.dependency_predicate(Box::new(move |frame: &Frame| -> bool {
                    if let Some(file) = frame.filename.as_ref().and_then(|f| f.canonicalize().ok()) {
                        !file.starts_with(&bp)
                    } else {
                        default_is_dependency_frame(frame)
                    }
                }));
            }

            let mut color_scheme = ColorScheme::classic();
            color_scheme.dependency_code = ColorSpec::new();
            color_scheme.dependency_code.set_dimmed(true);
            color_scheme.dependency_code = color_scheme.dependency_code_hash.clone();
            color_scheme.crate_code = ColorSpec::new();
            color_scheme.crate_code.set_fg(Some(Color::Blue));
            color_scheme.crate_code_hash = color_scheme.crate_code.clone();

            btp
                .color_scheme(color_scheme)
                .install(default_output_stream());
        }

        // Register object type builder.
        let object_builder = Box::new(|name: PlSmallStr, capacity: usize| {
            Box::new(ObjectChunkedBuilder::<ObjectValue>::new(name, capacity))
                as Box<dyn AnonymousObjectBuilder>
        });

        let object_converter = Arc::new(|av: AnyValue| {
            let object = Python::attach(|py| ObjectValue {
                inner: Wrap(av).into_py_any(py).unwrap(),
            });
            Box::new(object) as Box<dyn Any>
        });
        let pyobject_converter = Arc::new(|av: AnyValue| {
            let object = Python::attach(|py| Wrap(av).into_py_any(py).unwrap());
            Box::new(object) as Box<dyn Any>
        });
        fn object_array_getter(arr: &dyn Array, idx: usize) -> Option<AnyValue<'_>> {
            let arr = arr.as_any().downcast_ref::<ObjectArray<ObjectValue>>().unwrap();
            arr.get(idx).map(|v| AnyValue::Object(v))
        }
        fn with_gil(f: &mut dyn FnMut()) {
            Python::attach(|_| f())
        }

        polars_utils::python_convert_registry::register_converters(PythonConvertRegistry {
            from_py: FromPythonConvertRegistry {
                file_provider_result: Arc::new(|py_f| {
                    Python::attach(|py| {
                        Ok(Box::new(py_f.extract::<Wrap<FileProviderReturn>>(py)?.0) as _)
                    })
                }),
                series: Arc::new(|py_f| {
                    Python::attach(|py| {
                        Ok(Box::new(py_f.extract::<PySeries>(py)?.series.into_inner()) as _)
                    })
                }),
                df: Arc::new(|py_f| {
                    Python::attach(|py| {
                        Ok(Box::new(py_f.extract::<PyDataFrame>(py)?.df.into_inner()) as _)
                    })
                }),
                dsl_plan: Arc::new(|py_f| {
                    Python::attach(|py| {
                        Ok(Box::new(
                            py_f.extract::<PyLazyFrame>(py)?
                                .ldf
                                .into_inner()
                                .logical_plan,
                        ) as _)
                    })
                }),
                schema: Arc::new(|py_f| {
                    Python::attach(|py| {
                        Ok(Box::new(py_f.extract::<Wrap<polars_core::schema::Schema>>(py)?.0) as _)
                    })
                }),
            },
            to_py: polars_utils::python_convert_registry::ToPythonConvertRegistry {
                df: Arc::new(|df| {
                    Python::attach(|py| {
                        PyDataFrame::new(df.downcast_ref::<DataFrame>().unwrap().clone())
                            .into_py_any(py)
                    })
                }),
                series: Arc::new(|series| {
                    Python::attach(|py| {
                        PySeries::new(series.downcast_ref::<Series>().unwrap().clone())
                            .into_py_any(py)
                    })
                }),
                dsl_plan: Arc::new(|dsl_plan| {
                    Python::attach(|py| {
                        PyLazyFrame::from(LazyFrame::from(
                            dsl_plan
                                .downcast_ref::<polars_plan::dsl::DslPlan>()
                                .unwrap()
                                .clone(),
                        ))
                        .into_py_any(py)
                    })
                }),
                schema: Arc::new(|schema| {
                    Python::attach(|py| {
                        Wrap(
                            schema
                                .downcast_ref::<polars_core::schema::Schema>()
                                .unwrap()
                                .clone(),
                        )
                        .into_py_any(py)
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
            Arc::new(object_array_getter),
            Arc::new(with_gil)
        );

        use crate::dataset::dataset_provider_funcs;

        polars_plan::dsl::DATASET_PROVIDER_VTABLE.get_or_init(|| PythonDatasetProviderVTable {
            name: dataset_provider_funcs::name,
            schema: dataset_provider_funcs::schema,
            to_dataset_scan: dataset_provider_funcs::to_dataset_scan,
        });

        // Register SERIES UDF.
        python_dsl::CALL_COLUMNS_UDF_PYTHON = Some(python_function_caller_series);
        // Register DATAFRAME UDF.
        python_dsl::CALL_DF_UDF_PYTHON = Some(python_function_caller_df);
        // Register warning function for `polars_warn!`.
        polars_error::set_warning_function(warning_function);

        if catch_keyboard_interrupt {
            register_polars_keyboard_interrupt_hook();
        }

        use polars_core::datatypes::extension::UnknownExtensionTypeBehavior;
        let behavior = match std::env::var("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR").as_deref() {
            Ok("load_as_storage") => UnknownExtensionTypeBehavior::LoadAsStorage,
            Ok("load_as_extension") => UnknownExtensionTypeBehavior::LoadAsGeneric,
            Ok("") | Err(_) => UnknownExtensionTypeBehavior::WarnAndLoadAsStorage,
            _ => {
                polars_warn!("Invalid value for 'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR' environment variable. Expected one of 'load_as_storage' or 'load_as_extension'.");
                UnknownExtensionTypeBehavior::WarnAndLoadAsStorage
            },
        };
        polars_core::datatypes::extension::set_unknown_extension_type_behavior(behavior);
    });
}
