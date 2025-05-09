use polars::prelude::{CastColumnsPolicy, ExtraColumnsPolicy, MissingColumnsPolicy};
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyAnyMethods, PyModule};
use pyo3::{Bound, FromPyObject, PyAny, PyResult};

/// Interface to `class ScanOptions` on the Python side
pub struct PyScanOptions<'py>(Bound<'py, pyo3::PyAny>);

impl<'py> FromPyObject<'py> for PyScanOptions<'py> {
    fn extract_bound(ob: &Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(Self(ob.clone()))
    }
}

impl<'py> std::ops::Deref for PyScanOptions<'py> {
    type Target = Bound<'py, pyo3::PyAny>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyScanOptions<'_> {
    pub fn extra_columns_policy(&self) -> PyResult<ExtraColumnsPolicy> {
        Ok(
            match &*self.getattr("extra_columns")?.extract::<PyBackedStr>()? {
                "ignore" => ExtraColumnsPolicy::Ignore,
                "raise" => ExtraColumnsPolicy::Raise,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "unknown option for extra_columns: {}",
                        v
                    )));
                },
            },
        )
    }

    pub fn extract_cast_options(&self) -> PyResult<CastColumnsPolicy> {
        let ob = self;

        if ob.is_none() {
            // Initialize the default ScanCastOptions from Python.

            static DEFAULT: GILOnceCell<CastColumnsPolicy> = GILOnceCell::new();

            let out = DEFAULT.get_or_try_init(ob.py(), || {
                let ob = PyModule::import(ob.py(), "polars.io.cast_options")
                    .unwrap()
                    .getattr("ScanCastOptions")
                    .unwrap()
                    .call_method0("_default")
                    .unwrap();

                let out = Self(ob).extract_cast_options()?;

                // The default policy should match ERROR_ON_MISMATCH (but this can change).
                debug_assert_eq!(&out, &CastColumnsPolicy::ERROR_ON_MISMATCH);

                PyResult::Ok(out)
            })?;

            return Ok(out.clone());
        }

        let integer_upcast = match &*self.getattr("integer_cast")?.extract::<PyBackedStr>()? {
            "upcast" => true,
            "forbid" => false,
            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown option for integer_cast: {}",
                    v
                )));
            },
        };

        let mut float_upcast = false;
        let mut float_downcast = false;

        let float_cast_object = self.getattr("float_cast")?;

        parse_multiple_options("float_cast", float_cast_object, |v| {
            match v {
                "forbid" => {},
                "upcast" => float_upcast = true,
                "downcast" => float_downcast = true,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "unknown option for float_cast: {}",
                        v
                    )));
                },
            }

            Ok(())
        })?;

        let mut datetime_nanoseconds_downcast = false;
        let mut datetime_convert_timezone = false;

        let datetime_cast_object = self.getattr("datetime_cast")?;

        parse_multiple_options("datetime_cast", datetime_cast_object, |v| {
            match v {
                "forbid" => {},
                "nanosecond-downcast" => datetime_nanoseconds_downcast = true,
                "convert-timezone" => datetime_convert_timezone = true,
                v => {
                    return Err(PyValueError::new_err(format!(
                        "unknown option for datetime_cast: {}",
                        v
                    )));
                },
            };

            Ok(())
        })?;

        let missing_struct_fields = match &*self
            .getattr("missing_struct_fields")?
            .extract::<PyBackedStr>()?
        {
            "insert" => MissingColumnsPolicy::Insert,
            "raise" => MissingColumnsPolicy::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown option for missing_struct_fields: {}",
                    v
                )));
            },
        };

        let extra_struct_fields = match &*self
            .getattr("extra_struct_fields")?
            .extract::<PyBackedStr>()?
        {
            "ignore" => ExtraColumnsPolicy::Ignore,
            "raise" => ExtraColumnsPolicy::Raise,
            v => {
                return Err(PyValueError::new_err(format!(
                    "unknown option for extra_struct_fields: {}",
                    v
                )));
            },
        };

        Ok(CastColumnsPolicy {
            integer_upcast,
            float_upcast,
            float_downcast,
            datetime_nanoseconds_downcast,
            datetime_convert_timezone,
            missing_struct_fields,
            extra_struct_fields,
        })
    }
}

fn parse_multiple_options(
    parameter_name: &'static str,
    py_object: Bound<'_, PyAny>,
    mut parser_func: impl FnMut(&str) -> PyResult<()>,
) -> PyResult<()> {
    if let Ok(v) = py_object.extract::<PyBackedStr>() {
        parser_func(&v)?;
    } else if let Ok(v) = py_object.try_iter() {
        for v in v {
            parser_func(&v?.extract::<PyBackedStr>()?)?;
        }
    } else {
        return Err(PyValueError::new_err(format!(
            "unknown type for {}: {}",
            parameter_name, py_object
        )));
    }

    Ok(())
}
