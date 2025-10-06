use std::fmt;
use std::sync::Arc;

use polars_error::PolarsResult;

use crate::dsl::SpecialEq;

#[derive(Eq, PartialEq)]
pub enum PlanCallback<Args, Out> {
    #[cfg(feature = "python")]
    Python(SpecialEq<Arc<polars_utils::python_function::PythonFunction>>),
    Rust(SpecialEq<Arc<dyn Fn(Args) -> PolarsResult<Out> + Send + Sync>>),
}

impl<Args, Out> fmt::Debug for PlanCallback<Args, Out> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PlanCallback::")?;
        std::mem::discriminant(self).fmt(f)
    }
}

#[cfg(feature = "serde")]
impl<Args, Out> serde::Serialize for PlanCallback<Args, Out> {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;

        #[cfg(feature = "python")]
        if let Self::Python(v) = self {
            return v.serialize(_serializer);
        }

        Err(S::Error::custom(format!(
            "cannot serialize 'opaque' function in {self:?}"
        )))
    }
}

#[cfg(feature = "serde")]
impl<'de, Args, Out> serde::Deserialize<'de> for PlanCallback<Args, Out> {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[cfg(feature = "python")]
        {
            Ok(Self::Python(SpecialEq::new(Arc::new(
                polars_utils::python_function::PythonFunction::deserialize(_deserializer)?,
            ))))
        }
        #[cfg(not(feature = "python"))]
        {
            use serde::de::Error;
            Err(D::Error::custom("cannot deserialize PlanCallback"))
        }
    }
}

#[cfg(feature = "dsl-schema")]
impl<Args, Out> schemars::JsonSchema for PlanCallback<Args, Out> {
    fn schema_name() -> String {
        "PlanCallback".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "PlanCallback"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

impl<Args, Out> std::hash::Hash for PlanCallback<Args, Out> {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        // no-op.
    }
}

impl<Args, Out> Clone for PlanCallback<Args, Out> {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "python")]
            Self::Python(p) => Self::Python(p.clone()),
            Self::Rust(f) => Self::Rust(f.clone()),
        }
    }
}

pub trait PlanCallbackArgs {
    #[cfg(feature = "python")]
    fn into_pyany<'py>(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>>;
}
pub trait PlanCallbackOut: Sized {
    #[cfg(feature = "python")]
    fn from_pyany<'py>(pyany: pyo3::Py<pyo3::PyAny>, py: pyo3::Python<'py>)
    -> pyo3::PyResult<Self>;
}

#[cfg(feature = "python")]
mod _python {
    use polars_utils::pl_str::PlSmallStr;
    use pyo3::types::{PyAnyMethods, PyTuple};
    use pyo3::*;

    macro_rules! impl_pycb_type {
        ($($type:ty),+) => {
            $(
            impl super::PlanCallbackArgs for $type {
                fn into_pyany<'py>(self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                    Ok(self.into_pyobject(py)?.into_any().unbind())
                }
            }

            impl super::PlanCallbackOut for $type {
                fn from_pyany<'py>(pyany: Py<PyAny>, py: Python<'py>) -> PyResult<Self> {
                    pyany.bind(py).extract::<Self>()
                }
            }
            )+
        };
    }

    macro_rules! impl_pycb_type_to_from {
        ($($type:ty => $transformed:ty),+) => {
            $(
            impl super::PlanCallbackArgs for $type {
                fn into_pyany<'py>(self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                    Ok(<$transformed>::from(self).into_pyobject(py)?.into_any().unbind())
                }
            }

            impl super::PlanCallbackOut for $type {
                fn from_pyany<'py>(pyany: Py<PyAny>, py: Python<'py>) -> PyResult<Self> {
                    pyany.bind(py).extract::<$transformed>().map(Into::into)
                }
            }
            )+
        };
    }

    macro_rules! impl_registrycb_type {
        ($(($type:path, $from:ident, $to:ident)),+) => {
            $(
            impl super::PlanCallbackArgs for $type {
                fn into_pyany<'py>(self, _py: Python<'py>) -> PyResult<Py<PyAny>> {
                    let registry = polars_utils::python_convert_registry::get_python_convert_registry();
                    (registry.to_py.$to)(Box::new(self) as _)
                }
            }

            impl super::PlanCallbackOut for $type {
                fn from_pyany<'py>(pyany: Py<PyAny>, _py: Python<'py>) -> PyResult<Self> {
                    let registry = polars_utils::python_convert_registry::get_python_convert_registry();
                    let obj = (registry.from_py.$from)(pyany)?;
                    let obj = obj.downcast().unwrap();
                    Ok(*obj)
                }
            }
            )+
        };
    }

    impl<T: super::PlanCallbackArgs> super::PlanCallbackArgs for Option<T> {
        fn into_pyany<'py>(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            match self {
                None => Ok(py.None()),
                Some(v) => v.into_pyany(py),
            }
        }
    }

    impl<T: super::PlanCallbackOut> super::PlanCallbackOut for Option<T> {
        fn from_pyany<'py>(
            pyany: pyo3::Py<pyo3::PyAny>,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<Self> {
            if pyany.is_none(py) {
                Ok(None)
            } else {
                T::from_pyany(pyany, py).map(Some)
            }
        }
    }

    impl<T, U> super::PlanCallbackArgs for (T, U)
    where
        T: super::PlanCallbackArgs,
        U: super::PlanCallbackArgs,
    {
        fn into_pyany<'py>(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            PyTuple::new(py, [self.0.into_pyany(py)?, self.1.into_pyany(py)?])?.into_py_any(py)
        }
    }

    impl<T, U> super::PlanCallbackOut for (T, U)
    where
        T: super::PlanCallbackOut,
        U: super::PlanCallbackOut,
    {
        fn from_pyany<'py>(
            pyany: pyo3::Py<pyo3::PyAny>,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<Self> {
            use pyo3::prelude::*;
            let tuple = pyany.downcast_bound::<PyTuple>(py)?;
            Ok((
                T::from_pyany(tuple.get_item(0)?.unbind(), py)?,
                U::from_pyany(tuple.get_item(1)?.unbind(), py)?,
            ))
        }
    }

    impl_pycb_type! {
        bool,
        usize,
        String
    }
    impl_pycb_type_to_from! {
        PlSmallStr => String
    }
    impl_registrycb_type! {
        (polars_core::series::Series, series, series),
        (polars_core::frame::DataFrame, df, df),
        (crate::dsl::DslPlan, dsl_plan, dsl_plan),
        (polars_core::schema::Schema, schema, schema)
    }
}

#[cfg(not(feature = "python"))]
mod _no_python {
    impl<T> super::PlanCallbackArgs for T {}
    impl<T: Sized> super::PlanCallbackOut for T {}
}

impl<Args: PlanCallbackArgs, Out: PlanCallbackOut> PlanCallback<Args, Out> {
    pub fn call(&self, args: Args) -> PolarsResult<Out> {
        match self {
            #[cfg(feature = "python")]
            Self::Python(pyfn) => pyo3::Python::attach(|py| {
                let out = Out::from_pyany(pyfn.call1(py, (args.into_pyany(py)?,))?, py)?;
                Ok(out)
            }),
            Self::Rust(f) => f(args),
        }
    }

    #[cfg(feature = "python")]
    pub fn new_python(pyfn: polars_utils::python_function::PythonFunction) -> Self {
        Self::Python(SpecialEq::new(Arc::new(pyfn)))
    }

    pub fn new(f: impl Fn(Args) -> PolarsResult<Out> + Send + Sync + 'static) -> Self {
        Self::Rust(SpecialEq::new(Arc::new(f) as _))
    }
}
