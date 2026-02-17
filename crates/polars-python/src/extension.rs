use std::any::Any;
use std::borrow::Cow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

use polars::prelude::PlFixedStateQuality;
use polars::prelude::extension::{register_extension_type, unregister_extension_type};
use polars_core::datatypes::DataType;
use polars_core::datatypes::extension::{ExtensionTypeFactory, ExtensionTypeImpl};
use pyo3::prelude::*;

use crate::prelude::Wrap;
use crate::utils::to_py_err;

struct PyExtensionTypeFactory {
    cls: Arc<Py<PyAny>>,
}

#[derive(Clone)]
struct PyExtensionTypeImpl {
    name: String,
    display: String,
    metadata: Option<String>,
}

impl ExtensionTypeFactory for PyExtensionTypeFactory {
    fn create_type_instance(
        &self,
        name: &str,
        storage: &DataType,
        metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        Python::attach(|py| {
            let typ_obj = self
                .cls
                .bind(py)
                .call_method1("ext_from_params", (name, &Wrap(storage.clone()), metadata))
                .unwrap();

            let display = typ_obj
                .call_method0("_string_repr")
                .unwrap()
                .extract()
                .unwrap();
            let metadata = typ_obj
                .call_method0("ext_metadata")
                .unwrap()
                .extract()
                .unwrap();

            Box::new(PyExtensionTypeImpl {
                name: name.to_string(),
                display,
                metadata,
            })
        })
    }
}

impl ExtensionTypeImpl for PyExtensionTypeImpl {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.metadata.as_deref().map(Cow::Borrowed)
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> {
        Box::new(self.clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        let Some(other) = (other as &dyn Any).downcast_ref::<PyExtensionTypeImpl>() else {
            return false;
        };

        self.name == other.name && self.metadata == other.metadata
    }

    fn dyn_hash(&self) -> u64 {
        let mut hasher = PlFixedStateQuality::default().build_hasher();
        self.name.hash(&mut hasher);
        self.metadata.hash(&mut hasher);
        hasher.finish()
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.display)
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        if let Some(md) = &self.metadata {
            Cow::Owned(format!(
                "PyExtensionType(name='{}', metadata='{}')",
                self.name, md
            ))
        } else {
            Cow::Owned(format!("PyExtensionType(name='{}')", self.name))
        }
    }
}

#[pyfunction]
pub fn _register_extension_type(name: &str, cls: Option<&Bound<PyAny>>) -> PyResult<()> {
    register_extension_type(
        name,
        cls.map(|c| {
            Arc::new(PyExtensionTypeFactory {
                cls: Arc::new(c.clone().unbind()),
            }) as _
        }),
    )
    .map_err(to_py_err)
}

#[pyfunction]
pub fn _unregister_extension_type(name: &str) -> PyResult<()> {
    unregister_extension_type(name).map(drop).map_err(to_py_err)
}
