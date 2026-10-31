//! Host side of the generic I/O plugin contracts (`file_reader.v1`, `lazyframe_resolver.v1`).
//!
//! Plugins are Python packages. [`install_locator`] lets the engine find a plugin's contracts by
//! package name (also when a plan is deserialized), with the handshake of `polars_io_ext_ffi`:
//! `_polars_io_plugin_ids` and `_capsule(id)`. Afterwards the engine calls the plugin directly,
//! without Python.
use std::ffi::CStr;

use polars_io_ext_ffi::lazyframe_resolver_v1::{self, ResolverHandle};
use polars_io_ext_ffi::registry;
use polars_plan::dsl::DslBuilder;
use polars_plan::dsl::dsl_resolver::DslResolver;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

use crate::error::PyPolarsErr;
use crate::lazyframe::PyLazyFrame;
use crate::prelude::*;

/// The pip requirement that installs a plugin package.
fn install_hint(package: &str) -> String {
    match package {
        "polars_database" => "pip install 'polars[database]'".to_string(),
        p => format!("pip install {}", p.replace('_', "-")),
    }
}

fn locate(py: Python<'_>, package: &str, id: &CStr) -> Result<*const std::ffi::c_void, String> {
    let id_str = id.to_string_lossy();
    let module = py.import(package).map_err(|e| {
        format!(
            "the I/O plugin {package} could not be imported ({e}); install it with: {}",
            install_hint(package)
        )
    })?;
    let ids: Vec<String> = module
        .getattr("_polars_io_plugin_ids")
        .and_then(|ids| ids.extract())
        .map_err(|e| format!("{package} is not a Polars I/O plugin: {e}"))?;
    if !ids.iter().any(|i| *i == id_str) {
        return Err(format!(
            "the installed {package} does not support {id_str} (it supports {ids:?}); \
            install a version compatible with this Polars: {}",
            install_hint(package)
        ));
    }
    let capsule = module
        .call_method1("_capsule", (id_str.as_ref(),))
        .map_err(|e| format!("{package}: {e}"))?;
    let capsule = capsule
        .cast::<PyCapsule>()
        .map_err(|e| format!("{package}: {e}"))?;
    let ptr = capsule
        .pointer_checked(Some(id))
        .map_err(|e| format!("{package}: invalid capsule: {e}"))?;
    Ok(ptr.as_ptr() as *const _)
}

/// Lets the engine find I/O plugins through their Python packages. Idempotent.
pub fn install_locator() {
    registry::set_locator(Box::new(|package, id| {
        Python::attach(|py| locate(py, package, id))
    }));
}

#[pymethods]
impl PyLazyFrame {
    /// A LazyFrame of the resolver that the I/O plugin `package` handed over in `capsule`
    /// (`lazyframe_resolver_v1::HANDLE_CAPSULE_NAME`).
    #[staticmethod]
    fn _from_io_plugin_resolver(package: &str, capsule: &Bound<'_, PyAny>) -> PyResult<Self> {
        install_locator();
        let capsule = capsule.cast::<PyCapsule>()?;
        let ptr = capsule.pointer_checked(Some(lazyframe_resolver_v1::HANDLE_CAPSULE_NAME))?;
        // SAFETY: a capsule of this name points to a live resolver handle of the plugin.
        let handle = unsafe { *(ptr.as_ptr() as *const ResolverHandle) };
        let plugin =
            polars_io::io_plugin_ffi::lazyframe_resolver_v1(package).map_err(PyPolarsErr::from)?;
        // The capsule keeps its own reference.
        let handle = unsafe { (plugin.resolver_clone)(handle) };
        let resolver = unsafe { DslResolver::new_ffi(PlSmallStr::from(package), handle) }
            .map_err(PyPolarsErr::from)?;
        Ok(
            LazyFrame::from(DslBuilder::from_dsl_resolver(std::sync::Arc::new(resolver)).build())
                .into(),
        )
    }
}
