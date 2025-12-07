use std::sync::Arc;

use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{CloudScheme, UnifiedSinkArgs};
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};

use crate::functions::parse_cloud_options;
use crate::prelude::Wrap;

/// Interface to `class SinkOptions` on the Python side
pub struct PySinkOptions<'py>(Bound<'py, pyo3::PyAny>);

impl<'py> FromPyObject<'py> for PySinkOptions<'py> {
    fn extract_bound(ob: &Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Ok(Self(ob.clone()))
    }
}

impl PySinkOptions<'_> {
    pub fn extract_unified_sink_args(
        &self,
        cloud_scheme: Option<CloudScheme>,
    ) -> PyResult<UnifiedSinkArgs> {
        #[derive(FromPyObject)]
        struct Extract {
            mkdir: bool,
            maintain_order: bool,
            sync_on_close: Option<Wrap<SyncOnCloseType>>,
            storage_options: Option<Vec<(String, String)>>,
            credential_provider: Option<Py<PyAny>>,
            retries: usize,
        }

        let Extract {
            mkdir,
            maintain_order,
            sync_on_close,
            storage_options,
            credential_provider,
            retries,
        } = self.0.extract()?;

        let cloud_options =
            parse_cloud_options(cloud_scheme, storage_options, credential_provider, retries)?;

        let sync_on_close = sync_on_close.map_or(SyncOnCloseType::default(), |x| x.0);

        let unified_sink_args = UnifiedSinkArgs {
            mkdir,
            maintain_order,
            sync_on_close,
            cloud_options: cloud_options.map(Arc::new),
        };

        Ok(unified_sink_args)
    }
}
