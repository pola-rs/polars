use std::sync::Arc;

use polars::prelude::sync_on_close::SyncOnCloseType;
use polars::prelude::{CloudScheme, UnifiedSinkArgs};
use pyo3::prelude::*;

use crate::io::cloud_options::OptPyCloudOptions;
use crate::prelude::Wrap;

/// Interface to `class SinkOptions` on the Python side
pub struct PySinkOptions<'py>(Bound<'py, PyAny>);

impl<'a, 'py> FromPyObject<'a, 'py> for PySinkOptions<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(ob.to_owned()))
    }
}

impl PySinkOptions<'_> {
    pub fn extract_unified_sink_args(
        &self,
        cloud_scheme: Option<CloudScheme>,
    ) -> PyResult<UnifiedSinkArgs> {
        #[derive(FromPyObject)]
        struct Extract<'a> {
            mkdir: bool,
            maintain_order: bool,
            sync_on_close: Option<Wrap<SyncOnCloseType>>,
            storage_options: OptPyCloudOptions<'a>,
            credential_provider: Option<Py<PyAny>>,
        }

        let Extract {
            mkdir,
            maintain_order,
            sync_on_close,
            storage_options,
            credential_provider,
        } = self.0.extract()?;

        let cloud_options =
            storage_options.extract_opt_cloud_options(cloud_scheme, credential_provider)?;

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
