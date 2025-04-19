use std::path::Path;

use polars_core::prelude::*;
use polars_io::ipc::IpcScanOptions;

use crate::prelude::*;

impl LazyFrame {
    /// Create a LazyFrame directly from a ipc scan.
    pub fn scan_ipc(
        path: impl AsRef<Path>,
        options: IpcScanOptions,
        unified_scan_args: UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        Self::scan_ipc_sources(
            ScanSources::Paths([path.as_ref().to_path_buf()].into()),
            options,
            unified_scan_args,
        )
    }

    pub fn scan_ipc_sources(
        sources: ScanSources,
        options: IpcScanOptions,
        unified_scan_args: UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        let lf = DslBuilder::scan_ipc(sources, options, unified_scan_args)?
            .build()
            .into();

        Ok(lf)
    }
}
