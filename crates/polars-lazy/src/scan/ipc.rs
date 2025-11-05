use arrow::buffer::Buffer;
use polars_core::prelude::*;
use polars_io::ipc::IpcScanOptions;
use polars_utils::plpath::PlPath;

use crate::prelude::*;

impl LazyFrame {
    /// Create a LazyFrame directly from a ipc scan.
    pub fn scan_ipc(
        path: PlPath,
        options: IpcScanOptions,
        unified_scan_args: UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        Self::scan_ipc_sources(
            ScanSources::Paths(Buffer::from_iter([path])),
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
