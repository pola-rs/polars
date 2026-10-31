//! A LazyFrame resolver of an I/O plugin (`polars.io_plugin.lazyframe_resolver.v1`).
//!
//! The plugin answers with a scan of a `file_reader.v1` builder, which becomes a
//! `DslPlan::Scan` of an [`ExternalReaderBuilder::Ffi`].
use std::fmt;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::runtime::ASYNC;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_err};
use polars_io::external_reader::ExternalReaderBuilder;
use polars_io::external_reader::ffi::FfiFileReaderBuilder;
use polars_io::io_plugin_ffi::{ffi_err_to_polars, import_struct_schema, lazyframe_resolver_v1};
use polars_io_ext_ffi::common::{FfiBuf, FfiSlice};
use polars_io_ext_ffi::lazyframe_resolver_v1::{Plugin, Resolve, ResolverHandle};
use polars_utils::aliases::PlIndexSet;
use polars_utils::arena::Arena;
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;

use super::filter_expr::filter_document;
use super::{DslResolver, DslResolverVariant, ResolveDslArgs, ResolvedDsl};
use crate::dsl::{DslBuilder, ScanSources, UnifiedScanArgs};
use crate::plans::{AExpr, ExprIR};

/// One reference to a resolver handle of the plugin `package`.
pub struct FfiDslResolver {
    package: PlSmallStr,
    plugin: &'static Plugin,
    handle: ResolverHandle,
}

// The contract makes resolver handles thread-safe.
unsafe impl Send for FfiDslResolver {}
unsafe impl Sync for FfiDslResolver {}

impl DslResolver {
    /// A resolver of the plugin `package`, taking ownership of one reference to `handle`.
    ///
    /// # Safety
    /// `handle` must be a live resolver handle of `package`'s `lazyframe_resolver.v1` contract.
    pub unsafe fn new_ffi(package: PlSmallStr, handle: ResolverHandle) -> PolarsResult<Self> {
        let plugin = lazyframe_resolver_v1(&package)?;
        Ok(Self {
            variant: DslResolverVariant::Ffi(FfiDslResolver {
                package,
                plugin,
                handle,
            }),
        })
    }
}

impl FfiDslResolver {
    fn err(&self, e: polars_io_ext_ffi::common::FfiError) -> polars_error::PolarsError {
        ffi_err_to_polars(e, &self.package)
    }

    fn fingerprint(&self) -> u64 {
        unsafe { (self.plugin.resolver_fingerprint)(self.handle) }
    }

    pub fn name(&self) -> PolarsResult<PlSmallStr> {
        let name: FfiBuf = unsafe { (self.plugin.resolver_name)(self.handle) };
        Ok(format!("{}: {}", self.package, name.to_string_lossy()).into())
    }

    pub fn cse_eq(&self, other: &Self) -> PolarsResult<bool> {
        Ok(self.package == other.package && self.fingerprint() == other.fingerprint())
    }

    pub fn schema(
        &self,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<SchemaRef>> + Send>>> {
        let this = self.clone();
        // The plugin may connect to find the schema: on a blocking thread.
        let fut = AbortOnDropHandle(ASYNC.spawn_blocking(move || {
            let schema = unsafe { (this.plugin.resolver_schema)(this.handle) }
                .into_result()
                .map_err(|e| this.err(e))?;
            Ok(Arc::new(import_struct_schema(schema, &this.package)?))
        }));
        Ok(Box::pin(async move { fut.await.unwrap() }))
    }

    pub fn resolve_dsl(
        &self,
        args: ResolveDslArgs,
        filters_eir: Buffer<ExprIR>,
        existing_resolved_version_key: Option<PlSmallStr>,
        expr_arena: &Arena<AExpr>,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<ResolvedDsl>> + Send>>> {
        let ResolveDslArgs {
            projection,
            slice,
            filters: _,
            filter_columns,
            filter_drop_columns_idx,
        } = args;

        #[allow(clippy::unnecessary_cast)] // `IdxSize` is `u32` without bigidx
        let idx_max = polars_utils::IdxSize::MAX as u64;
        let request = Resolve {
            projection: projection.map(|p| p.iter().map(|c| c.to_string()).collect()),
            // The resolver is only handed a limit (= offset + len), never the offset; the slice
            // is applied on top of the resolved scan.
            // An open-ended slice (`len` of `IdxSize::MAX` or more) sends no limit, and neither
            // does one that a database's `LIMIT` (a signed 64-bit integer) can't hold.
            limit: slice
                .filter(|(offset, len)| *offset >= 0 && *len < idx_max)
                .and_then(|(offset, len)| (offset as u64).checked_add(len))
                .filter(|limit| *limit <= i64::MAX as u64),
            filters: filters_eir
                .iter()
                .map(|e| filter_document(e.node(), expr_arena))
                .collect(),
            filter_columns: filter_columns.iter().map(|c| c.to_string()).collect(),
            filter_drop_columns_idx: filter_drop_columns_idx.map(|i| i as u64),
            existing_version_key: existing_resolved_version_key.map(|k| k.to_string()),
            target_n_sources: None,
            verbose: polars_core::config::verbose(),
        };

        let this = self.clone();
        // The plugin may connect to plan the scan: on a blocking thread.
        let fut = AbortOnDropHandle(ASYNC.spawn_blocking(move || {
            let resolved = request
                .with_ffi(|args| unsafe { (this.plugin.resolver_resolve)(this.handle, args) })
                .into_result()
                .map_err(|e| this.err(e))?;
            this.resolved_dsl(resolved)
        }));
        Ok(Box::pin(async move { fut.await.unwrap() }))
    }

    fn resolved_dsl(
        &self,
        resolved: polars_io_ext_ffi::lazyframe_resolver_v1::Resolved,
    ) -> PolarsResult<ResolvedDsl> {
        let polars_io_ext_ffi::lazyframe_resolver_v1::Resolved { scan, version_key } = resolved;
        let version_key = version_key
            .into_option()
            .map(|k| PlSmallStr::from(k.to_string_lossy()));
        let Some(scan) = scan.into_option() else {
            return Ok(ResolvedDsl {
                dsl: None,
                version_key,
                ..Default::default()
            });
        };
        let reader_package = PlSmallStr::from(scan.reader_package.to_string_lossy());
        // Owns the plugin's reference to the builder from here on, also on errors below.
        let reader =
            unsafe { FfiFileReaderBuilder::from_owned_handle(reader_package, scan.reader) }?;
        let schema = import_struct_schema(scan.schema, &self.package)?;
        let sources: Vec<PlRefPath> = scan
            .sources
            .as_slice()
            .iter()
            .map(|s| {
                std::str::from_utf8(s)
                    .map(PlRefPath::new)
                    .map_err(|e| polars_err!(ComputeError: "{}: invalid source: {e}", self.package))
            })
            .collect::<PolarsResult<_>>()?;

        let dsl = DslBuilder::from_external_reader_builder(
            ScanSources::Paths(Buffer::from_owner(sources)),
            ExternalReaderBuilder::Ffi(reader),
            UnifiedScanArgs {
                schema: Some(Arc::new(schema)),
                // Sources are the plugin's ids, not paths.
                expand_paths: false,
                ..Default::default()
            },
        )
        .build();

        Ok(ResolvedDsl {
            dsl: Some(dsl),
            version_key,
            applied_filters: scan
                .applied_filters
                .as_slice()
                .iter()
                .map(|&i| i as usize)
                .collect::<PlIndexSet<usize>>(),
            slice_offset_applied: false,
        })
    }

    fn serialize_bytes(&self) -> PolarsResult<FfiBuf> {
        unsafe { (self.plugin.resolver_serialize)(self.handle) }
            .into_result()
            .map_err(|e| self.err(e))
    }

    fn deserialize_bytes(package: PlSmallStr, bytes: &[u8]) -> PolarsResult<Self> {
        let plugin = lazyframe_resolver_v1(&package)?;
        let handle = unsafe { (plugin.resolver_deserialize)(FfiSlice::new(bytes)) }
            .into_result()
            .map_err(|e| ffi_err_to_polars(e, &package))?;
        Ok(Self {
            package,
            plugin,
            handle,
        })
    }
}

impl Clone for FfiDslResolver {
    fn clone(&self) -> Self {
        Self {
            package: self.package.clone(),
            plugin: self.plugin,
            handle: unsafe { (self.plugin.resolver_clone)(self.handle) },
        }
    }
}

impl Drop for FfiDslResolver {
    fn drop(&mut self) {
        unsafe { (self.plugin.resolver_release)(self.handle) }
    }
}

impl fmt::Debug for FfiDslResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FfiDslResolver")
            .field("package", &self.package)
            .finish()
    }
}

impl Hash for FfiDslResolver {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.package.hash(state);
        self.fingerprint().hash(state);
    }
}

#[cfg(feature = "serde")]
mod _serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    #[derive(Serialize, Deserialize)]
    struct Serialized {
        package: PlSmallStr,
        #[serde(with = "bytes")]
        bytes: Vec<u8>,
    }

    mod bytes {
        use serde::{Deserializer, Serializer};

        pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
            s.serialize_bytes(v)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
            polars_utils::pl_serialize::deserialize_map_bytes(d, |b| b.into_owned())
        }
    }

    impl Serialize for FfiDslResolver {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::Error;
            let bytes = self.serialize_bytes().map_err(S::Error::custom)?;
            Serialized {
                package: self.package.clone(),
                bytes: bytes.to_vec(),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for FfiDslResolver {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            use serde::de::Error;
            let s = Serialized::deserialize(deserializer)?;
            Self::deserialize_bytes(s.package, &s.bytes).map_err(D::Error::custom)
        }
    }
}
