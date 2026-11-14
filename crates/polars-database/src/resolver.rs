//! `polars.io_plugin.lazyframe_resolver.v1`: planning a database scan.
//!
//! The first call that needs the schema resolves the scan (dialect, backend, schema, catalog
//! metadata; [`resolve_scan`]) and keeps the result for the life of the resolver. Each
//! resolution then translates the pushed filters, resolves the partitions and renders their SQL
//! ([`compile_scan`]) into a [`ReaderBuilder`], whose sources are the partition ids.
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_err};
use polars_io_ext_ffi::common::{ArrowSchema, FfiBuf, FfiResult, FfiSlice, FfiVec};
use polars_io_ext_ffi::file_reader_v1::BuilderHandle;
use polars_io_ext_ffi::lazyframe_resolver_v1::{
    Plugin, Resolve, ResolveArgs, Resolved, ResolvedScan, ResolverHandle,
};
use serde::{Deserialize, Serialize};

use crate::database::{
    DatabaseScanConfig, DatabaseScanOptions, PushedFilter, ScanPushdown, compile_scan, resolve_scan,
};
use crate::ffi::{
    borrow_handle, clone_handle, export_schema, ffi_call, ffi_call_infallible, into_handle,
    release_handle,
};
use crate::filter::{Class, Node, Translator};
use crate::reader::ReaderBuilder;

/// Prefix of serialized resolvers, so that a change of the format fails clearly.
const SERDE_MAGIC: &[u8] = b"PLDBRS1";

pub struct Resolver {
    options: Arc<DatabaseScanOptions>,
    /// The user's `schema=`.
    user_schema: Option<SchemaRef>,
    /// Set by the first call that needs the schema.
    config: Mutex<Option<Arc<DatabaseScanConfig>>>,
}

#[derive(Serialize, Deserialize)]
struct SerializedResolver {
    options: Arc<DatabaseScanOptions>,
    user_schema: Option<SchemaRef>,
}

impl Resolver {
    pub fn new(options: DatabaseScanOptions, user_schema: Option<SchemaRef>) -> Self {
        Self {
            options: Arc::new(options),
            user_schema,
            config: Mutex::new(None),
        }
    }

    /// The resolved scan. Connects on the first call; the result is kept.
    fn config(&self) -> PolarsResult<Arc<DatabaseScanConfig>> {
        let mut config = self.config.lock().unwrap();
        if let Some(c) = &*config {
            return Ok(c.clone());
        }
        let c = Arc::new(resolve_scan(
            self.options.clone(),
            self.user_schema.clone(),
        )?);
        *config = Some(c.clone());
        Ok(c)
    }

    fn fingerprint(&self) -> u64 {
        let mut h = polars_utils::aliases::PlFixedStateQuality::default().build_hasher();
        self.options.fingerprint().hash(&mut h);
        if let Some(schema) = &self.user_schema {
            for (name, dtype) in schema.iter() {
                name.hash(&mut h);
                dtype.hash(&mut h);
            }
        }
        h.finish()
    }

    fn serialize(&self) -> PolarsResult<Vec<u8>> {
        let mut out = SERDE_MAGIC.to_vec();
        out.extend(polars_utils::pl_serialize::serialize_to_bytes::<_, false>(
            &SerializedResolver {
                options: self.options.clone(),
                user_schema: self.user_schema.clone(),
            },
        )?);
        Ok(out)
    }

    fn deserialize(bytes: &[u8]) -> PolarsResult<Self> {
        let bytes = bytes.strip_prefix(SERDE_MAGIC).ok_or_else(|| {
            polars_err!(
                ComputeError:
                "cannot deserialize a scan_database plan: it was made by an incompatible \
                version of {}", crate::PACKAGE_NAME
            )
        })?;
        let s: SerializedResolver =
            polars_utils::pl_serialize::deserialize_from_reader::<_, _, false>(bytes)?;
        Ok(Self {
            options: s.options,
            user_schema: s.user_schema,
            config: Mutex::new(None),
        })
    }

    pub fn resolve(&self, args: &Resolve) -> PolarsResult<(ResolvedScanOwned, Vec<u64>)> {
        let config = self.config()?;

        let columns: Option<Vec<PlSmallStr>> = args.projection.as_ref().map(|p| {
            p.iter()
                .filter(|c| config.schema.contains(c))
                .map(|c| PlSmallStr::from(c.as_str()))
                .collect()
        });
        let mut pushdown = ScanPushdown {
            columns: columns.clone(),
            total_filters: args.filters.len(),
            ..Default::default()
        };

        // Without pushdown the filters are only counted.
        let mut applied = vec![];
        if config.pushdown_disabled.is_none() {
            let translator = Translator { config: &config };
            for (i, doc) in args.filters.iter().enumerate() {
                let node = Node::parse(doc);
                if let Some((expr, class)) = translator.translate(&node) {
                    let exact = class == Class::Exact;
                    if exact {
                        applied.push(i as u64);
                    }
                    // those the translated predicate compares: a conjunct dropped from it
                    // isn't compared in the database
                    let mut columns = vec![];
                    translator.pushed_columns(&node, &mut columns);
                    pushdown.filters.push(PushedFilter {
                        expr,
                        exact,
                        columns,
                    });
                }
            }
        }
        // The host never sends a limit together with filters; a limit only holds without them.
        if args.filters.is_empty() {
            pushdown.limit = args.limit;
        }

        let target = args.target_n_sources.map(|n| n as usize);
        let compiled = compile_scan(&config, pushdown, target)?;
        let n = compiled.partitions.len();
        if args.verbose {
            eprintln!(
                "[scan_database]: resolved: {n} partition(s), {}/{} filter(s) applied exactly",
                applied.len(),
                args.filters.len()
            );
        }

        let mut new_config = DatabaseScanConfig::clone(&config);
        new_config.compiled = Some(Arc::new(compiled));

        let schema = Arc::new(match &columns {
            Some(columns) => config.schema.try_project(columns)?,
            None => (*config.schema).clone(),
        });
        Ok((
            ResolvedScanOwned {
                reader: Arc::new(ReaderBuilder::new(Arc::new(new_config), schema.clone())),
                schema,
                // Source ids are absolute indices into the compiled partitions.
                sources: (0..n).map(|i| i.to_string()).collect(),
                row_estimate: config.stats.rows,
            },
            applied,
        ))
    }
}

pub struct ResolvedScanOwned {
    pub schema: SchemaRef,
    pub sources: Vec<String>,
    pub reader: Arc<ReaderBuilder>,
    pub row_estimate: Option<u64>,
}

unsafe fn resolver<'a>(h: ResolverHandle) -> &'a Resolver {
    unsafe { borrow_handle::<Resolver>(h.0) }
}

pub fn into_resolver_handle(r: Resolver) -> ResolverHandle {
    ResolverHandle(into_handle(Arc::new(r)))
}

unsafe extern "C" fn resolver_deserialize(bytes: FfiSlice<'_, u8>) -> FfiResult<ResolverHandle> {
    ffi_call(|| {
        Ok(into_resolver_handle(Resolver::deserialize(
            bytes.as_slice(),
        )?))
    })
}

unsafe extern "C" fn resolver_serialize(h: ResolverHandle) -> FfiResult<FfiBuf> {
    ffi_call(|| Ok(FfiBuf::from_vec(unsafe { resolver(h) }.serialize()?)))
}

unsafe extern "C" fn resolver_clone(h: ResolverHandle) -> ResolverHandle {
    ResolverHandle(unsafe { clone_handle::<Resolver>(h.0) })
}

/// # Safety
/// `h` must be a live resolver handle, not used afterwards.
pub unsafe extern "C" fn resolver_release(h: ResolverHandle) {
    ffi_call_infallible(|| {
        unsafe { release_handle::<Resolver>(h.0) };
        #[cfg(feature = "python")]
        crate::python::release_queued_objects();
    })
}

unsafe extern "C" fn resolver_name(_h: ResolverHandle) -> FfiBuf {
    FfiBuf::from_string("scan_database".to_string())
}

unsafe extern "C" fn resolver_fingerprint(h: ResolverHandle) -> u64 {
    ffi_call_infallible(|| unsafe { resolver(h) }.fingerprint())
}

unsafe extern "C" fn resolver_schema(h: ResolverHandle) -> FfiResult<ArrowSchema> {
    ffi_call(|| Ok(export_schema(&unsafe { resolver(h) }.config()?.schema)))
}

unsafe extern "C" fn resolver_resolve(
    h: ResolverHandle,
    args: *const ResolveArgs<'_>,
) -> FfiResult<Resolved> {
    ffi_call(|| {
        let args = Resolve::from_ffi(unsafe { &*args })
            .map_err(|e| polars_err!(ComputeError: "invalid resolve arguments: {e}"))?;
        let (scan, applied) = unsafe { resolver(h) }.resolve(&args)?;
        Ok(Resolved {
            scan: Some(ResolvedScan {
                schema: export_schema(&scan.schema),
                sources: FfiVec::from_vec(
                    scan.sources.into_iter().map(FfiBuf::from_string).collect(),
                ),
                reader_package: FfiBuf::from_string(crate::PACKAGE_NAME.to_string()),
                reader: BuilderHandle(into_handle(scan.reader)),
                applied_filters: FfiVec::from_vec(applied),
                row_estimate: scan.row_estimate.into(),
            })
            .into(),
            // The resolution is kept while the arguments don't change: partitions only need
            // to be correct, not balanced (the outer ones are open-ended).
            version_key: None.into(),
        })
    })
}

pub static PLUGIN_V1: Plugin = Plugin {
    resolver_deserialize,
    resolver_serialize,
    resolver_clone,
    resolver_release,
    resolver_name,
    resolver_fingerprint,
    resolver_schema,
    resolver_resolve,
};
