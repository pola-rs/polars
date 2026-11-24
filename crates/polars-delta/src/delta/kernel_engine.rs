//! Delta Kernel metadata I/O over the host's request-scoped storage.
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};

use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::{self};
use delta_kernel::schema::{DataType, SchemaRef, StructField, StructType};
use delta_kernel::{
    Engine, EvaluationHandler, FileMeta, JsonHandler, KernelError, KernelResult, ParquetHandler,
    StorageHandler,
};
use delta_kernel_default_engine::DefaultEngine;
use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;
use percent_encoding::percent_decode_str;
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};
use polars_utils::aliases::{PlHashMap, PlHashSet};
use tokio::sync::Mutex;
use url::Url;

use super::error::{Result, invalid};
use super::host::Storage;
use super::{log, runtime};

mod checkpoint_bounds;
mod evaluation;
mod handlers;
mod input;
mod store;

const ORIGIN: &str = "polars-delta://host";
const ROOT: &str = "polars-delta://host/";
const LOG_DIR: &str = "/_delta_log/";
pub(super) const MISSING_MODIFIED_MS: i64 = i64::MIN;

/// Call from the blocking planner thread. Kernel handles and cursors are scoped to `run`.
/// The lease still revokes access on errors, unwinding, or if an Arc was retained by Kernel.
pub(super) fn with_engine<T>(
    storage: &Storage,
    root: &str,
    schema_only: bool,
    run: impl FnOnce(&dyn Engine, Url) -> Result<T>,
) -> Result<T> {
    let io = Arc::new(HostIo {
        root: root.to_owned(),
        schema_only,
        prefix: format!(
            "{}/",
            log::absolute(root, "_delta_log/")?.trim_end_matches('/')
        ),
        live: Mutex::new(Some(Live {
            storage: storage.clone(),
            pinned: PlHashMap::default(),
            changed: None,
        })),
        aliases: Default::default(),
        sidecars: Default::default(),
    });
    let lease = Lease(&io);
    let store = Arc::new(store::HostStore(io.clone()));
    let inner = DefaultEngine::builder(store.clone())
        .with_task_executor(executor())
        .with_buffer_size(NonZeroUsize::MIN)
        .build();
    let engine = RequestEngine {
        eval: Arc::new(evaluation::HostEvaluation(inner.evaluation_handler())),
        storage: Arc::new(handlers::HostStorage(io.clone())),
        json: Arc::new(handlers::HostJson(io.clone(), inner.json_handler())),
        parquet: Arc::new(handlers::HostParquet {
            io: io.clone(),
            inner: inner.parquet_handler(),
            bounds: checkpoint_bounds::CheckpointBounds::new(store),
        }),
    };
    let result = run(&engine, Url::parse(ROOT).unwrap());
    match lease.close() {
        Some(error) => Err(error.into_ffi()),
        None => result,
    }
}

fn executor() -> Arc<TokioMultiThreadExecutor> {
    static EXECUTOR: OnceLock<Arc<TokioMultiThreadExecutor>> = OnceLock::new();
    EXECUTOR
        .get_or_init(|| {
            runtime::block_on(None, async {
                Arc::new(TokioMultiThreadExecutor::new(
                    tokio::runtime::Handle::current(),
                ))
            })
        })
        .clone()
}

struct RequestEngine {
    eval: Arc<dyn EvaluationHandler>,
    storage: Arc<dyn StorageHandler>,
    json: Arc<dyn JsonHandler>,
    parquet: Arc<dyn ParquetHandler>,
}

impl Engine for RequestEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.eval.clone()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.storage.clone()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.json.clone()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet.clone()
    }
}

struct HostIo {
    root: String,
    schema_only: bool,
    prefix: String,
    // Hold the lock throughout each callback so closing waits for all borrowed host I/O.
    live: Mutex<Option<Live>>,
    aliases: std::sync::Mutex<PlHashMap<Path, Source>>,
    sidecars: std::sync::Mutex<PlHashSet<Url>>,
}

struct Live {
    storage: Storage,
    pinned: PlHashMap<String, u64>,
    changed: Option<HostError>,
}

#[derive(Clone)]
struct Source {
    url: String,
    expected_size: Option<u64>,
    modified_ms: i64,
}

impl HostIo {
    fn relative_path<'a>(&self, location: &'a Url) -> KernelResult<&'a str> {
        let base = &location[..url::Position::BeforePath];
        if base != ORIGIN || location.query().is_some() || location.fragment().is_some() {
            return Err(KernelError::generic("expected a Delta metadata URL"));
        }
        location
            .path()
            .strip_prefix(LOG_DIR)
            .ok_or_else(|| KernelError::generic("expected a URL under _delta_log"))
    }

    fn metadata_path(&self, location: &Url) -> KernelResult<String> {
        let relative = self.relative_path(location)?;
        let raw = percent_decode_str(relative)
            .decode_utf8()
            .map_err(KernelError::generic)?;
        Ok(format!("{}{raw}", self.prefix))
    }

    /// Alias the input to the stock engine, including an absolute sidecar URL supplied by the
    /// table. Neither its Parquet footer reader nor its body reader can enter the signed-URL path.
    fn alias_parquet(&self, file: &FileMeta) -> KernelResult<FileMeta> {
        let url = if file.location.scheme() == "polars-delta" {
            if &file.location[..url::Position::BeforePath] == ORIGIN
                && !file.location.path().starts_with(LOG_DIR)
                && file.location.query().is_none()
                && file.location.fragment().is_none()
            {
                log::absolute(&self.root, file.location.path()).map_err(HostError::kernel)?
            } else {
                self.metadata_path(&file.location)?
            }
        } else {
            // A URI-based sidecar may name an absolute object outside _delta_log.
            // Send signed HTTP URLs, including their query, only to the host.
            match (file.location.scheme(), file.location.query()) {
                ("http" | "https", Some(_)) => file.location.to_string(),
                _ => {
                    log::absolute(&self.root, file.location.as_str()).map_err(HostError::kernel)?
                },
            }
        };
        mask_crc(&url)?;
        let mut aliases = self.aliases.lock().unwrap();
        let name = format!("host-metadata/{}", aliases.len());
        let location = Url::parse(ROOT)?.join(&name)?;
        aliases.insert(
            Path::from(name),
            Source {
                url,
                expected_size: (file.size != 0).then_some(file.size),
                modified_ms: file.last_modified,
            },
        );
        Ok(FileMeta::new(location, file.last_modified, file.size))
    }
}

fn mask_crc(path: &str) -> KernelResult<()> {
    if path.ends_with(".crc") {
        return Err(KernelError::file_not_found(path));
    }
    Ok(())
}

// With structured stats selected and JSON synthesis disabled, Kernel omits the text from
// native checkpoints. Retain that column for rows whose native stats struct is null.
fn with_checkpoint_stats_text(schema: SchemaRef) -> KernelResult<SchemaRef> {
    let Some(DataType::Struct(add)) = schema.field("add").map(|f| f.data_type()) else {
        return Ok(schema);
    };
    if add.field("stats_parsed").is_none() || add.field("stats").is_some() {
        return Ok(schema);
    }
    let add = add
        .fields()
        .cloned()
        .chain([StructField::nullable("stats", DataType::STRING)]);
    let add = StructType::try_new(add)?;
    let fields = schema.fields().cloned().map(|mut field| {
        if field.name() == "add" {
            field.data_type = add.clone().into();
        }
        field
    });
    Ok(Arc::new(StructType::try_new(fields)?))
}

struct Lease<'a>(&'a HostIo);

impl Lease<'_> {
    fn close(&self) -> Option<HostError> {
        futures::executor::block_on(self.0.live.lock())
            .take()
            .and_then(|live| live.changed)
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Debug, Clone)]
struct HostError {
    kind: FfiErrorKind,
    message: String,
}

impl HostError {
    fn kernel(error: FfiError) -> KernelError {
        Self::from(error).into_kernel()
    }

    fn into_kernel(self) -> KernelError {
        if self.kind == FfiErrorKind::NOT_FOUND {
            return KernelError::file_not_found(self.message);
        }
        KernelError::generic_err(self)
    }

    fn into_ffi(self) -> FfiError {
        FfiError::new(self.kind, self.message)
    }
}

impl From<FfiError> for HostError {
    fn from(error: FfiError) -> Self {
        Self {
            kind: error.kind(),
            message: error.message(),
        }
    }
}

impl From<HostError> for object_store::Error {
    fn from(error: HostError) -> Self {
        if error.kind == FfiErrorKind::NOT_FOUND {
            return Self::NotFound {
                // Kernel preserves the path and discards the source when converting to
                // KernelError::FileNotFound. Keep the original host message there.
                path: error.message.clone(),
                source: Box::new(error),
            };
        }
        Self::Generic {
            store: "Polars Delta host",
            source: Box::new(error),
        }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl Error for HostError {}

pub(super) fn host_error(error: &KernelError) -> Option<FfiError> {
    let mut source: &dyn Error = match error {
        KernelError::FileNotFound(message) => {
            return Some(FfiError::new(FfiErrorKind::NOT_FOUND, message.clone()));
        },
        KernelError::Backtraced { source, .. } => return host_error(source),
        KernelError::ObjectStore(error) => error,
        KernelError::Parquet(error) => error,
        other => other,
    };
    loop {
        if let Some(error) = source.downcast_ref::<HostError>() {
            return Some(error.clone().into_ffi());
        }
        source = source.source()?;
    }
}

fn closed() -> FfiError {
    invalid("Delta metadata engine is closed")
}

fn read_only() -> KernelError {
    KernelError::generic("Delta metadata engine is read-only")
}

#[cfg(test)]
mod tests;
