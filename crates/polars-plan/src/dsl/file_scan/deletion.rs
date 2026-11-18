use std::sync::{Arc, OnceLock, RwLock};

use polars_arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_core::prelude::PlIndexMap;
use polars_error::{PolarsResult, polars_err};
use polars_utils::pl_path::PlRefPath;

#[cfg(feature = "python")]
pub use super::python_delta_dv_provider::{
    DELTA_DV_PROVIDER_VTABLE, DeltaDeletionVectorProvider, DeltaDeletionVectorProviderVTable,
};

#[derive(Debug, Clone, Eq, PartialEq, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DeletionFilesList {
    // Chose to use a hashmap keyed by the scan source index.
    // * There may be data files without deletion files.
    // * A single data file may have multiple associated deletion files.
    //
    // Note that this uses `PlIndexMap` instead of `PlHashMap` for schemars compatibility.
    //
    // Other possible options:
    // * ListArray(inner: Utf8Array)
    //
    /// Iceberg deletes
    Iceberg(Arc<PlIndexMap<usize, IcebergDeletes>>),
    /// Native, lazily loaded Delta deletion vectors, keyed by scan source index.
    Delta(Arc<PlIndexMap<usize, DeltaDeletionVector>>),
    #[cfg(feature = "python")]
    DeltaPythonCallback(DeltaDeletionVectorProvider),
}

#[derive(Debug, Clone, Eq, PartialEq, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergDeletes {
    PositionDeletes(Buffer<PlRefPath>),
    DeletionVector {
        path: PlRefPath,
        /// Offset of the blob in the Puffin file. Without it, the blob is found by its
        /// referenced data file, which must then be unique in the Puffin file.
        content_offset: Option<u64>,
    },
}

impl IcebergDeletes {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        match self {
            IcebergDeletes::PositionDeletes(x) => x.len(),
            IcebergDeletes::DeletionVector { .. } => 1,
        }
    }
}

impl DeletionFilesList {
    /// Converts `Some(v)` to `None` if `v` is empty.
    pub fn filter_empty(this: Option<Self>) -> Option<Self> {
        use DeletionFilesList::*;

        match this {
            Some(Iceberg(paths)) => (!paths.is_empty()).then_some(Iceberg(paths)),
            #[cfg(feature = "python")]
            Some(DeltaPythonCallback(provider)) => Some(DeltaPythonCallback(provider)),
            Some(Delta(vectors)) => (!vectors.is_empty()).then_some(Delta(vectors)),
            None => None,
        }
    }

    /// Returns the number of files with deletions, but only if known at plan time.
    pub fn num_files_with_deletions(&self) -> Option<usize> {
        use DeletionFilesList::*;

        match self {
            Iceberg(paths) => Some(paths.len()),
            #[cfg(feature = "python")]
            DeltaPythonCallback(_) => None,
            Delta(vectors) => Some(vectors.len()),
        }
    }
}

impl std::hash::Hash for DeletionFilesList {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use DeletionFilesList::*;

        std::mem::discriminant(self).hash(state);

        match self {
            Iceberg(paths) => {
                for i in 0..8 {
                    usize::hash(&paths.get_index(i).map_or(0, |x| *x.0), state)
                }
            },
            #[cfg(feature = "python")]
            DeltaPythonCallback(provider) => provider.hash(state),
            Delta(vectors) => {
                for (k, v) in vectors.iter().take(8) {
                    k.hash(state);
                    v.hash(state);
                }
            },
        }
    }
}

impl std::fmt::Display for DeletionFilesList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use DeletionFilesList::*;

        match self {
            Delta(vectors) => write!(f, "delta-deletion-vector: {} sources", vectors.len())?,
            Iceberg(paths) => {
                let s = if paths.len() == 1 { "" } else { "s" };
                write!(f, "iceberg-position-delete: {} source{s}", paths.len())?;
            },
            #[cfg(feature = "python")]
            DeltaPythonCallback(_) => {
                write!(f, "delta-deletion-vector-python-callback")?;
            },
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct DeltaDeletionVector {
    pub location: DeltaDeletionVectorLocation,
    pub size_in_bytes: u32,
    pub cardinality: u64,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DeltaDeletionVectorLocation {
    Inline(Buffer<u8>),
    File { path: PlRefPath, offset: u64 },
}

/// Per-vector resource limit on the serialized size of a Delta deletion vector (host policy).
pub const DELTA_DV_MAX_SERIALIZED_BYTES: usize = 512 * 1024 * 1024;

/// Decodes the Delta deletion-vector format into keep masks.
///
/// Polars does not implement the format itself: the `polars-delta` plugin provides this over the
/// `polars.io_plugin.delta.v2` contract, and the Python layer registers it with
/// [`set_delta_deletion_vector_decoder`]. The host does all I/O; only decoding crosses the plugin
/// boundary. Masks are Arrow keep masks (set bit = keep); rows after the last deleted row are
/// kept and not part of the mask.
pub trait DeltaDeletionVectorDecoder: Send + Sync {
    /// Decode an inline vector (the portable bitmap, without frame).
    fn decode_inline(
        &self,
        data: &[u8],
        size: u32,
        cardinality: u64,
        max_rows: usize,
    ) -> PolarsResult<Bitmap>;

    /// Decode a vector stored in a file at `offset`, including its size prefix and CRC.
    fn decode_frame(
        &self,
        frame: &[u8],
        offset: u64,
        expected_size: u32,
        cardinality: u64,
        max_rows: usize,
    ) -> PolarsResult<Bitmap>;

    /// Validate the first bytes of a deletion vector file.
    fn check_file_version(&self, header: &[u8]) -> PolarsResult<()>;
}

static DELTA_DV_DECODER: RwLock<Option<Arc<dyn DeltaDeletionVectorDecoder>>> = RwLock::new(None);

/// Loads and registers the decoder on first need (see [`set_delta_deletion_vector_decoder_loader`]).
/// It returns the reason when it cannot (not installed, incompatible, failed to load).
type DecoderLoader = Box<dyn Fn() -> Result<(), String> + Send + Sync>;
static DELTA_DV_LOADER: OnceLock<DecoderLoader> = OnceLock::new();
/// When the loader last failed, and why. Retried after [`DELTA_DV_LOAD_RETRY_AFTER`], so a
/// package installed or fixed while the process runs is found, without retrying per vector.
static DELTA_DV_LOAD_FAILURE: std::sync::Mutex<Option<(std::time::Instant, String)>> =
    std::sync::Mutex::new(None);
const DELTA_DV_LOAD_RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(5);

/// Register the decoder; replaces a previous one.
pub fn set_delta_deletion_vector_decoder(decoder: Arc<dyn DeltaDeletionVectorDecoder>) {
    *DELTA_DV_DECODER.write().unwrap() = Some(decoder);
}

/// Register a callback that finds and registers the decoder (returning the reason if it did not),
/// for processes that run a plan which was resolved in another one. After a failure the callback
/// is not run again for a few seconds. It may block (it takes the Python GIL), so no lock is held while it
/// runs: concurrent callers each try it, and none waits on another.
pub fn set_delta_deletion_vector_decoder_loader(loader: DecoderLoader) {
    let _ = DELTA_DV_LOADER.set(loader);
}

pub fn delta_deletion_vector_decoder() -> Option<Arc<dyn DeltaDeletionVectorDecoder>> {
    DELTA_DV_DECODER.read().unwrap().clone()
}

/// The registered decoder, loading it first if a loader is set and none is registered, or an
/// error naming the package that provides it. May block while loading: call it off the async
/// executor's hot path (e.g. `spawn_blocking`) when [`delta_deletion_vector_decoder`] is `None`.
pub fn require_delta_deletion_vector_decoder() -> PolarsResult<Arc<dyn DeltaDeletionVectorDecoder>>
{
    if let Some(d) = delta_deletion_vector_decoder() {
        return Ok(d);
    }
    let mut reason = None;
    if let Some(loader) = DELTA_DV_LOADER.get() {
        let recent = {
            let failure = DELTA_DV_LOAD_FAILURE.lock().unwrap();
            match &*failure {
                Some((at, why)) if at.elapsed() < DELTA_DV_LOAD_RETRY_AFTER => Some(why.clone()),
                _ => None,
            }
        };
        reason = match recent {
            Some(why) => Some(why),
            None => match loader() {
                Ok(()) => None,
                Err(why) => {
                    *DELTA_DV_LOAD_FAILURE.lock().unwrap() =
                        Some((std::time::Instant::now(), why.clone()));
                    Some(why)
                },
            },
        };
    }
    delta_deletion_vector_decoder().ok_or_else(|| {
        let why = reason
            .as_deref()
            .unwrap_or("the `polars-delta` package is not installed or could not be loaded");
        polars_err!(
            ComputeError:
            "reading Delta deletion vectors natively requires the `polars-delta` package: {why} \
            (`pip install -U polars-delta`)"
        )
    })
}
