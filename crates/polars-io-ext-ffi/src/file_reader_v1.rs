//! Contract `polars.io_plugin.file_reader.v1`: a multi-scan file reader implemented by a plugin.
//!
//! **Frozen once released**: any change requires a new module and ID (see the crate docs).
//!
//! This is the C ABI form of Polars' multi-scan `FileReaderBuilder` / `FileReader` (the Python
//! form is `polars.io.external_reader`). A scan has a list of sources (strings that the plugin
//! interprets, e.g. paths or partition ids) and one builder. For each execution of the scan the
//! host opens a [`ScanHandle`] (state shared by its readers, e.g. a connection pool) and builds
//! one reader per source.
//!
//! Handles:
//! * [`BuilderHandle`]: reference counted ([`Plugin::builder_clone`] /
//!   [`Plugin::builder_release`]), immutable, thread-safe. It is part of the query plan: it can
//!   be serialized and deserialized, also on another machine.
//! * [`ScanHandle`]: one execution of the scan. Thread-safe. Releasing it stops readers that wait
//!   for a resource of the scan.
//! * [`ReaderHandle`]: one source. Used by one host thread at a time, except for
//!   [`Plugin::reader_cancel`], which may be called from any thread at any time.
//!
//! Every function may block (I/O); the host calls them from blocking threads without holding the
//! GIL. Batches are read from the stream returned by [`Plugin::reader_begin_read`], also from a
//! blocking thread. After a `get_next` failure, [`Plugin::reader_take_error`] returns the error
//! with its kind.
use std::ffi::{CStr, c_void};

use crate::common::{
    ArrowArrayStream, ArrowSchema, FfiBuf, FfiOption, FfiResult, FfiSlice, FfiStr,
};

pub const ID: &CStr = c"polars.io_plugin.file_reader.v1";

macro_rules! handle {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug)]
        pub struct $name(pub *const c_void);
        // Handles are thread-safe (see the module docs).
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
    };
}

handle!(BuilderHandle);
handle!(ScanHandle);
handle!(ReaderHandle);

/// [`Capabilities::flags`].
pub mod flags {
    /// Sources may be read in any order when the query does not need the row order.
    pub const UNORDERED: u32 = 1 << 0;
    /// Sources have remote latency (e.g. ids of remote objects), whatever they look like.
    pub const SOURCES_REMOTE: u32 = 1 << 1;
}

#[repr(C)]
pub struct Capabilities {
    pub flags: u32,
    /// Upper bound on readers running at once (e.g. a connection budget). The host never runs
    /// more, in ordered mode too.
    pub max_concurrent_scans: FfiOption<u64>,
    /// Number of readers to initialize ahead when the sources are remote.
    pub preferred_remote_pre_init: FfiOption<u64>,
}

/// Arguments of [`Plugin::reader_begin_read`], borrowed for the call.
#[repr(C)]
pub struct BeginReadArgs<'a> {
    /// Columns to read, in order; never empty (the host gets row counts from
    /// [`Plugin::reader_n_rows`]).
    pub columns: FfiSlice<'a, FfiStr<'a>>,
    pub verbose: bool,
}

/// The contract's entry point, pointed to by the capsule named [`ID`].
#[repr(C)]
pub struct Plugin {
    pub builder_deserialize:
        unsafe extern "C" fn(bytes: FfiSlice<'_, u8>) -> FfiResult<BuilderHandle>,
    pub builder_serialize: unsafe extern "C" fn(builder: BuilderHandle) -> FfiResult<FfiBuf>,
    /// Returns a new reference to the same builder.
    pub builder_clone: unsafe extern "C" fn(builder: BuilderHandle) -> BuilderHandle,
    pub builder_release: unsafe extern "C" fn(builder: BuilderHandle),
    /// Name for explain and error messages, e.g. `database[postgres/adbc:postgresql]`.
    pub builder_name: unsafe extern "C" fn(builder: BuilderHandle) -> FfiBuf,
    /// Stable hash of the builder (without secrets). Builders with the same package and
    /// fingerprint read the same data: the host may deduplicate their scans.
    pub builder_fingerprint: unsafe extern "C" fn(builder: BuilderHandle) -> u64,
    /// Properties shown by explain, as alternating keys and values.
    pub builder_explain:
        unsafe extern "C" fn(builder: BuilderHandle) -> FfiResult<crate::common::FfiVec<FfiBuf>>,
    pub builder_capabilities:
        unsafe extern "C" fn(builder: BuilderHandle) -> FfiResult<Capabilities>,
    pub scan_new: unsafe extern "C" fn(builder: BuilderHandle) -> FfiResult<ScanHandle>,
    pub scan_release: unsafe extern "C" fn(scan: ScanHandle),
    pub reader_new:
        unsafe extern "C" fn(scan: ScanHandle, source: FfiStr<'_>) -> FfiResult<ReaderHandle>,
    pub reader_release: unsafe extern "C" fn(reader: ReaderHandle),
    /// The columns of the source, as an Arrow struct schema.
    pub reader_schema: unsafe extern "C" fn(reader: ReaderHandle) -> FfiResult<ArrowSchema>,
    pub reader_n_rows: unsafe extern "C" fn(reader: ReaderHandle) -> FfiResult<u64>,
    /// Struct arrays with exactly `args.columns`, typed as in [`Plugin::reader_schema`]. Releasing
    /// the stream before its end stops the read.
    pub reader_begin_read: unsafe extern "C" fn(
        reader: ReaderHandle,
        args: *const BeginReadArgs<'_>,
    ) -> FfiResult<ArrowArrayStream>,
    /// Stops a running read: a blocked `get_next` returns an error soon after. May be called
    /// from any thread, also when no read runs.
    pub reader_cancel: unsafe extern "C" fn(reader: ReaderHandle),
    /// The error of the last failed `get_next` of the reader's stream, if any.
    pub reader_take_error: unsafe extern "C" fn(reader: ReaderHandle) -> FfiResult<()>,
}

/// Owned form of [`BeginReadArgs`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeginRead {
    pub columns: Vec<String>,
    pub verbose: bool,
}

impl BeginRead {
    /// Call `f` with the borrowed FFI form.
    pub fn with_ffi<R>(&self, f: impl FnOnce(&BeginReadArgs<'_>) -> R) -> R {
        let columns: Vec<FfiStr<'_>> = self.columns.iter().map(|c| FfiStr::new(c)).collect();
        f(&BeginReadArgs {
            columns: FfiSlice::new(&columns),
            verbose: self.verbose,
        })
    }

    /// Copy from the FFI form.
    pub fn from_ffi(args: &BeginReadArgs<'_>) -> Result<Self, std::str::Utf8Error> {
        Ok(Self {
            columns: args
                .columns
                .as_slice()
                .iter()
                .map(|c| c.as_str().map(str::to_owned))
                .collect::<Result<_, _>>()?,
            verbose: args.verbose,
        })
    }
}
