//! Unreleased contract `polars.io_plugin.delta.v1`: native Delta snapshot planning and
//! deletion-vector decoding.
//! Output schema is a logical Arrow struct with original Delta field metadata and mapped PARQUET:field_id.
//! Files contain path:utf8view,size:u64,record_count:nullable u64,partition_values:typed struct,
//! optional stats:struct{<logical_col>_nc/min/max}, and nullable deletion_vector:struct{
//! path:nullable utf8view,inline:nullable binaryview,offset:u64,size_in_bytes:u32,cardinality:u64}.
//! Exactly one DV path/inline is set; inline contains decoded bitmapData, file paths are absolute.
//! The full snapshot is returned: request.limit never permits dropping files before host filtering.
//! Host and storage handles live only during plan; output streams own all returned data.
use std::ffi::{CStr, c_void};

use crate::common::{
    ArrowArrayStream, ArrowSchema, FfiBuf, FfiFuture, FfiOption, FfiResult, FfiSlice, FfiStr,
    FfiVec,
};

pub const ID: &CStr = c"polars.io_plugin.delta.v1";

/// Object listing entry. URLs are absolute; missing modification times cannot support legacy timestamp travel.
#[repr(C)]
pub struct ObjectMeta {
    pub url: FfiBuf,
    pub size: u64,
    pub modified_ms: FfiOption<i64>,
}

/// Opaque host storage object, released with [`Host::storage_release`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct StorageHandle(pub *const c_void);

// Host storage objects are thread-safe.
unsafe impl Send for StorageHandle {}
unsafe impl Sync for StorageHandle {}

/// Functions the host provides for the duration of one [`Plugin::plan`] call. URL arguments are
/// borrowed only for the call; the host copies them before returning the future.
#[repr(C)]
pub struct Host {
    pub ctx: *const c_void,
    /// Storage for URLs with the scheme of `url`, using the cloud options and credential
    /// provider of the scan.
    pub get_storage: unsafe extern "C" fn(
        ctx: *const c_void,
        url: FfiStr,
    ) -> FfiFuture<FfiResult<StorageHandle>>,
    /// Whole object.
    pub storage_get:
        unsafe extern "C" fn(storage: StorageHandle, url: FfiStr) -> FfiFuture<FfiResult<FfiBuf>>,
    /// Size of the object in bytes. Errors with kind `NOT_FOUND` if it does not exist.
    pub storage_head:
        unsafe extern "C" fn(storage: StorageHandle, url: FfiStr) -> FfiFuture<FfiResult<u64>>,
    /// Complete, nonrecursive listing of direct children of prefix, sorted by URL.
    /// start_after is a strict lexical exclusive bound. No mutable listing cache.
    pub storage_list: unsafe extern "C" fn(
        storage: StorageHandle,
        prefix: FfiStr,
        start_after: FfiOption<FfiStr>,
    ) -> FfiFuture<FfiResult<FfiVec<ObjectMeta>>>,
    /// Exact half-open byte range; short reads are errors.
    pub storage_get_range: unsafe extern "C" fn(
        storage: StorageHandle,
        url: FfiStr,
        start: u64,
        end: u64,
    ) -> FfiFuture<FfiResult<FfiBuf>>,
    /// Release a storage object. Futures returned for it must have completed or been dropped.
    pub storage_release: unsafe extern "C" fn(storage: StorageHandle),
    /// Print a verbose log message. Only called when [`FfiRequest::verbose`] is set.
    pub log: unsafe extern "C" fn(ctx: *const c_void, msg: FfiStr),
}

unsafe impl Send for Host {}
unsafe impl Sync for Host {}

#[repr(C)]
pub struct Output {
    pub header: FfiOutputHeader,
    pub schema: ArrowSchema,
    pub files: ArrowArrayStream,
}

/// A decoded deletion vector as an Arrow keep mask: packed bits, low bit first, set = keep, no
/// validity. Rows at or after `len` are kept. `bytes` is owned by the plugin's allocator and
/// released by dropping it, so the host can wrap it without copying.
#[repr(C)]
pub struct KeepMask {
    pub bytes: FfiBuf,
    pub len: u64,
}

/// The contract's entry point, pointed to by the capsule named [`ID`].
///
/// The deletion-vector functions are pure decoders: the host reads the bytes (it owns storage and
/// credentials), and they have no state, so they may be called from any thread at any time after
/// the plugin is loaded, not only during `plan`. Corrupt vectors and ones over a resource limit
/// fail with kind `OTHER`.
#[repr(C)]
pub struct Plugin {
    /// Errors with kind `INVALID_INPUT` are invalid user parameters (e.g. an unknown snapshot
    /// ID) and are reported with their message unchanged.
    pub plan:
        unsafe extern "C" fn(host: *const Host, request: *const FfiRequest) -> FfiResult<Output>,
    /// Decode an inline vector (portable Roaring bitmap bytes, as in `Files.deletion_vector.inline`).
    /// `data.len()` must equal `size`, and the bitmap must hold `cardinality` positions, the
    /// largest below `max_rows`.
    pub decode_inline: unsafe extern "C" fn(
        data: FfiSlice<'_, u8>,
        size: u32,
        cardinality: u64,
        max_rows: u64,
    ) -> FfiResult<KeepMask>,
    /// Decode a stored vector frame (`[size: u32 BE][bitmap][crc32 BE]`, `size + 8` bytes) read
    /// from a deletion vector file at `offset` (`deletion_vector.offset`). Validates the offset
    /// (a frame follows the file's version byte), the size prefix against `expected_size`, the
    /// CRC, the bitmap, `cardinality` and `max_rows`.
    pub decode_frame: unsafe extern "C" fn(
        frame: FfiSlice<'_, u8>,
        offset: u64,
        expected_size: u32,
        cardinality: u64,
        max_rows: u64,
    ) -> FfiResult<KeepMask>,
    /// Validate the first byte of a deletion vector file (the format version).
    pub check_file_version: unsafe extern "C" fn(header: FfiSlice<'_, u8>) -> FfiResult<()>,
}

#[repr(C)]
pub struct FfiRequest<'a> {
    pub table_uri: FfiStr<'a>,
    pub version: FfiOption<i64>,
    pub timestamp_ms: FfiOption<i64>,
    pub projection: FfiOption<FfiSlice<'a, FfiStr<'a>>>,
    pub filter_columns: FfiOption<FfiSlice<'a, FfiStr<'a>>>,
    pub statistics_columns: FfiOption<FfiSlice<'a, FfiStr<'a>>>,
    pub limit: FfiOption<u64>,
    pub max_threads: FfiOption<u64>,
    pub use_metadata_statistics: bool,
    pub fast_deletion_count: bool,
    pub verbose: bool,
    pub testing_fail: FfiOption<FfiStr<'a>>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Request {
    pub table_uri: String,
    pub version: Option<i64>,
    pub timestamp_ms: Option<i64>,
    pub projection: Option<Vec<String>>,
    pub filter_columns: Option<Vec<String>>,
    pub statistics_columns: Option<Vec<String>>,
    pub limit: Option<u64>,
    pub max_threads: Option<u64>,
    pub use_metadata_statistics: bool,
    pub fast_deletion_count: bool,
    pub verbose: bool,
    pub testing_fail: Option<String>,
}
impl Request {
    pub fn with_ffi<R>(&self, f: impl FnOnce(&FfiRequest<'_>) -> R) -> R {
        fn strs(v: &Option<Vec<String>>) -> Option<Vec<FfiStr<'_>>> {
            v.as_ref()
                .map(|v| v.iter().map(|s| FfiStr::new(s)).collect())
        }
        let projection = strs(&self.projection);
        let filter_columns = strs(&self.filter_columns);
        let statistics_columns = strs(&self.statistics_columns);
        f(&FfiRequest {
            table_uri: FfiStr::new(&self.table_uri),
            version: self.version.into(),
            timestamp_ms: self.timestamp_ms.into(),
            projection: projection.as_deref().map(FfiSlice::new).into(),
            filter_columns: filter_columns.as_deref().map(FfiSlice::new).into(),
            statistics_columns: statistics_columns.as_deref().map(FfiSlice::new).into(),
            limit: self.limit.into(),
            max_threads: self.max_threads.into(),
            use_metadata_statistics: self.use_metadata_statistics,
            fast_deletion_count: self.fast_deletion_count,
            verbose: self.verbose,
            testing_fail: self.testing_fail.as_deref().map(FfiStr::new).into(),
        })
    }
    pub fn from_ffi(r: &FfiRequest<'_>) -> Result<Self, std::str::Utf8Error> {
        let string = |s: FfiStr<'_>| s.as_str().map(str::to_owned);
        let strings = |v: FfiOption<FfiSlice<'_, FfiStr<'_>>>| {
            v.into_option()
                .map(|v| v.as_slice().iter().map(|s| string(*s)).collect())
                .transpose()
        };
        Ok(Self {
            table_uri: string(r.table_uri)?,
            version: r.version.into_option(),
            timestamp_ms: r.timestamp_ms.into_option(),
            projection: strings(r.projection)?,
            filter_columns: strings(r.filter_columns)?,
            statistics_columns: strings(r.statistics_columns)?,
            limit: r.limit.into_option(),
            max_threads: r.max_threads.into_option(),
            use_metadata_statistics: r.use_metadata_statistics,
            fast_deletion_count: r.fast_deletion_count,
            verbose: r.verbose,
            testing_fail: r.testing_fail.into_option().map(string).transpose()?,
        })
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowCount {
    pub physical: u64,
    pub deleted: u64,
}
#[repr(C)]
pub struct FfiOutputHeader {
    pub resolved_version: i64,
    pub column_mapping_mode: FfiBuf,
    pub row_count: FfiOption<RowCount>,
    pub statistics: bool,
}
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OutputHeader {
    pub resolved_version: i64,
    pub column_mapping_mode: String,
    pub row_count: Option<(u64, u64)>,
    pub statistics: bool,
}
impl OutputHeader {
    pub fn into_ffi(self) -> FfiOutputHeader {
        FfiOutputHeader {
            resolved_version: self.resolved_version,
            column_mapping_mode: FfiBuf::from_string(self.column_mapping_mode),
            row_count: self
                .row_count
                .map(|(physical, deleted)| RowCount { physical, deleted })
                .into(),
            statistics: self.statistics,
        }
    }
    pub fn from_ffi(h: &FfiOutputHeader) -> Self {
        Self {
            resolved_version: h.resolved_version,
            column_mapping_mode: h.column_mapping_mode.to_string_lossy(),
            row_count: h.row_count.into_option().map(|c| (c.physical, c.deleted)),
            statistics: h.statistics,
        }
    }
}

// The layout rule: a plugin's `Plugin` static is read through this layout, so any added or
// reordered function pointer needs a new ID. The pointer count is pinned here.
const _: () = {
    let ptr = size_of::<*const c_void>();
    assert!(size_of::<Plugin>() == 4 * ptr);
    assert!(size_of::<Host>() == 8 * ptr);
};
