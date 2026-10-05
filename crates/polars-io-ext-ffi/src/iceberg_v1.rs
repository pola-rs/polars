//! Contract `polars.io_plugin.iceberg.v1`: Iceberg scan planning.
//!
//! **Frozen once released**: any change requires a new module and ID (see the crate docs).
//!
//! The host (Polars) calls [`Plugin::plan`] from a blocking thread without holding the GIL. The
//! plugin plans on its own threads and reads table metadata and manifests through [`Host`],
//! whose futures run on the host's IO runtime and may be awaited from any executor. `Host` and
//! everything borrowed from it are valid only until `plan` returns.
//!
//! The request is an [`FfiRequest`], borrowed for the call. The output is an [`Output`]:
//! * `header`: [`FfiOutputHeader`].
//! * `schema`: the table schema of the selected snapshot as an Arrow struct schema, with the
//!   Iceberg field ID of every field in the `PARQUET:field_id` metadata key.
//! * `files`: stream of struct arrays, one row per data file to scan, in scan order, with the
//!   columns below (and no others).
//! * `table_values`: stream of at most one struct array of length 1, with the columns below.
//!
//! `files` columns:
//! * `path: utf8view` (absolute URL), `size: u64` (bytes), `record_count: u64`
//! * `deletes: large_list<struct{kind: utf8view, path: utf8view}>`, where `kind` is `position`
//!   (Parquet position delete file) or `deletion_vector` (Puffin file); a data file has either
//!   position delete files or a single deletion vector.
//! * `constants: struct{<field id>: <value>}` (only if non-empty): identity-partition values of
//!   projected fields, per data file; null if the file has no value.
//! * `stats: struct{<col>_nc: u64, <col>_min, <col>_max}` (only if statistics were requested
//!   and there are filter columns): per-file null counts and bounds of the filter columns, null
//!   if unknown. Struct columns have per-leaf struct statistics.
//!
//! `table_values` columns:
//! * `initial_defaults: struct{<field id>: <value>}` (only if non-empty): `initial-default`s of
//!   projected fields, including nested ones.
use std::ffi::{CStr, c_void};

use crate::common::{
    ArrowArrayStream, ArrowSchema, FfiBuf, FfiFuture, FfiOption, FfiResult, FfiSlice, FfiStr,
    FfiVec,
};

pub const ID: &CStr = c"polars.io_plugin.iceberg.v1";

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
    pub table_values: ArrowArrayStream,
}

/// The contract's entry point, pointed to by the capsule named [`ID`].
#[repr(C)]
pub struct Plugin {
    /// Errors with kind `INVALID_INPUT` are invalid user parameters (e.g. an unknown snapshot
    /// ID) and are reported with their message unchanged.
    pub plan:
        unsafe extern "C" fn(host: *const Host, request: *const FfiRequest) -> FfiResult<Output>,
}

/// The request, borrowed for the duration of [`Plugin::plan`]. See [`Request`] for the fields.
#[repr(C)]
pub struct FfiRequest<'a> {
    pub metadata_location: FfiStr<'a>,
    pub snapshot_id: FfiOption<i64>,
    pub from_snapshot_id_exclusive: FfiOption<i64>,
    pub to_snapshot_id_inclusive: FfiOption<i64>,
    pub projection: FfiOption<FfiSlice<'a, FfiStr<'a>>>,
    pub filter_columns: FfiOption<FfiSlice<'a, FfiStr<'a>>>,
    pub row_filter: FfiOption<FfiStr<'a>>,
    pub limit: FfiOption<u64>,
    pub max_threads: FfiOption<u64>,
    pub use_metadata_statistics: bool,
    pub fast_deletion_count: bool,
    pub verbose: bool,
    pub testing_fail: FfiOption<FfiStr<'a>>,
}

/// Owned form of [`FfiRequest`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Request {
    /// Absolute location of the table's current `metadata.json`.
    pub metadata_location: String,
    /// Snapshot to scan; the current snapshot if `None` and no incremental bound is set.
    pub snapshot_id: Option<i64>,
    /// Incremental scan of the appends after this snapshot.
    pub from_snapshot_id_exclusive: Option<i64>,
    /// Incremental scan up to this snapshot; the current snapshot if `None`.
    pub to_snapshot_id_inclusive: Option<i64>,
    /// Names of the projected top-level columns; all if `None`.
    pub projection: Option<Vec<String>>,
    /// Names of the columns referenced by the predicate. Statistics are returned if this is
    /// set (even if empty) and `use_metadata_statistics` is set.
    pub filter_columns: Option<Vec<String>>,
    /// Iceberg expression in the REST catalog JSON format, used for pruning only: the host
    /// applies the full predicate to the scanned rows.
    pub row_filter: Option<String>,
    /// Maximum number of rows needed; the plugin may plan fewer files.
    pub limit: Option<u64>,
    /// Upper bound for the number of plugin threads.
    pub max_threads: Option<u64>,
    /// Use file statistics and row counts from metadata.
    pub use_metadata_statistics: bool,
    /// Report the row count even if rows are deleted.
    pub fast_deletion_count: bool,
    pub verbose: bool,
    /// Testing only: `"io"` (fail with an IO error) or `"panic"`.
    pub testing_fail: Option<String>,
}

impl Request {
    /// Call `f` with the borrowed FFI form of this request.
    pub fn with_ffi<R>(&self, f: impl FnOnce(&FfiRequest<'_>) -> R) -> R {
        fn strs(v: &Option<Vec<String>>) -> Option<Vec<FfiStr<'_>>> {
            v.as_ref()
                .map(|v| v.iter().map(|s| FfiStr::new(s)).collect())
        }
        fn opt_str(s: &Option<String>) -> FfiOption<FfiStr<'_>> {
            s.as_deref().map(FfiStr::new).into()
        }
        let projection = strs(&self.projection);
        let filter_columns = strs(&self.filter_columns);

        f(&FfiRequest {
            metadata_location: FfiStr::new(&self.metadata_location),
            snapshot_id: self.snapshot_id.into(),
            from_snapshot_id_exclusive: self.from_snapshot_id_exclusive.into(),
            to_snapshot_id_inclusive: self.to_snapshot_id_inclusive.into(),
            projection: projection.as_deref().map(FfiSlice::new).into(),
            filter_columns: filter_columns.as_deref().map(FfiSlice::new).into(),
            row_filter: opt_str(&self.row_filter),
            limit: self.limit.into(),
            max_threads: self.max_threads.into(),
            use_metadata_statistics: self.use_metadata_statistics,
            fast_deletion_count: self.fast_deletion_count,
            verbose: self.verbose,
            testing_fail: opt_str(&self.testing_fail),
        })
    }

    /// Copy a borrowed request. Errors if a string is not UTF-8.
    pub fn from_ffi(r: &FfiRequest<'_>) -> Result<Self, std::str::Utf8Error> {
        let string = |s: FfiStr<'_>| s.as_str().map(str::to_owned);
        let opt_string = |s: FfiOption<FfiStr<'_>>| s.into_option().map(string).transpose();
        let strings = |v: FfiOption<FfiSlice<'_, FfiStr<'_>>>| {
            v.into_option()
                .map(|v| v.as_slice().iter().map(|s| string(*s)).collect())
                .transpose()
        };

        Ok(Self {
            metadata_location: string(r.metadata_location)?,
            snapshot_id: r.snapshot_id.into_option(),
            from_snapshot_id_exclusive: r.from_snapshot_id_exclusive.into_option(),
            to_snapshot_id_inclusive: r.to_snapshot_id_inclusive.into_option(),
            projection: strings(r.projection)?,
            filter_columns: strings(r.filter_columns)?,
            row_filter: opt_string(r.row_filter)?,
            limit: r.limit.into_option(),
            max_threads: r.max_threads.into_option(),
            use_metadata_statistics: r.use_metadata_statistics,
            fast_deletion_count: r.fast_deletion_count,
            verbose: r.verbose,
            testing_fail: opt_string(r.testing_fail)?,
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
pub struct ConstantError {
    pub field_id: u32,
    /// UTF-8 message.
    pub reason: FfiBuf,
}

/// Output metadata, owned by the plugin. See [`OutputHeader`] for the fields.
#[repr(C)]
pub struct FfiOutputHeader {
    pub row_count: FfiOption<RowCount>,
    pub constant_errors: FfiVec<ConstantError>,
    pub statistics: bool,
}

/// Owned form of [`FfiOutputHeader`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OutputHeader {
    /// `(physical rows, deleted rows)`, if they can be used as the row count.
    pub row_count: Option<(u64, u64)>,
    /// Field IDs whose identity-partition constants could not be loaded, with the reason. The
    /// host raises the reason if the column is read from a file that lacks it.
    pub constant_errors: Vec<(u32, String)>,
    /// Whether statistics were requested; the host then builds a statistics table (with the
    /// per-file row counts) even if `files` has no `stats` column.
    pub statistics: bool,
}

impl OutputHeader {
    pub fn into_ffi(self) -> FfiOutputHeader {
        FfiOutputHeader {
            row_count: self
                .row_count
                .map(|(physical, deleted)| RowCount { physical, deleted })
                .into(),
            constant_errors: FfiVec::from_vec(
                self.constant_errors
                    .into_iter()
                    .map(|(field_id, reason)| ConstantError {
                        field_id,
                        reason: FfiBuf::from_string(reason),
                    })
                    .collect(),
            ),
            statistics: self.statistics,
        }
    }

    /// Copy the plugin's header; invalid UTF-8 in messages is replaced.
    pub fn from_ffi(h: &FfiOutputHeader) -> Self {
        Self {
            row_count: h.row_count.into_option().map(|c| (c.physical, c.deleted)),
            constant_errors: h
                .constant_errors
                .as_slice()
                .iter()
                .map(|e| (e.field_id, e.reason.to_string_lossy()))
                .collect(),
            statistics: h.statistics,
        }
    }
}

// The frozen layout.
const _: () = {
    let ptr = size_of::<*const c_void>();
    assert!(size_of::<StorageHandle>() == ptr);
    assert!(size_of::<Host>() == 6 * ptr);
    assert!(size_of::<Plugin>() == ptr);
    assert!(size_of::<FfiBuf>() == 4 * ptr);
    assert!(size_of::<ArrowSchema>() == 9 * 8);
    assert!(size_of::<ArrowArrayStream>() == 5 * 8);
    assert!(size_of::<FfiStr>() == 2 * ptr);
    assert!(size_of::<FfiSlice<FfiStr>>() == 2 * ptr);
    assert!(size_of::<FfiVec<ConstantError>>() == 4 * ptr);
    assert!(size_of::<FfiOption<i64>>() == 16);
    assert!(size_of::<FfiOption<u64>>() == 16);
    assert!(size_of::<FfiOption<FfiStr>>() == 3 * ptr);
    assert!(size_of::<FfiRequest>() == 2 * ptr + 5 * 16 + 4 * 3 * ptr + 8);
    assert!(size_of::<RowCount>() == 16);
    assert!(size_of::<ConstantError>() == 5 * ptr);
    assert!(size_of::<FfiOutputHeader>() == 24 + 4 * ptr + 8);
    assert!(size_of::<Output>() == size_of::<FfiOutputHeader>() + 9 * 8 + 2 * 5 * 8);
};
