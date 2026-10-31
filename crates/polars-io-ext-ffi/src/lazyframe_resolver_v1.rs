//! Contract `polars.io_plugin.lazyframe_resolver.v1`: a LazyFrame resolver implemented by a
//! plugin.
//!
//! **Frozen once released**: any change requires a new module and ID (see the crate docs).
//!
//! This is the C ABI form of Polars' `LazyFrameResolver` (the DSL resolver node). The host
//! calls [`Plugin::resolver_resolve`] while optimizing a query, with the projection, the
//! filters and the limit pushed down to the resolver. The plugin answers with a scan: a list of
//! sources read by a [`file_reader_v1`](crate::file_reader_v1) builder, the schema of its rows,
//! and which filters the scan applies itself. The host builds the scan node, then applies the
//! slice, the filters not applied by the scan and the projection on top.
//!
//! [`ResolverHandle`]s are reference counted, immutable (caches aside) and thread-safe. They are
//! part of the query plan: they can be serialized and deserialized, also on another machine.
//! Every function may block (I/O); the host calls them from blocking threads without holding
//! the GIL.
//!
//! # Filter documents
//!
//! Each filter is a conjunct of the predicate pushed to the resolver, as a JSON document (format
//! `filter_expr.v1`). A node is one of:
//! * `{"col": name}`
//! * `{"lit": value}` (see below)
//! * `{"op": op, "args": [node, …], …}` with `op` one of
//!   * `and`, `or` (2 args)
//!   * `eq`, `neq`, `lt`, `lt_eq`, `gt`, `gt_eq`, `eq_missing`, `neq_missing` (2 args)
//!   * `not`, `is_null`, `is_not_null` (1 arg)
//!   * `is_between` (3 args: expression, lower, upper) with `"closed"`: `both`, `left`, `right`
//!     or `none`
//!   * `is_in` (1 arg) with `"values": {"dtype": dtype, "values": [value, …]}` and
//!     `"nulls_equal": bool`
//!   * `starts_with`, `ends_with`, `contains_literal` (2 args: string, pattern)
//!   * `cast` (1 arg) with `"dtype": dtype` (a strict or non-strict cast)
//! * `{"unknown": description}`: any other expression. It evaluates to an unknown value of the
//!   expression's type; a plugin can still use the rest of the filter (e.g. one side of an
//!   `and`).
//!
//! A literal value is `{"dtype": dtype, "value": v}`, the `values` of `is_in` are bare `v`s of
//! the given `dtype`. `dtype` and `v`:
//! * `null` (`v` is null), `bool` (`true`/`false`), `str` (a string)
//! * `i8` … `i128`, `u8` … `u128`: decimal string
//! * `f32`, `f64`: string that parses back to the same float (`NaN`, `inf`, `-inf`)
//! * `date`: days since the epoch (number)
//! * `time`: nanoseconds since midnight (decimal string)
//! * `datetime[<unit>]`, `datetime[<unit>, <time zone>]`, `duration[<unit>]` with unit `ns`,
//!   `us` or `ms`: count since the epoch (decimal string)
//! * `decimal[<precision>, <scale>]`: unscaled value (decimal string)
//!
//! Any other value of `v` is null. A literal of another dtype is `{"unknown": …}`.
//!
//! The document is frozen with this contract: new node kinds or dtypes need a new contract ID.
use std::ffi::{CStr, c_void};

use crate::common::{ArrowSchema, FfiBuf, FfiOption, FfiResult, FfiSlice, FfiStr, FfiVec};
use crate::file_reader_v1::BuilderHandle;

pub const ID: &CStr = c"polars.io_plugin.lazyframe_resolver.v1";

/// Name of the PyCapsule in which a plugin hands a resolver to a Python host. Its pointer points
/// to a [`ResolverHandle`]; the capsule owns one reference, the host takes its own with
/// [`Plugin::resolver_clone`].
pub const HANDLE_CAPSULE_NAME: &CStr = c"polars.io_plugin.lazyframe_resolver.v1.handle";

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct ResolverHandle(pub *const c_void);
// Resolver handles are thread-safe (see the module docs).
unsafe impl Send for ResolverHandle {}
unsafe impl Sync for ResolverHandle {}

/// Arguments of [`Plugin::resolver_resolve`], borrowed for the call.
#[repr(C)]
pub struct ResolveArgs<'a> {
    /// Columns to produce; all columns of the schema if `None`.
    pub projection: FfiOption<FfiSlice<'a, FfiStr<'a>>>,
    /// Rows needed from the start; the host applies the slice itself as well. Never set
    /// together with filters.
    pub limit: FfiOption<u64>,
    /// Filter documents; the host keeps the rows for which all of them are true.
    pub filters: FfiSlice<'a, FfiStr<'a>>,
    /// Columns that the filters reference.
    pub filter_columns: FfiSlice<'a, FfiStr<'a>>,
    /// Columns at and after this index of `projection` are needed only by the filters.
    pub filter_drop_columns_idx: FfiOption<u64>,
    /// `version_key` of the previous resolution with the same arguments, if any.
    pub existing_version_key: FfiOption<FfiStr<'a>>,
    /// Number of sources the host would like (e.g. one per worker); a hint.
    pub target_n_sources: FfiOption<u64>,
    pub verbose: bool,
}

/// A scan to read.
#[repr(C)]
pub struct ResolvedScan {
    /// Columns of the scan's rows: the projected ones (all if there is no projection), as an
    /// Arrow struct schema.
    pub schema: ArrowSchema,
    pub sources: FfiVec<FfiBuf>,
    /// Python package of the plugin providing [`crate::file_reader_v1`] for `reader`.
    pub reader_package: FfiBuf,
    pub reader: BuilderHandle,
    /// Indices into `ResolveArgs::filters` of the filters that the scan applies exactly.
    pub applied_filters: FfiVec<u64>,
    /// Estimated number of rows of the scan.
    pub row_estimate: FfiOption<u64>,
}

#[repr(C)]
pub struct Resolved {
    /// `None` if the previous resolution (`existing_version_key`) is still valid.
    pub scan: FfiOption<ResolvedScan>,
    /// If set, the host calls the resolver again on every execution; otherwise it may reuse the
    /// resolution while the arguments do not change.
    pub version_key: FfiOption<FfiBuf>,
}

/// The contract's entry point, pointed to by the capsule named [`ID`].
#[repr(C)]
pub struct Plugin {
    pub resolver_deserialize:
        unsafe extern "C" fn(bytes: FfiSlice<'_, u8>) -> FfiResult<ResolverHandle>,
    pub resolver_serialize: unsafe extern "C" fn(resolver: ResolverHandle) -> FfiResult<FfiBuf>,
    /// Returns a new reference to the same resolver.
    pub resolver_clone: unsafe extern "C" fn(resolver: ResolverHandle) -> ResolverHandle,
    pub resolver_release: unsafe extern "C" fn(resolver: ResolverHandle),
    /// Name shown by explain.
    pub resolver_name: unsafe extern "C" fn(resolver: ResolverHandle) -> FfiBuf,
    /// Stable hash of the resolver (without secrets). Resolvers with the same package and
    /// fingerprint resolve to the same data: the host may deduplicate them.
    pub resolver_fingerprint: unsafe extern "C" fn(resolver: ResolverHandle) -> u64,
    /// All columns, as an Arrow struct schema.
    pub resolver_schema: unsafe extern "C" fn(resolver: ResolverHandle) -> FfiResult<ArrowSchema>,
    pub resolver_resolve: unsafe extern "C" fn(
        resolver: ResolverHandle,
        args: *const ResolveArgs<'_>,
    ) -> FfiResult<Resolved>,
}

/// Owned form of [`ResolveArgs`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Resolve {
    pub projection: Option<Vec<String>>,
    pub limit: Option<u64>,
    pub filters: Vec<String>,
    pub filter_columns: Vec<String>,
    pub filter_drop_columns_idx: Option<u64>,
    pub existing_version_key: Option<String>,
    pub target_n_sources: Option<u64>,
    pub verbose: bool,
}

impl Resolve {
    /// Call `f` with the borrowed FFI form.
    pub fn with_ffi<R>(&self, f: impl FnOnce(&ResolveArgs<'_>) -> R) -> R {
        fn strs(v: &[String]) -> Vec<FfiStr<'_>> {
            v.iter().map(|s| FfiStr::new(s)).collect()
        }
        let projection = self.projection.as_deref().map(strs);
        let filters = strs(&self.filters);
        let filter_columns = strs(&self.filter_columns);
        f(&ResolveArgs {
            projection: projection.as_deref().map(FfiSlice::new).into(),
            limit: self.limit.into(),
            filters: FfiSlice::new(&filters),
            filter_columns: FfiSlice::new(&filter_columns),
            filter_drop_columns_idx: self.filter_drop_columns_idx.into(),
            existing_version_key: self.existing_version_key.as_deref().map(FfiStr::new).into(),
            target_n_sources: self.target_n_sources.into(),
            verbose: self.verbose,
        })
    }

    /// Copy from the FFI form.
    pub fn from_ffi(args: &ResolveArgs<'_>) -> Result<Self, std::str::Utf8Error> {
        fn strs(v: FfiSlice<'_, FfiStr<'_>>) -> Result<Vec<String>, std::str::Utf8Error> {
            v.as_slice()
                .iter()
                .map(|s| s.as_str().map(str::to_owned))
                .collect()
        }
        Ok(Self {
            projection: args.projection.into_option().map(strs).transpose()?,
            limit: args.limit.into_option(),
            filters: strs(args.filters)?,
            filter_columns: strs(args.filter_columns)?,
            filter_drop_columns_idx: args.filter_drop_columns_idx.into_option(),
            existing_version_key: args
                .existing_version_key
                .into_option()
                .map(|s| s.as_str().map(str::to_owned))
                .transpose()?,
            target_n_sources: args.target_n_sources.into_option(),
            verbose: args.verbose,
        })
    }
}
