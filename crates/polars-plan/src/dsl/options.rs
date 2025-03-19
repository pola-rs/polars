use std::hash::{Hash, Hasher};
#[cfg(feature = "json")]
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::*;
#[cfg(feature = "csv")]
use polars_io::csv::write::CsvWriterOptions;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcWriterOptions;
#[cfg(feature = "json")]
use polars_io::json::JsonWriterOptions;
#[cfg(feature = "parquet")]
use polars_io::parquet::write::ParquetWriteOptions;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_io::{HiveOptions, RowIndex, is_cloud_url};
#[cfg(feature = "iejoin")]
use polars_ops::frame::IEJoinOptions;
use polars_ops::frame::{CrossJoinFilter, CrossJoinOptions, JoinTypeOptions};
use polars_ops::prelude::{JoinArgs, JoinType};
#[cfg(feature = "dynamic_group_by")]
use polars_time::DynamicGroupOptions;
#[cfg(feature = "dynamic_group_by")]
use polars_time::RollingGroupOptions;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use super::{AExpr, Expr, ExprIR};
use crate::dsl::Selector;

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingCovOptions {
    pub window_size: IdxSize,
    pub min_periods: IdxSize,
    pub ddof: u8,
}

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StrptimeOptions {
    /// Formatting string
    pub format: Option<PlSmallStr>,
    /// If set then polars will return an error if any date parsing fails
    pub strict: bool,
    /// If polars may parse matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    pub exact: bool,
    /// use a cache of unique, converted dates to apply the datetime conversion.
    pub cache: bool,
}

impl Default for StrptimeOptions {
    fn default() -> Self {
        StrptimeOptions {
            format: None,
            strict: true,
            exact: true,
            cache: true,
        }
    }
}

#[derive(Clone, PartialEq, Eq, IntoStaticStr, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum JoinTypeOptionsIR {
    #[cfg(feature = "iejoin")]
    IEJoin(IEJoinOptions),
    #[cfg_attr(all(feature = "serde", not(feature = "ir_serde")), serde(skip))]
    // Fused cross join and filter (only in in-memory engine)
    Cross { predicate: ExprIR },
}

impl Hash for JoinTypeOptionsIR {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use JoinTypeOptionsIR::*;
        match self {
            #[cfg(feature = "iejoin")]
            IEJoin(opt) => opt.hash(state),
            Cross { predicate } => predicate.node().hash(state),
        }
    }
}

impl JoinTypeOptionsIR {
    pub fn compile<C: FnOnce(&ExprIR) -> PolarsResult<Arc<dyn CrossJoinFilter>>>(
        self,
        plan: C,
    ) -> PolarsResult<JoinTypeOptions> {
        use JoinTypeOptionsIR::*;
        match self {
            Cross { predicate } => {
                let predicate = plan(&predicate)?;

                Ok(JoinTypeOptions::Cross(CrossJoinOptions { predicate }))
            },
            #[cfg(feature = "iejoin")]
            IEJoin(opt) => Ok(JoinTypeOptions::IEJoin(opt)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JoinOptions {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    pub args: JoinArgs,
    pub options: Option<JoinTypeOptionsIR>,
    /// Proxy of the number of rows in both sides of the joins
    /// Holds `(Option<known_size>, estimated_size)`
    pub rows_left: (Option<usize>, usize),
    pub rows_right: (Option<usize>, usize),
}

impl Default for JoinOptions {
    fn default() -> Self {
        JoinOptions {
            allow_parallel: true,
            force_parallel: false,
            // Todo!: make default
            args: JoinArgs::new(JoinType::Left),
            options: Default::default(),
            rows_left: (None, usize::MAX),
            rows_right: (None, usize::MAX),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum WindowType {
    /// Explode the aggregated list and just do a hstack instead of a join
    /// this requires the groups to be sorted to make any sense
    Over(WindowMapping),
    #[cfg(feature = "dynamic_group_by")]
    Rolling(RollingGroupOptions),
}

impl From<WindowMapping> for WindowType {
    fn from(value: WindowMapping) -> Self {
        Self::Over(value)
    }
}

impl Default for WindowType {
    fn default() -> Self {
        Self::Over(WindowMapping::default())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum WindowMapping {
    /// Map the group values to the position
    #[default]
    GroupsToRows,
    /// Explode the aggregated list and just do a hstack instead of a join
    /// this requires the groups to be sorted to make any sense
    Explode,
    /// Join the groups as 'List<group_dtype>' to the row positions.
    /// warning: this can be memory intensive
    Join,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NestedType {
    #[cfg(feature = "dtype-array")]
    Array,
    // List,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnpivotArgsDSL {
    pub on: Vec<Selector>,
    pub index: Vec<Selector>,
    pub variable_name: Option<PlSmallStr>,
    pub value_name: Option<PlSmallStr>,
}

pub type FileCount = u32;

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Generic options for all file types.
pub struct FileScanOptions {
    // Slice applied before predicates
    pub pre_slice: Option<(i64, usize)>,
    pub with_columns: Option<Arc<[PlSmallStr]>>,
    pub cache: bool,
    pub row_index: Option<RowIndex>,
    pub rechunk: bool,
    pub file_counter: FileCount,
    pub hive_options: HiveOptions,
    pub glob: bool,
    pub include_file_paths: Option<PlSmallStr>,
    pub allow_missing_columns: bool,
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Engine {
    Auto,
    OldStreaming,
    Streaming,
    InMemory,
    Gpu,
}

impl FromStr for Engine {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            // "cpu" for backwards compatibility
            "auto" => Ok(Engine::Auto),
            "cpu" | "in-memory" => Ok(Engine::InMemory),
            "streaming" => Ok(Engine::Streaming),
            "old-streaming" => Ok(Engine::OldStreaming),
            "gpu" => Ok(Engine::Gpu),
            v => Err(format!(
                "`engine` must be one of {{'auto', 'in-memory', 'streaming', 'old-streaming', 'gpu'}}, got {v}",
            )),
        }
    }
}

impl Engine {
    pub fn into_static_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::OldStreaming => "old-streaming",
            Self::Streaming => "streaming",
            Self::InMemory => "in-memory",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Debug, Copy, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnionOptions {
    pub slice: Option<(i64, usize)>,
    // known row_output, estimated row output
    pub rows: (Option<usize>, usize),
    pub parallel: bool,
    pub from_partitioned_ds: bool,
    pub flattened_by_opt: bool,
    pub rechunk: bool,
    pub maintain_order: bool,
}

#[derive(Clone, Debug, Copy, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HConcatOptions {
    pub parallel: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GroupbyOptions {
    #[cfg(feature = "dynamic_group_by")]
    pub dynamic: Option<DynamicGroupOptions>,
    #[cfg(feature = "dynamic_group_by")]
    pub rolling: Option<RollingGroupOptions>,
    /// Take only a slice of the result
    pub slice: Option<(i64, usize)>,
}

impl GroupbyOptions {
    pub(crate) fn is_rolling(&self) -> bool {
        #[cfg(feature = "dynamic_group_by")]
        {
            self.rolling.is_some()
        }
        #[cfg(not(feature = "dynamic_group_by"))]
        {
            false
        }
    }

    pub(crate) fn is_dynamic(&self) -> bool {
        #[cfg(feature = "dynamic_group_by")]
        {
            self.dynamic.is_some()
        }
        #[cfg(not(feature = "dynamic_group_by"))]
        {
            false
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistinctOptionsDSL {
    /// Subset of columns that will be taken into account.
    pub subset: Option<Vec<Selector>>,
    /// This will maintain the order of the input.
    /// Note that this is more expensive.
    /// `maintain_order` is not supported in the streaming
    /// engine.
    pub maintain_order: bool,
    /// Which rows to keep.
    pub keep_strategy: UniqueKeepStrategy,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LogicalPlanUdfOptions {
    ///  allow predicate pushdown optimizations
    pub predicate_pd: bool,
    ///  allow projection pushdown optimizations
    pub projection_pd: bool,
    // used for formatting
    pub fmt_str: &'static str,
}

#[derive(Clone, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnonymousScanOptions {
    pub skip_rows: Option<usize>,
    pub fmt_str: &'static str,
}

/// Options that apply to all sinks.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SinkOptions {
    /// Call sync when closing the file.
    pub sync_on_close: SyncOnCloseType,

    /// The output file needs to maintain order of the data that comes in.
    pub maintain_order: bool,

    /// Recursively create all the directories in the path.
    pub mkdir: bool,
}

impl Default for SinkOptions {
    fn default() -> Self {
        Self {
            sync_on_close: Default::default(),
            maintain_order: true,
            mkdir: false,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileSinkType {
    pub path: Arc<PathBuf>,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SinkTypeIR {
    Memory,
    File(FileSinkType),
    Partition(PartitionSinkTypeIR),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionSinkType {
    pub path_f_string: Arc<PathBuf>,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub variant: PartitionVariant,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionSinkTypeIR {
    pub path_f_string: Arc<PathBuf>,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub variant: PartitionVariantIR,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SinkType {
    Memory,
    File(FileSinkType),
    Partition(PartitionSinkType),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PartitionVariant {
    MaxSize(IdxSize),
    Parted {
        key_exprs: Vec<Expr>,
        include_key: bool,
    },
    ByKey {
        key_exprs: Vec<Expr>,
        include_key: bool,
    },
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartitionVariantIR {
    MaxSize(IdxSize),
    Parted {
        key_exprs: Vec<ExprIR>,
        include_key: bool,
    },
    ByKey {
        key_exprs: Vec<ExprIR>,
        include_key: bool,
    },
}

impl SinkTypeIR {
    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Memory => {},
            Self::File(f) => f.hash(state),
            Self::Partition(f) => {
                f.path_f_string.hash(state);
                f.file_type.hash(state);
                f.sink_options.hash(state);
                f.variant.traverse_and_hash(expr_arena, state);
                f.cloud_options.hash(state);
            },
        }
    }
}

impl PartitionVariantIR {
    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::MaxSize(size) => size.hash(state),
            Self::Parted {
                key_exprs,
                include_key,
            }
            | Self::ByKey {
                key_exprs,
                include_key,
            } => {
                include_key.hash(state);
                for key_expr in key_exprs.as_slice() {
                    key_expr.traverse_and_hash(expr_arena, state);
                }
            },
        }
    }
}

impl SinkType {
    pub(crate) fn is_cloud_destination(&self) -> bool {
        if let Self::File(f) = self {
            if is_cloud_url(f.path.as_ref()) {
                return true;
            }
        }

        false
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct FileSinkOptions {
    pub path: Arc<PathBuf>,
    pub file_type: FileType,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FileType {
    #[cfg(feature = "parquet")]
    Parquet(ParquetWriteOptions),
    #[cfg(feature = "ipc")]
    Ipc(IpcWriterOptions),
    #[cfg(feature = "csv")]
    Csv(CsvWriterOptions),
    #[cfg(feature = "json")]
    Json(JsonWriterOptions),
}
//
// Arguments given to `concat`. Differs from `UnionOptions` as the latter is IR state.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnionArgs {
    pub parallel: bool,
    pub rechunk: bool,
    pub to_supertypes: bool,
    pub diagonal: bool,
    // If it is a union from a scan over multiple files.
    pub from_partitioned_ds: bool,
    pub maintain_order: bool,
}

impl Default for UnionArgs {
    fn default() -> Self {
        Self {
            parallel: true,
            rechunk: false,
            to_supertypes: false,
            diagonal: false,
            from_partitioned_ds: false,
            maintain_order: true,
        }
    }
}

impl From<UnionArgs> for UnionOptions {
    fn from(args: UnionArgs) -> Self {
        UnionOptions {
            slice: None,
            parallel: args.parallel,
            rows: (None, 0),
            from_partitioned_ds: args.from_partitioned_ds,
            flattened_by_opt: false,
            rechunk: args.rechunk,
            maintain_order: args.maintain_order,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg(feature = "json")]
pub struct NDJsonReadOptions {
    pub n_threads: Option<usize>,
    pub infer_schema_length: Option<NonZeroUsize>,
    pub chunk_size: NonZeroUsize,
    pub low_memory: bool,
    pub ignore_errors: bool,
    pub schema: Option<SchemaRef>,
    pub schema_overwrite: Option<SchemaRef>,
}
