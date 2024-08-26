#[cfg(feature = "json")]
use std::num::NonZeroUsize;
use std::path::PathBuf;

use bitflags::bitflags;
use polars_core::prelude::*;
use polars_core::utils::SuperTypeOptions;
#[cfg(feature = "csv")]
use polars_io::csv::write::CsvWriterOptions;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcWriterOptions;
#[cfg(feature = "json")]
use polars_io::json::JsonWriterOptions;
#[cfg(feature = "parquet")]
use polars_io::parquet::write::ParquetWriteOptions;
use polars_io::{HiveOptions, RowIndex};
#[cfg(feature = "dynamic_group_by")]
use polars_time::{DynamicGroupOptions, RollingGroupOptions};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::dsl::Selector;
use crate::plans::{ColumnName, ExprIR};
#[cfg(feature = "python")]
use crate::prelude::python_udf::PythonFunction;

pub type FileCount = u32;

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Generic options for all file types.
pub struct FileScanOptions {
    pub slice: Option<(i64, usize)>,
    pub with_columns: Option<Arc<[String]>>,
    pub cache: bool,
    pub row_index: Option<RowIndex>,
    pub rechunk: bool,
    pub file_counter: FileCount,
    pub hive_options: HiveOptions,
    pub glob: bool,
    pub include_file_paths: Option<Arc<str>>,
}

#[derive(Clone, Debug, Copy, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnionOptions {
    pub slice: Option<(i64, usize)>,
    pub parallel: bool,
    // known row_output, estimated row output
    pub rows: (Option<usize>, usize),
    pub from_partitioned_ds: bool,
    pub flattened_by_opt: bool,
    pub rechunk: bool,
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct DistinctOptionsIR {
    /// Subset of columns that will be taken into account.
    pub subset: Option<Arc<[ColumnName]>>,
    /// This will maintain the order of the input.
    /// Note that this is more expensive.
    /// `maintain_order` is not supported in the streaming
    /// engine.
    pub maintain_order: bool,
    /// Which rows to keep.
    pub keep_strategy: UniqueKeepStrategy,
    /// Take only a slice of the result
    pub slice: Option<(i64, usize)>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ApplyOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    // e.g. [g1, g1, g2] -> [[g1, g1], g2]
    GroupWise,
    // collect groups to a list and then apply
    // e.g. [g1, g1, g2] -> list([g1, g1, g2])
    ApplyList,
    // do not collect before apply
    // e.g. [g1, g1, g2] -> [g1, g1, g2]
    ElementWise,
}

// a boolean that can only be set to `false` safely
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnsafeBool(bool);
impl Default for UnsafeBool {
    fn default() -> Self {
        UnsafeBool(true)
    }
}

bitflags!(
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct FunctionFlags: u8 {
            // Raise if use in group by
            const ALLOW_GROUP_AWARE = 1 << 0;
            // For example a `unique` or a `slice`
            const CHANGES_LENGTH = 1 << 1;
            // The physical expression may rename the output of this function.
            // If set to `false` the physical engine will ensure the left input
            // expression is the output name.
            const ALLOW_RENAME = 1 << 2;
            // if set, then the `Series` passed to the function in the group_by operation
            // will ensure the name is set. This is an extra heap allocation per group.
            const PASS_NAME_TO_APPLY = 1 << 3;
            /// There can be two ways of expanding wildcards:
            ///
            /// Say the schema is 'a', 'b' and there is a function `f`. In this case, `f('*')` can expand
            /// to:
            /// 1. `f('a', 'b')`
            /// 2. `f('a'), f('b')`
            ///
            /// Setting this to true, will lead to behavior 1.
            ///
            /// This also accounts for regex expansion.
            const INPUT_WILDCARD_EXPANSION = 1 << 4;
            /// Automatically explode on unit length if it ran as final aggregation.
            ///
            /// this is the case for aggregations like sum, min, covariance etc.
            /// We need to know this because we cannot see the difference between
            /// the following functions based on the output type and number of elements:
            ///
            /// x: {1, 2, 3}
            ///
            /// head_1(x) -> {1}
            /// sum(x) -> {4}
            const RETURNS_SCALAR = 1 << 5;
            /// This can happen with UDF's that use Polars within the UDF.
            /// This can lead to recursively entering the engine and sometimes deadlocks.
            /// This flag must be set to handle that.
            const OPTIONAL_RE_ENTRANT = 1 << 6;
            /// Whether this function allows no inputs.
            const ALLOW_EMPTY_INPUTS = 1 << 7;
        }
);

impl Default for FunctionFlags {
    fn default() -> Self {
        Self::from_bits_truncate(0) | Self::ALLOW_GROUP_AWARE
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FunctionOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    pub collect_groups: ApplyOptions,
    // used for formatting, (only for anonymous functions)
    #[cfg_attr(feature = "serde", serde(skip_deserializing))]
    pub fmt_str: &'static str,
    // if the expression and its inputs should be cast to supertypes
    // `None` -> Don't cast.
    // `Some` -> cast with given options.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub cast_to_supertypes: Option<SuperTypeOptions>,
    // Validate the output of a `map`.
    // this should always be true or we could OOB
    pub check_lengths: UnsafeBool,
    pub flags: FunctionFlags,
}

impl FunctionOptions {
    /// Any function that is sensitive to the number of elements in a group
    /// - Aggregations
    /// - Sorts
    /// - Counts
    pub fn is_groups_sensitive(&self) -> bool {
        matches!(self.collect_groups, ApplyOptions::GroupWise)
    }

    #[cfg(feature = "fused")]
    pub(crate) unsafe fn no_check_lengths(&mut self) {
        self.check_lengths = UnsafeBool(false);
    }
    pub fn check_lengths(&self) -> bool {
        self.check_lengths.0
    }

    pub fn is_elementwise(&self) -> bool {
        self.collect_groups == ApplyOptions::ElementWise
            && !self
                .flags
                .contains(FunctionFlags::CHANGES_LENGTH | FunctionFlags::RETURNS_SCALAR)
    }
}

impl Default for FunctionOptions {
    fn default() -> Self {
        FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            fmt_str: "",
            cast_to_supertypes: None,
            check_lengths: UnsafeBool(true),
            flags: Default::default(),
        }
    }
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

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg(feature = "python")]
pub struct PythonOptions {
    /// A function that returns a Python Generator.
    /// The generator should produce Polars DataFrame's.
    pub scan_fn: Option<PythonFunction>,
    /// Schema of the file.
    pub schema: SchemaRef,
    /// Schema the reader will produce when the file is read.
    pub output_schema: Option<SchemaRef>,
    // Projected column names.
    pub with_columns: Option<Arc<[String]>>,
    // Which interface is the python function.
    pub python_source: PythonScanSource,
    /// Optional predicate the reader must apply.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub predicate: PythonPredicate,
    /// A `head` call passed to the reader.
    pub n_rows: Option<usize>,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PythonScanSource {
    Pyarrow,
    Cuda,
    #[default]
    IOPlugin,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum PythonPredicate {
    // A pyarrow predicate python expression
    // can be evaluated with python.eval
    PyArrow(String),
    Polars(ExprIR),
    #[default]
    None,
}

#[derive(Clone, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnonymousScanOptions {
    pub skip_rows: Option<usize>,
    pub fmt_str: &'static str,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SinkType {
    Memory,
    File {
        path: Arc<PathBuf>,
        file_type: FileType,
    },
    #[cfg(feature = "cloud")]
    Cloud {
        uri: Arc<String>,
        file_type: FileType,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
    },
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProjectionOptions {
    pub run_parallel: bool,
    pub duplicate_check: bool,
    // Should length-1 Series be broadcast to the length of the dataframe.
    // Only used by CSE optimizer
    pub should_broadcast: bool,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            run_parallel: true,
            duplicate_check: true,
            should_broadcast: true,
        }
    }
}

impl ProjectionOptions {
    /// Conservatively merge the options of two [`ProjectionOptions`]
    pub fn merge_options(&self, other: &Self) -> Self {
        Self {
            run_parallel: self.run_parallel & other.run_parallel,
            duplicate_check: self.duplicate_check & other.duplicate_check,
            should_broadcast: self.should_broadcast | other.should_broadcast,
        }
    }
}

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
}

impl Default for UnionArgs {
    fn default() -> Self {
        Self {
            parallel: true,
            rechunk: false,
            to_supertypes: false,
            diagonal: false,
            from_partitioned_ds: false,
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
