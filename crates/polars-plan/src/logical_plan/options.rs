use std::path::PathBuf;

use polars_core::prelude::*;
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

#[cfg(feature = "python")]
use crate::prelude::python_udf::PythonFunction;

pub type FileCount = u32;

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Generic options for all file types.
pub struct FileScanOptions {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub cache: bool,
    pub row_index: Option<RowIndex>,
    pub rechunk: bool,
    pub file_counter: FileCount,
    pub hive_options: HiveOptions,
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
pub struct DistinctOptions {
    /// Subset of columns that will be taken into account.
    pub subset: Option<Arc<Vec<String>>>,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FunctionOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    pub collect_groups: ApplyOptions,
    // used for formatting, (only for anonymous functions)
    #[cfg_attr(feature = "serde", serde(skip_deserializing))]
    pub fmt_str: &'static str,
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
    pub input_wildcard_expansion: bool,
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
    pub returns_scalar: bool,
    // if the expression and its inputs should be cast to supertypes
    pub cast_to_supertypes: bool,
    // The physical expression may rename the output of this function.
    // If set to `false` the physical engine will ensure the left input
    // expression is the output name.
    pub allow_rename: bool,
    // if set, then the `Series` passed to the function in the group_by operation
    // will ensure the name is set. This is an extra heap allocation per group.
    pub pass_name_to_apply: bool,
    // For example a `unique` or a `slice`
    pub changes_length: bool,
    // Validate the output of a `map`.
    // this should always be true or we could OOB
    pub check_lengths: UnsafeBool,
    // Raise if use in group by
    pub allow_group_aware: bool,
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
}

impl Default for FunctionOptions {
    fn default() -> Self {
        FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            input_wildcard_expansion: false,
            returns_scalar: false,
            fmt_str: "",
            cast_to_supertypes: false,
            allow_rename: false,
            pass_name_to_apply: false,
            changes_length: false,
            check_lengths: UnsafeBool(true),
            allow_group_aware: true,
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
    pub scan_fn: Option<PythonFunction>,
    pub schema: SchemaRef,
    pub output_schema: Option<SchemaRef>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub pyarrow: bool,
    // a pyarrow predicate python expression
    // can be evaluated with python.eval
    pub predicate: Option<String>,
    // a `head` call passed to pyarrow
    pub n_rows: Option<usize>,
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
