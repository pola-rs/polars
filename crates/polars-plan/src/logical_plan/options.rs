use std::path::PathBuf;

use polars_core::prelude::*;
#[cfg(feature = "csv")]
use polars_io::csv::SerializeOptions;
#[cfg(feature = "csv")]
use polars_io::csv::{CsvEncoding, NullValues};
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcCompression;
#[cfg(feature = "parquet")]
use polars_io::parquet::ParquetCompression;
use polars_io::RowCount;
#[cfg(feature = "dynamic_group_by")]
use polars_time::{DynamicGroupOptions, RollingGroupOptions};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use crate::prelude::python_udf::PythonFunction;

pub type FileCount = u32;

#[cfg(feature = "csv")]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsvParserOptions {
    pub separator: u8,
    pub comment_char: Option<u8>,
    pub quote_char: Option<u8>,
    pub eol_char: u8,
    pub has_header: bool,
    pub skip_rows: usize,
    pub low_memory: bool,
    pub ignore_errors: bool,
    pub null_values: Option<NullValues>,
    pub encoding: CsvEncoding,
    pub try_parse_dates: bool,
    pub raise_if_empty: bool,
    pub truncate_ragged_lines: bool,
}

#[cfg(feature = "parquet")]
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParquetOptions {
    pub parallel: polars_io::parquet::ParallelStrategy,
    pub low_memory: bool,
    pub use_statistics: bool,
}

#[cfg(feature = "parquet")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParquetWriteOptions {
    /// Data page compression
    pub compression: ParquetCompression,
    /// Compute and write column statistics.
    pub statistics: bool,
    /// If `None` will be all written to a single row group.
    pub row_group_size: Option<usize>,
    /// if `None` will be 1024^2 bytes
    pub data_pagesize_limit: Option<usize>,
    /// maintain the order the data was processed
    pub maintain_order: bool,
}

#[cfg(feature = "ipc")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpcWriterOptions {
    /// Data page compression
    pub compression: Option<IpcCompression>,
    /// maintain the order the data was processed
    pub maintain_order: bool,
}

#[cfg(feature = "csv")]
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsvWriterOptions {
    pub include_bom: bool,
    pub include_header: bool,
    pub batch_size: usize,
    pub maintain_order: bool,
    pub serialize_options: SerializeOptions,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpcScanOptions {
    pub memmap: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Generic options for all file types
pub struct FileScanOptions {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub cache: bool,
    pub row_count: Option<RowCount>,
    pub rechunk: bool,
    pub file_counter: FileCount,
    pub hive_partitioning: bool,
}

#[derive(Clone, Debug, Copy, Default, Eq, PartialEq)]
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

#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GroupbyOptions {
    #[cfg(feature = "dynamic_group_by")]
    pub dynamic: Option<DynamicGroupOptions>,
    #[cfg(feature = "dynamic_group_by")]
    pub rolling: Option<RollingGroupOptions>,
    /// Take only a slice of the result
    pub slice: Option<(i64, usize)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
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
    /// Say the schema is 'a', 'b' and there is a function f
    /// f('*')
    /// can expand to:
    /// 1.
    ///     f('a', 'b')
    /// or
    /// 2.
    ///     f('a'), f('b')
    ///
    /// setting this to true, will lead to behavior 1.
    ///
    /// this also accounts for regex expansion
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
pub struct SortArguments {
    pub descending: Vec<bool>,
    pub nulls_last: bool,
    pub slice: Option<(i64, usize)>,
    pub maintain_order: bool,
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

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnonymousScanOptions {
    pub skip_rows: Option<usize>,
    pub fmt_str: &'static str,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub enum FileType {
    #[cfg(feature = "parquet")]
    Parquet(ParquetWriteOptions),
    #[cfg(feature = "ipc")]
    Ipc(IpcWriterOptions),
    #[cfg(feature = "csv")]
    Csv(CsvWriterOptions),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct ProjectionOptions {
    pub run_parallel: bool,
    pub duplicate_check: bool,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            run_parallel: true,
            duplicate_check: true,
        }
    }
}
