use std::path::PathBuf;

use polars_core::prelude::*;
#[cfg(feature = "csv-file")]
use polars_io::csv::{CsvEncoding, NullValues};
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcCompression;
#[cfg(feature = "parquet")]
use polars_io::parquet::ParquetCompression;
use polars_io::RowCount;
#[cfg(feature = "dynamic_groupby")]
use polars_time::{DynamicGroupOptions, RollingGroupOptions};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::Expr;

pub type FileCount = u32;

#[cfg(feature = "csv-file")]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsvParserOptions {
    pub delimiter: u8,
    pub comment_char: Option<u8>,
    pub quote_char: Option<u8>,
    pub eol_char: u8,
    pub has_header: bool,
    pub skip_rows: usize,
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub low_memory: bool,
    pub ignore_errors: bool,
    pub cache: bool,
    pub null_values: Option<NullValues>,
    pub rechunk: bool,
    pub encoding: CsvEncoding,
    pub row_count: Option<RowCount>,
    pub parse_dates: bool,
    pub file_counter: FileCount,
}

#[cfg(feature = "parquet")]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParquetOptions {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub cache: bool,
    pub parallel: polars_io::parquet::ParallelStrategy,
    pub rechunk: bool,
    pub row_count: Option<RowCount>,
    pub file_counter: FileCount,
    pub low_memory: bool,
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

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpcScanOptions {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub cache: bool,
    pub row_count: Option<RowCount>,
    pub rechunk: bool,
    pub memmap: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpcScanOptionsInner {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub cache: bool,
    pub row_count: Option<RowCount>,
    pub rechunk: bool,
    pub file_counter: FileCount,
    pub memmap: bool,
}

impl From<IpcScanOptions> for IpcScanOptionsInner {
    fn from(options: IpcScanOptions) -> Self {
        Self {
            n_rows: options.n_rows,
            with_columns: options.with_columns,
            cache: options.cache,
            row_count: options.row_count,
            rechunk: options.rechunk,
            file_counter: Default::default(),
            memmap: options.memmap,
        }
    }
}

#[derive(Clone, Debug, Copy, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnionOptions {
    pub slice: bool,
    pub slice_offset: i64,
    pub slice_len: IdxSize,
    pub parallel: bool,
    // known row_output, estimated row output
    pub rows: (Option<usize>, usize),
    pub from_partitioned_ds: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GroupbyOptions {
    #[cfg(feature = "dynamic_groupby")]
    pub dynamic: Option<DynamicGroupOptions>,
    #[cfg(feature = "dynamic_groupby")]
    pub rolling: Option<RollingGroupOptions>,
    pub slice: Option<(i64, usize)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistinctOptions {
    pub subset: Option<Arc<Vec<String>>>,
    pub maintain_order: bool,
    pub keep_strategy: UniqueKeepStrategy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ApplyOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    // e.g. [g1, g1, g2] -> [[g1, g2], g2]
    ApplyGroups,
    // collect groups to a list and then apply
    // e.g. [g1, g1, g2] -> list([g1, g1, g2])
    ApplyList,
    // do not collect before apply
    // e.g. [g1, g1, g2] -> [g1, g1, g2]
    ApplyFlat,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WindowOptions {
    /// Explode the aggregated list and just do a hstack instead of a join
    /// this requires the groups to be sorted to make any sense
    pub explode: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FunctionOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    pub collect_groups: ApplyOptions,
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

    /// automatically explode on unit length it ran as final aggregation.
    ///
    /// this is the case for aggregations like sum, min, covariance etc.
    /// We need to know this because we cannot see the difference between
    /// the following functions based on the output type and number of elements:
    ///
    /// x: {1, 2, 3}
    ///
    /// head_1(x) -> {1}
    /// sum(x) -> {4}
    pub auto_explode: bool,
    // used for formatting, (only for anonymous functions)
    #[cfg_attr(feature = "serde", serde(skip_deserializing))]
    pub fmt_str: &'static str,

    // if the expression and its inputs should be cast to supertypes
    pub cast_to_supertypes: bool,
    // apply physical expression may rename the output of this function
    pub allow_rename: bool,
    // if set, then the `Series` passed to the function in the groupby operation
    // will ensure the name is set. This is an extra heap allocation per group.
    pub pass_name_to_apply: bool,
}

impl FunctionOptions {
    /// Any function that is sensitive to the number of elements in a group
    /// - Aggregations
    /// - Sorts
    /// - Counts
    pub fn is_groups_sensitive(&self) -> bool {
        matches!(self.collect_groups, ApplyOptions::ApplyGroups)
    }
}

impl Default for FunctionOptions {
    fn default() -> Self {
        FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: false,
            auto_explode: false,
            fmt_str: "",
            cast_to_supertypes: false,
            allow_rename: false,
            pass_name_to_apply: false,
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

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SortArguments {
    pub reverse: Vec<bool>,
    // Can only be true in case of a single column.
    pub nulls_last: bool,
    pub slice: Option<(i64, usize)>,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg(feature = "python")]
pub struct PythonOptions {
    // Serialized Fn() -> PolarsResult<DataFrame>
    pub scan_fn: Vec<u8>,
    pub schema: SchemaRef,
    pub output_schema: Option<SchemaRef>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub pyarrow: bool,
    // a pyarrow predicate python expression
    // can be evaluated with python.eval
    pub predicate: Option<String>,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnonymousScanOptions {
    pub schema: SchemaRef,
    pub output_schema: Option<SchemaRef>,
    pub skip_rows: Option<usize>,
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub predicate: Option<Expr>,
    pub fmt_str: &'static str,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct FileSinkOptions {
    pub path: Arc<PathBuf>,
    pub file_type: FileType,
}

#[cfg(any(feature = "parquet", feature = "ipc"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub enum FileType {
    #[cfg(feature = "parquet")]
    Parquet(ParquetWriteOptions),
    #[cfg(feature = "ipc")]
    Ipc(IpcWriterOptions),
}

#[cfg(not(any(feature = "parquet", feature = "ipc")))]
pub type FileType = ();
