use polars_core::prelude::*;
use polars_io::csv::{CsvEncoding, NullValues};
use polars_io::RowCount;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

pub type FileCount = u32;

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsvParserOptions {
    pub(crate) delimiter: u8,
    pub(crate) comment_char: Option<u8>,
    pub(crate) quote_char: Option<u8>,
    pub(crate) eol_char: u8,
    pub(crate) has_header: bool,
    pub(crate) skip_rows: usize,
    pub(crate) n_rows: Option<usize>,
    pub(crate) with_columns: Option<Arc<Vec<String>>>,
    pub(crate) low_memory: bool,
    pub(crate) ignore_errors: bool,
    pub(crate) cache: bool,
    pub(crate) null_values: Option<NullValues>,
    pub(crate) rechunk: bool,
    pub(crate) encoding: CsvEncoding,
    pub(crate) row_count: Option<RowCount>,
    pub(crate) parse_dates: bool,
    pub(crate) file_counter: FileCount,
}
#[cfg(feature = "parquet")]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParquetOptions {
    pub(crate) n_rows: Option<usize>,
    pub(crate) with_columns: Option<Arc<Vec<String>>>,
    pub(crate) cache: bool,
    pub(crate) parallel: polars_io::parquet::ParallelStrategy,
    pub(crate) rechunk: bool,
    pub(crate) row_count: Option<RowCount>,
    pub(crate) file_counter: FileCount,
    pub(crate) low_memory: bool,
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
    pub(crate) n_rows: Option<usize>,
    pub(crate) with_columns: Option<Arc<Vec<String>>>,
    pub(crate) cache: bool,
    pub(crate) row_count: Option<RowCount>,
    pub(crate) rechunk: bool,
    pub(crate) file_counter: FileCount,
    pub(crate) memmap: bool,
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
    pub(crate) slice: bool,
    pub(crate) slice_offset: i64,
    pub(crate) slice_len: IdxSize,
    pub(crate) parallel: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GroupbyOptions {
    pub(crate) dynamic: Option<DynamicGroupOptions>,
    pub(crate) rolling: Option<RollingGroupOptions>,
    pub(crate) slice: Option<(i64, usize)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistinctOptions {
    pub(crate) subset: Option<Arc<Vec<String>>>,
    pub(crate) maintain_order: bool,
    pub(crate) keep_strategy: UniqueKeepStrategy,
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
    pub(crate) explode: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FunctionOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    pub(crate) collect_groups: ApplyOptions,
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
    pub(crate) input_wildcard_expansion: bool,

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
    pub(crate) auto_explode: bool,
    // used for formatting, (only for anonymous functions)
    #[cfg_attr(feature = "serde", serde(skip_deserializing))]
    pub(crate) fmt_str: &'static str,

    // if the expression and its inputs should be cast to supertypes
    pub(crate) cast_to_supertypes: bool,
    pub(crate) allow_rename: bool,
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
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LogicalPlanUdfOptions {
    ///  allow predicate pushdown optimizations
    pub(crate) predicate_pd: bool,
    ///  allow projection pushdown optimizations
    pub(crate) projection_pd: bool,
    // used for formatting
    pub(crate) fmt_str: &'static str,
}

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SortArguments {
    pub(crate) reverse: Vec<bool>,
    // Can only be true in case of a single column.
    pub(crate) nulls_last: bool,
    pub(crate) slice: Option<(i64, usize)>,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg(feature = "python")]
pub struct PythonOptions {
    // Serialized Fn() -> PolarsResult<DataFrame>
    pub(crate) scan_fn: Vec<u8>,
    pub(crate) schema: SchemaRef,
    pub(crate) output_schema: Option<SchemaRef>,
    pub(crate) with_columns: Option<Arc<Vec<String>>>,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnonymousScanOptions {
    pub schema: SchemaRef,
    pub output_schema: Option<SchemaRef>,
    pub skip_rows: Option<usize>,
    pub n_rows: Option<usize>,
    pub with_columns: Option<Arc<Vec<String>>>,
    pub(crate) fmt_str: &'static str,
}
