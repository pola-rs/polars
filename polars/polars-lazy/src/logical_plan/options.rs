use crate::prelude::*;
use polars_core::prelude::*;
use polars_io::csv::{CsvEncoding, NullValues};
use polars_io::RowCount;

#[derive(Clone, Debug)]
pub struct CsvParserOptions {
    pub(crate) delimiter: u8,
    pub(crate) comment_char: Option<u8>,
    pub(crate) quote_char: Option<u8>,
    pub(crate) has_header: bool,
    pub(crate) skip_rows: usize,
    pub(crate) n_rows: Option<usize>,
    pub(crate) with_columns: Option<Vec<String>>,
    pub(crate) low_memory: bool,
    pub(crate) ignore_errors: bool,
    pub(crate) cache: bool,
    pub(crate) null_values: Option<NullValues>,
    pub(crate) rechunk: bool,
    pub(crate) encoding: CsvEncoding,
    pub(crate) row_count: Option<RowCount>,
}
#[cfg(feature = "parquet")]
#[derive(Clone, Debug)]
pub struct ParquetOptions {
    pub(crate) n_rows: Option<usize>,
    pub(crate) with_columns: Option<Vec<String>>,
    pub(crate) cache: bool,
    pub(crate) parallel: bool,
    pub(crate) row_count: Option<RowCount>,
}

#[derive(Clone, Debug)]
pub struct IpcScanOptions {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Vec<String>>,
    pub cache: bool,
    pub row_count: Option<RowCount>,
}

#[derive(Clone, Debug, Copy, Default)]
pub struct UnionOptions {
    pub(crate) slice: bool,
    pub(crate) slice_offset: i64,
    pub(crate) slice_len: u32,
}

#[derive(Clone, Debug)]
pub struct GroupbyOptions {
    pub(crate) dynamic: Option<DynamicGroupOptions>,
    pub(crate) rolling: Option<RollingGroupOptions>,
}

#[derive(Clone, Debug)]
pub struct DistinctOptions {
    pub(crate) subset: Option<Arc<Vec<String>>>,
    pub(crate) maintain_order: bool,
    pub(crate) keep_strategy: DistinctKeepStrategy,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ApplyOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    ApplyGroups,
    // collect groups to a list and then apply
    ApplyList,
    // do not collect before apply
    ApplyFlat,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WindowOptions {
    /// Explode the aggregated list and just do a hstack instead of a join
    /// this requires the groups to be sorted to make any sense
    pub(crate) explode: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
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
    /// x: [1, 2, 3]
    ///
    /// head_1(x) -> [1]
    /// sum(x) -> [4]
    pub(crate) auto_explode: bool,
    // used for formatting
    pub(crate) fmt_str: &'static str,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LogicalPlanUdfOptions {
    ///  allow predicate pushdown optimizations
    pub(crate) predicate_pd: bool,
    ///  allow projection pushdown optimizations
    pub(crate) projection_pd: bool,
    // used for formatting
    pub(crate) fmt_str: &'static str,
}

#[derive(Clone, PartialEq, Debug)]
pub struct SortArguments {
    pub(crate) reverse: Vec<bool>,
    // Can only be true in case of a single column.
    pub(crate) nulls_last: bool,
}
