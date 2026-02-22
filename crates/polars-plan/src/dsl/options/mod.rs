use std::hash::Hash;
#[cfg(feature = "json")]
use std::num::NonZeroUsize;
use std::sync::Arc;

pub mod file_provider;
pub mod sink;
pub use polars_config::Engine;
use polars_core::error::PolarsResult;
use polars_core::prelude::*;
#[cfg(feature = "csv")]
use polars_io::csv::write::CsvWriterOptions;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcWriterOptions;
#[cfg(feature = "json")]
use polars_io::ndjson::NDJsonWriterOptions;
#[cfg(feature = "parquet")]
use polars_io::parquet::write::ParquetWriteOptions;
#[cfg(feature = "iejoin")]
use polars_ops::frame::IEJoinOptions;
use polars_ops::frame::{CrossJoinFilter, CrossJoinOptions, JoinTypeOptions};
use polars_ops::prelude::{JoinArgs, JoinType};
#[cfg(feature = "dynamic_group_by")]
use polars_time::DynamicGroupOptions;
#[cfg(feature = "dynamic_group_by")]
use polars_time::RollingGroupOptions;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use sink::{
    CallbackSinkType, FileSinkOptions, PartitionStrategy, PartitionStrategyIR,
    PartitionedSinkOptions, PartitionedSinkOptionsIR, SinkDestination, SinkTarget, SinkType,
    SinkTypeIR, UnifiedSinkArgs,
};
use strum_macros::IntoStaticStr;

use super::Expr;
use crate::dsl::Selector;
use crate::plans::ExprIR;

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RollingCovOptions {
    pub window_size: IdxSize,
    pub min_periods: IdxSize,
    pub ddof: u8,
}

#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum JoinTypeOptionsIR {
    #[cfg(feature = "iejoin")]
    IEJoin(IEJoinOptions),
    // Fused cross join and filter (only used in the in-memory engine)
    CrossAndFilter {
        predicate: ExprIR, // Must be elementwise.
    },
}

impl Hash for JoinTypeOptionsIR {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use JoinTypeOptionsIR::*;
        match self {
            #[cfg(feature = "iejoin")]
            IEJoin(opt) => opt.hash(state),
            CrossAndFilter { predicate } => {
                predicate.node().hash(state);
            },
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
            CrossAndFilter { predicate } => {
                let predicate = plan(&predicate)?;

                Ok(JoinTypeOptions::Cross(CrossJoinOptions { predicate }))
            },
            #[cfg(feature = "iejoin")]
            IEJoin(opt) => Ok(JoinTypeOptions::IEJoin(opt)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct JoinOptionsIR {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    pub args: JoinArgs,
    pub options: Option<JoinTypeOptionsIR>,
}

impl From<JoinOptions> for JoinOptionsIR {
    fn from(opts: JoinOptions) -> Self {
        Self {
            allow_parallel: opts.allow_parallel,
            force_parallel: opts.force_parallel,
            args: opts.args,
            options: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct JoinOptions {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    pub args: JoinArgs,
}

impl Default for JoinOptions {
    fn default() -> Self {
        Self {
            allow_parallel: true,
            force_parallel: false,
            // Todo!: make default
            args: JoinArgs::new(JoinType::Left),
        }
    }
}

impl From<JoinOptionsIR> for JoinOptions {
    fn from(opts: JoinOptionsIR) -> Self {
        Self {
            allow_parallel: opts.allow_parallel,
            force_parallel: opts.force_parallel,
            args: opts.args,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct UnpivotArgsDSL {
    pub on: Option<Selector>,
    pub index: Selector,
    pub variable_name: Option<PlSmallStr>,
    pub value_name: Option<PlSmallStr>,
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
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

impl Default for UnionOptions {
    fn default() -> Self {
        Self {
            slice: None,
            rows: (None, 0),
            parallel: true,
            from_partitioned_ds: false,
            flattened_by_opt: false,
            rechunk: false,
            maintain_order: true,
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct HConcatOptions {
    pub parallel: bool,
    pub strict: bool,
    // Treat unit values as scalar.
    // E.g. broadcast them instead of fill nulls.
    pub broadcast_unit_length: bool,
}

impl Default for HConcatOptions {
    fn default() -> Self {
        Self {
            parallel: true,
            strict: false,
            broadcast_unit_length: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct GroupbyOptions {
    #[cfg(feature = "dynamic_group_by")]
    pub dynamic: Option<DynamicGroupOptions>,
    #[cfg(feature = "dynamic_group_by")]
    pub rolling: Option<RollingGroupOptions>,
    /// Take only a slice of the result
    pub slice: Option<(i64, usize)>,
}

impl GroupbyOptions {
    pub fn is_rolling(&self) -> bool {
        #[cfg(feature = "dynamic_group_by")]
        {
            self.rolling.is_some()
        }
        #[cfg(not(feature = "dynamic_group_by"))]
        {
            false
        }
    }

    pub fn is_dynamic(&self) -> bool {
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
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct DistinctOptionsDSL {
    /// Subset of columns/expressions that will be taken into account.
    pub subset: Option<Vec<Expr>>,
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

const _: () = {
    assert!(std::mem::size_of::<FileWriteFormat>() <= 50);
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Hash, strum_macros::IntoStaticStr)]
pub enum FileWriteFormat {
    #[cfg(feature = "parquet")]
    Parquet(Arc<ParquetWriteOptions>),
    #[cfg(feature = "ipc")]
    Ipc(IpcWriterOptions),
    #[cfg(feature = "csv")]
    Csv(CsvWriterOptions),
    #[cfg(feature = "json")]
    NDJson(NDJsonWriterOptions),
}

impl FileWriteFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            #[cfg(feature = "parquet")]
            Self::Parquet(_) => "parquet",
            #[cfg(feature = "ipc")]
            Self::Ipc(_) => "ipc",
            #[cfg(feature = "csv")]
            Self::Csv(_) => "csv",
            #[cfg(feature = "json")]
            Self::NDJson(_) => "jsonl",

            #[allow(unreachable_patterns)]
            _ => unreachable!("enable file type features"),
        }
    }
}

//
// Arguments given to `concat`. Differs from `UnionOptions` as the latter is IR state.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnionArgs {
    pub parallel: bool,
    pub rechunk: bool,
    pub to_supertypes: bool,
    pub diagonal: bool,
    pub strict: bool,
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
            // By default, strict should be true in v2.0.0
            strict: false,
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
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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
