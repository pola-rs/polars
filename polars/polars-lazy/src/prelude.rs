pub(crate) use polars_utils::arena::{Arena, Node};

#[cfg(feature = "temporal")]
pub(crate) use polars_time::in_nanoseconds_window;
#[cfg(feature = "dynamic_groupby")]
pub(crate) use polars_time::{DynamicGroupOptions, PolarsTemporalGroupby, RollingGroupOptions};

#[cfg(not(feature = "dynamic_groupby"))]
#[derive(Clone, Debug)]
pub struct DynamicGroupOptions {
    pub index_column: String,
}
#[cfg(not(feature = "dynamic_groupby"))]
#[derive(Clone, Debug)]
pub struct RollingGroupOptions {
    pub index_column: String,
}

pub use crate::{
    dsl::*,
    frame::*,
    logical_plan::{
        optimizer::{type_coercion::TypeCoercionRule, Optimize, *},
        options::*,
        *,
    },
    physical_plan::{expressions::*, planner::DefaultPlanner, Executor, PhysicalPlanner},
};

#[cfg(feature = "csv-file")]
pub(crate) use crate::physical_plan::executors::scan::CsvExec;
#[cfg(feature = "parquet")]
pub(crate) use crate::physical_plan::executors::scan::ParquetExec;

pub(crate) use crate::{
    logical_plan::{aexpr::*, alp::*, conversion::*},
    physical_plan::{
        executors::{
            cache::CacheExec,
            drop_duplicates::DropDuplicatesExec,
            explode::ExplodeExec,
            filter::FilterExec,
            groupby::{GroupByExec, PartitionGroupByExec},
            join::JoinExec,
            melt::MeltExec,
            projection::ProjectionExec,
            scan::DataFrameExec,
            slice::SliceExec,
            sort::SortExec,
            stack::StackExec,
            udf::UdfExec,
        },
        expressions::{
            aggregation::{AggQuantileExpr, AggregationExpr},
            alias::AliasExpr,
            apply::ApplyExpr,
            cast::CastExpr,
            column::ColumnExpr,
            filter::FilterExpr,
            is_not_null::IsNotNullExpr,
            is_null::IsNullExpr,
            literal::LiteralExpr,
            not::NotExpr,
            slice::SliceExpr,
            sort::SortExpr,
            sortby::SortByExpr,
            take::TakeExpr,
            ternary::TernaryExpr,
            window::WindowExpr,
        },
    },
};
