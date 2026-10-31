//! Polars SQL
//! This crate provides a SQL interface for Polars DataFrames
#![deny(missing_docs)]
mod catalog;
mod context;
mod dyn_visit;
pub mod function_registry;
mod functions;
mod group_context;
mod grouping_sets;
mod join_keys;
pub mod keywords;
mod lateral;
mod literal_folding;
mod resolver;
mod sql_expr;
mod sql_visitors;
mod subquery;
mod subquery_group_by;
mod subquery_outer_select;
mod subquery_pull_up;
#[cfg(any(
    feature = "csv",
    feature = "json",
    feature = "parquet",
    feature = "ipc"
))]
mod table_function_args;
mod table_functions;
mod types;
mod window_frames;

pub use catalog::{
    CatalogAlter, CatalogNamespace, CatalogPartitionField, CatalogStatement, CatalogTable,
};
pub use context::{SQLContext, extract_table_identifiers};
pub use resolver::register_sql_resolver;
pub use sql_expr::sql_expr;

polars_utils::define_unique_column_name!("SQL");

/// The column `name`. Unlike the DSL `col`, a name such as `*` or `^a$` is read as the column
/// of that name, not as a selector: a SQL identifier only names a column.
pub(crate) fn col(name: impl Into<polars_utils::pl_str::PlSmallStr>) -> polars_plan::dsl::Expr {
    polars_plan::dsl::Expr::Column(name.into())
}

/// The field `name` of the struct `expr`, read by name like [`col`].
pub(crate) fn struct_field(
    expr: polars_plan::dsl::Expr,
    name: impl Into<polars_utils::pl_str::PlSmallStr>,
) -> polars_plan::dsl::Expr {
    use polars_plan::dsl::FunctionExpr;
    use polars_plan::dsl::function_expr::StructFunction;
    expr.map_unary(FunctionExpr::StructExpr(StructFunction::FieldByName(
        name.into(),
    )))
}
