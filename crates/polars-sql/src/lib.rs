//! Polars SQL
//! This crate provides a SQL interface for Polars DataFrames
#![deny(missing_docs)]
mod context;
pub mod function_registry;
mod functions;
mod grouping_sets;
pub mod keywords;
mod literal_folding;
mod resolver;
mod sql_expr;
mod sql_visitors;
mod subquery;
mod table_functions;
mod types;

pub use context::{SQLContext, extract_table_identifiers};
pub use resolver::register_sql_resolver;
pub use sql_expr::sql_expr;

pub(crate) fn unique_column_name() -> polars_utils::pl_str::PlSmallStr {
    static COUNTER: polars_utils::relaxed_cell::RelaxedCell<u64> =
        polars_utils::relaxed_cell::RelaxedCell::new_u64(0);
    let idx = COUNTER.fetch_add(1);
    polars_utils::format_pl_smallstr!("_POLARS_TMP_SQL_{idx}")
}
