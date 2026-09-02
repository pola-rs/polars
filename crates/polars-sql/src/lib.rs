//! Polars SQL
//! This crate provides a SQL interface for Polars DataFrames
#![deny(missing_docs)]
mod context;
pub mod function_registry;
mod functions;
pub mod keywords;
mod resolver;
mod sql_expr;
mod sql_visitors;
mod subquery;
mod table_functions;
mod types;

pub use context::{SQLContext, extract_table_identifiers};
pub use resolver::register_sql_resolver;
pub use sql_expr::sql_expr;
