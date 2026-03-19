//! Polars SQL
//! This crate provides a SQL interface for Polars DataFrames
#![deny(missing_docs)]
mod context;
pub mod function_registry;
mod functions;
pub mod keywords;
mod sql_expr;
mod sql_visitors;
mod table_functions;
mod types;

pub use context::{SQLContext, extract_table_identifiers};
pub use sql_expr::sql_expr;
