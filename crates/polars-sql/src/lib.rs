//! Polars SQL
//! This crate provides a SQL interface for Polars DataFrames
#![deny(missing_docs)]
mod context;
pub mod function_registry;
mod functions;
pub mod keywords;
mod sql_expr;
mod table_functions;

pub use context::SQLContext;
pub use sql_expr::sql_expr;
