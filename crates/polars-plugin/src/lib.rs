//! Runtime and macro support for Polars expression plugins.
//!
//! Use [`polars_expr`] and [`prelude`] when authoring a plugin `cdylib`.
//! This crate does not require PyO3; keyword arguments retain the existing
//! Python-pickle encoding used by the Polars plugin ABI.

#[doc(hidden)]
pub mod derive;
#[doc(hidden)]
pub mod export;
pub mod prelude;

pub use polars_plugin_derive::polars_expr;
