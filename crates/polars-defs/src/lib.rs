//! Shared option, argument and enum definitions used by both the Polars query planner
//! (`polars-plan`) and the compute crates (`polars-ops`, `polars-time`, `polars-stream`, etc).
//!
//! This crate holds plain data types only: no kernels, no execution. It exists so that
//! `polars-plan` does not have to depend on the compute crates to name the options it
//! stores in the DSL and IR.
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod expr;
pub mod join;
pub mod time;
