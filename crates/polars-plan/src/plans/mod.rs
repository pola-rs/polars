use std::sync::Arc;

use polars_core::prelude::*;

use crate::prelude::*;

pub(crate) mod aexpr;
pub(crate) mod anonymous_scan;
pub(crate) mod ir;

mod apply;
mod builder_ir;
pub(crate) mod conversion;
#[cfg(feature = "debugging")]
pub(crate) mod debug;
pub mod expr_ir;
mod functions;
pub mod hive;
pub(crate) mod iterator;
mod lit;
pub(crate) mod optimizer;
pub(crate) mod options;
#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "python")]
pub use python::*;
mod schema;
pub mod visitor;

pub use aexpr::*;
pub use anonymous_scan::*;
pub use apply::*;
pub use builder_ir::*;
pub use conversion::*;
pub(crate) use expr_ir::*;
pub use functions::*;
pub use ir::*;
pub use iterator::*;
pub use lit::*;
pub use optimizer::*;
pub use schema::*;

#[derive(Clone, Copy, Debug, Default)]
pub enum Context {
    /// Any operation that is done on groups
    Aggregation,
    /// Any operation that is done while projection/ selection of data
    #[default]
    Default,
}
