mod ir;
mod metrics;
mod physical;

pub use ir::*;
pub use metrics::*;
pub use physical::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SortColumnDescription {
    pub expr: String,
    pub descending: bool,
    pub nulls_last: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PredicateFileSkipDescription {
    pub no_residual_predicate: bool,
    pub original_len: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, strum_macros::IntoStaticStr)]
pub enum PythonPredicateDescription {
    #[default]
    None,
    PyArrow {
        predicate: String,
        has_residual: bool,
    },
    Polars {
        predicate: String,
    },
}
