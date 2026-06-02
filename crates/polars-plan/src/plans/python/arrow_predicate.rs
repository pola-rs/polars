use polars_utils::python_function::PythonObject;
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use crate::plans::ExprIR;

// Predicate handed to a pyarrow python scan.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct ArrowPredicate {
    pub predicate: ExprIR,
    pub pyarrow_predicate: PythonObject,
    /// Set when only some minterms of the original predicate could be converted
    /// to pyarrow; the engine then post-applies the original predicate after
    /// the scan.
    pub has_residual: bool,
}

impl PartialEq for ArrowPredicate {
    fn eq(&self, other: &Self) -> bool {
        self.predicate == other.predicate && self.has_residual == other.has_residual
    }
}
