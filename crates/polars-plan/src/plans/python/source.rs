use std::sync::Arc;

use polars_core::schema::SchemaRef;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::plans::{ExprIR, PlSmallStr};
use crate::prelude::python_udf::PythonFunction;

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PythonOptions {
    /// A function that returns a Python Generator.
    /// The generator should produce Polars DataFrame's.
    pub scan_fn: Option<PythonFunction>,
    /// Schema of the file.
    pub schema: SchemaRef,
    /// Schema the reader will produce when the file is read.
    pub output_schema: Option<SchemaRef>,
    // Projected column names.
    pub with_columns: Option<Arc<[PlSmallStr]>>,
    // Which interface is the python function.
    pub python_source: PythonScanSource,
    /// A `head` call passed to the reader.
    pub n_rows: Option<usize>,
    /// Optional predicate the reader must apply.
    pub predicate: PythonPredicate,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PythonScanSource {
    Pyarrow,
    Cuda,
    #[default]
    IOPlugin,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PythonPredicate {
    // A pyarrow predicate python expression
    // can be evaluated with python.eval
    PyArrow(String),
    Polars(ExprIR),
    #[default]
    None,
}
