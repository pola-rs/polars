use pyo3::prelude::*;
use polars_plan::prelude::{FileCount, FileScanOptions};
use crate::dataframe::PyDataFrame;
use super::*;

#[pyclass]
pub struct PlanPythonScan {
    #[pyo3(get)]
    options: PyObject,
    #[pyo3(get)]
    predicate: PyObject,
}

#[pyclass]
/// Slice the table
pub struct PlanSlice {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    offset: i64,
    #[pyo3(get)]
    len: IdxSize,
}

#[pyclass]
pub struct PlanSelection {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    predicate: PyObject,
}

#[pyclass]
#[derive(Clone)]
pub struct PyFileOptions {
    inner: FileScanOptions,
}

#[pymethods]
impl PyFileOptions {
    #[getter]
    fn n_rows(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .n_rows
            .map_or_else(|| py.None(), |n| n.into_py(py)))
    }
    #[getter]
    fn with_columns(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .with_columns
            .as_ref()
            .map_or_else(|| py.None(), |cols| cols.to_object(py)))
    }
    #[getter]
    fn cache(&self, _py: Python<'_>) -> PyResult<bool> {
        Ok(self.inner.cache)
    }
    #[getter]
    fn row_index(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .row_index
            .as_ref()
            .map_or_else(|| py.None(), |n| (&n.name, n.offset).to_object(py)))
    }
    #[getter]
    fn rechunk(&self, _py: Python<'_>) -> PyResult<bool> {
        Ok(self.inner.rechunk)
    }
    #[getter]
    fn file_counter(&self, _py: Python<'_>) -> PyResult<FileCount> {
        Ok(self.inner.file_counter)
    }
    #[getter]
    fn hive_partitioning(&self, _py: Python<'_>) -> PyResult<bool> {
        Ok(self.inner.hive_partitioning)
    }
}

#[pyclass]
pub struct PlanScan {
    #[pyo3(get)]
    paths: PyObject,
    #[pyo3(get)]
    file_info: PyObject,
    #[pyo3(get)]
    predicate: PyObject,
    #[pyo3(get)]
    output_schema: PyObject,
    #[pyo3(get)]
    file_options: PyFileOptions,
    #[pyo3(get)]
    scan_type: PyObject,
}

#[pyclass]
pub struct PlanDataFrameScan {
    #[pyo3(get)]
    df: PyDataFrame,
    #[pyo3(get)]
    schema: PyObject, // SchemaRef
    #[pyo3(get)]
    output_schema: PyObject, // Option<SchemaRef> Optional[dict]
    #[pyo3(get)]
    projection: PyObject,
    #[pyo3(get)]
    selection: PyObject,
}

#[pyclass]
/// Column selection
pub struct PlanProjection {
    #[pyo3(get)]
    expr: Vec<PyObject>,
    #[pyo3(get)]
    cse_expr: Vec<PyObject>,
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    schema: PyObject, // SchemaRef,
    options: (), //ProjectionOptions,
}

#[pyclass]
/// Sort the table
pub struct PlanSort {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    by_column: Vec<PyObject>,
    #[pyo3(get)]
    args: PyObject, // SortArguments,
}

/// Cache the input at this point in the LP
#[pyclass]
pub struct PlanCache {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    id_: usize,
    #[pyo3(get)]
    count: usize,
}

#[pyclass]
/// Groupby aggregation
pub struct PlanAggregate {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    keys: Vec<PyObject>,
    #[pyo3(get)]
    aggs: Vec<PyObject>,
    #[pyo3(get)]
    schema: PyObject, // SchemaRef,
    apply: (), // Option<Arc<dyn DataFrameUdf>>,
    #[pyo3(get)]
    maintain_order: bool,
    #[pyo3(get)]
    options: PyObject, // Arc<GroupbyOptions>,
}

#[pyclass]
/// Join operation
pub struct PlanJoin {
    #[pyo3(get)]
    input_left: PyObject,
    #[pyo3(get)]
    input_right: PyObject,
    #[pyo3(get)]
    schema: PyObject, // SchemaRef,
    #[pyo3(get)]
    left_on: Vec<PyObject>,
    #[pyo3(get)]
    right_on: Vec<PyObject>,
    #[pyo3(get)]
    options: PyObject, // Arc<JoinOptions>,
}

#[pyclass]
/// Adding columns to the table without a Join
pub struct PlanHStack {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    exprs: Vec<PyObject>,
    #[pyo3(get)]
    schema: PyObject, // SchemaRef,
    options: (), // ProjectionOptions,
}

#[pyclass]
/// Remove duplicates from the table
pub struct PlanDistinct {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    options: PyObject, // DistinctOptions,
}
#[pyclass]
/// A (User Defined) Function
pub struct PlanMapFunction {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    function: PyObject, // FunctionNode,
}
#[pyclass]
pub struct PlanUnion {
    #[pyo3(get)]
    inputs: Vec<PyObject>,
    #[pyo3(get)]
    options: PyObject, // UnionOptions,
}
#[pyclass]
/// Horizontal concatenation of multiple plans
pub struct PlanHConcat {
    #[pyo3(get)]
    inputs: Vec<PyObject>,
    #[pyo3(get)]
    schema: PyObject, // SchemaRef,
    options: (), // HConcatOptions,
}
#[pyclass]
/// This allows expressions to access other tables
pub struct PlanExtContext {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    contexts: Vec<PyObject>,
    #[pyo3(get)]
    schema: PyObject, // SchemaRef,
}

#[pyclass]
pub struct PlanSink {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    payload: PyObject, // SinkType,
}

// Expressions
#[pyclass]
pub struct ExprAlias {
    #[pyo3(get)]
    expr: PyObject,
    #[pyo3(get)]
    name: PyObject,
}

#[pymethods]
impl ExprAlias {
    fn reconstruct(&self, expr: PyObject) -> Self {
        Self {
            expr,
            name: self.name.clone(),
        }
    }
}

#[pyclass]
pub struct ExprColumn {
    #[pyo3(get)]
    name: PyObject,
}

#[pymethods]
impl ExprColumn {
    fn reconstruct(&self, name: PyObject) -> Self {
        Self { name }
    }
    #[new]
    fn new(name: PyObject) -> Self {
        Self { name }
    }
}

#[pyclass]
pub struct ExprLiteral {
    #[pyo3(get)]
    value: PyObject,
    #[pyo3(get)]
    dtype: PyObject,
}

impl ExprLiteral {
    fn new(py: Python<'_>, value: PyObject, dtype: PyObject) -> Self {
        Self {
            value,
            dtype: dtype,
        }
    }
}

#[pymethods]
impl ExprLiteral {
    fn reconstruct(&self, value: PyObject, dtype: PyObject) -> Self {
        Self { value, dtype }
    }
}

#[pyclass]
pub struct ExprBinaryExpr {
    #[pyo3(get)]
    left: PyObject,
    #[pyo3(get)]
    op: PyOperator,
    #[pyo3(get)]
    right: PyObject,
}

#[pymethods]
impl ExprBinaryExpr {
    fn reconstruct(&self, left: PyObject, right: PyObject) -> Self {
        Self {
            left,
            op: self.op,
            right,
        }
    }
}

#[pyclass]
pub struct ExprCast {
    #[pyo3(get)]
    expr: PyObject,
    #[pyo3(get)]
    dtype: PyObject,
}

impl ExprCast {
    fn new(py: Python<'_>, expr: PyObject, dtype: PyObject) -> Self {
        Self { expr, dtype }
    }
}

#[pymethods]
impl ExprCast {
    fn reconstruct(&self, expr: PyObject, dtype: PyObject) -> Self {
        Self { expr, dtype }
    }
}

#[pyclass]
pub struct ExprSort {
    #[pyo3(get)]
    expr: PyObject,
    #[pyo3(get)]
    options: PyObject,
}

#[pymethods]
impl ExprSort {
    fn reconstruct(&self, expr: PyObject) -> Self {
        Self {
            expr,
            options: self.options.clone(),
        }
    }
}

#[pyclass]
pub struct ExprGather {
    #[pyo3(get)]
    expr: PyObject,
    #[pyo3(get)]
    idx: PyObject,
    #[pyo3(get)]
    scalar: bool,
}

#[pymethods]
impl ExprGather {
    fn reconstruct(&self, expr: PyObject, idx: PyObject, scalar: bool) -> Self {
        Self { expr, idx, scalar }
    }
}

#[pyclass]
pub struct ExprFilter {
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    by: PyObject,
}

#[pymethods]
impl ExprFilter {
    fn reconstruct(&self, input: PyObject, by: PyObject) -> Self {
        Self { input, by }
    }
}

#[pyclass]
pub struct ExprSortBy {
    #[pyo3(get)]
    expr: PyObject,
    #[pyo3(get)]
    by: Vec<PyObject>,
    #[pyo3(get)]
    descending: PyObject,
}

#[pymethods]
impl ExprSortBy {
    fn reconstruct(&self, expr: PyObject, by: Vec<PyObject>) -> Self {
        Self {
            expr,
            by,
            descending: self.descending.clone(),
        }
    }
}

#[pyclass]
pub struct ExprAgg {
    #[pyo3(get)]
    name: PyObject,
    #[pyo3(get)]
    arguments: PyObject,
    #[pyo3(get)]
    // Arbitrary control options
    options: PyObject,
}

#[pymethods]
impl ExprAgg {
    fn reconstruct(&self, arguments: PyObject) -> Self {
        Self {
            name: self.name.clone(),
            arguments,
            options: self.options.clone(),
        }
    }
}

#[pyclass]
pub struct ExprTernary {
    #[pyo3(get)]
    predicate: PyObject,
    #[pyo3(get)]
    truthy: PyObject,
    #[pyo3(get)]
    falsy: PyObject,
}

#[pymethods]
impl ExprTernary {
    fn reconstruct(&self, predicate: PyObject, truthy: PyObject, falsy: PyObject) -> Self {
        Self {
            predicate,
            truthy,
            falsy,
        }
    }
}

#[pyclass]
pub struct ExprFunction {
    #[pyo3(get)]
    input: Vec<PyObject>,
    #[pyo3(get)]
    function_data: PyObject,
    #[pyo3(get)]
    options: PyObject,
}

#[pymethods]
impl ExprFunction {
    fn reconstruct(&self, input: Vec<PyObject>) -> Self {
        Self {
            input,
            function_data: self.function_data.clone(),
            options: self.options.clone(),
        }
    }
}

#[pyclass]
pub struct ExprCount {}

#[pyclass]
pub struct ExprWindow {
    #[pyo3(get)]
    function: PyObject,
    #[pyo3(get)]
    partition_by: PyObject,
    #[pyo3(get)]
    options: PyObject,
}