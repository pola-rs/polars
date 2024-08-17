use polars_core::prelude::{IdxSize, UniqueKeepStrategy};
use polars_ops::prelude::JoinType;
use polars_plan::plans::IR;
use polars_plan::prelude::{
    FileCount, FileScan, FileScanOptions, FunctionIR, PythonPredicate, PythonScanSource,
};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use super::expr_nodes::PyGroupbyOptions;
use crate::lazyframe::visit::PyExprIR;
use crate::PyDataFrame;

#[pyclass]
/// Scan a table with an optional predicate from a python function
pub struct PythonScan {
    #[pyo3(get)]
    options: PyObject,
}

#[pyclass]
/// Slice the table
pub struct Slice {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    offset: i64,
    #[pyo3(get)]
    len: IdxSize,
}

#[pyclass]
/// Filter the table with a boolean expression
pub struct Filter {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    predicate: PyExprIR,
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
            .slice
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
            .map_or_else(|| py.None(), |n| (n.name.as_ref(), n.offset).to_object(py)))
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
    fn hive_options(&self, _py: Python<'_>) -> PyResult<PyObject> {
        Err(PyNotImplementedError::new_err("hive options"))
    }
}

#[pyclass]
/// Scan a table from file
pub struct Scan {
    #[pyo3(get)]
    paths: PyObject,
    #[pyo3(get)]
    file_info: PyObject,
    #[pyo3(get)]
    predicate: Option<PyExprIR>,
    #[pyo3(get)]
    file_options: PyFileOptions,
    #[pyo3(get)]
    scan_type: PyObject,
}

#[pyclass]
/// Scan a table from an existing dataframe
pub struct DataFrameScan {
    #[pyo3(get)]
    df: PyDataFrame,
    #[pyo3(get)]
    projection: PyObject,
    #[pyo3(get)]
    selection: Option<PyExprIR>,
}

#[pyclass]
/// Project out columns from a table
pub struct SimpleProjection {
    #[pyo3(get)]
    input: usize,
}

#[pyclass]
/// Column selection
pub struct Select {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    expr: Vec<PyExprIR>,
    #[pyo3(get)]
    should_broadcast: bool,
}

#[pyclass]
/// Sort the table
pub struct Sort {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    by_column: Vec<PyExprIR>,
    #[pyo3(get)]
    sort_options: (bool, Vec<bool>, Vec<bool>),
    #[pyo3(get)]
    slice: Option<(i64, usize)>,
}

#[pyclass]
/// Cache the input at this point in the LP
pub struct Cache {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    id_: usize,
    #[pyo3(get)]
    cache_hits: u32,
}

#[pyclass]
/// Groupby aggregation
pub struct GroupBy {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    keys: Vec<PyExprIR>,
    #[pyo3(get)]
    aggs: Vec<PyExprIR>,
    #[pyo3(get)]
    apply: (),
    #[pyo3(get)]
    maintain_order: bool,
    #[pyo3(get)]
    options: PyObject,
}

#[pyclass]
/// Join operation
pub struct Join {
    #[pyo3(get)]
    input_left: usize,
    #[pyo3(get)]
    input_right: usize,
    #[pyo3(get)]
    left_on: Vec<PyExprIR>,
    #[pyo3(get)]
    right_on: Vec<PyExprIR>,
    #[pyo3(get)]
    options: PyObject,
}

#[pyclass]
/// Adding columns to the table without a Join
pub struct HStack {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    exprs: Vec<PyExprIR>,
    #[pyo3(get)]
    should_broadcast: bool,
}

#[pyclass]
/// Like Select, but all operations produce a single row.
pub struct Reduce {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    exprs: Vec<PyExprIR>,
}

#[pyclass]
/// Remove duplicates from the table
pub struct Distinct {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    options: PyObject,
}
#[pyclass]
/// A (User Defined) Function
pub struct MapFunction {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    function: PyObject,
}
#[pyclass]
pub struct Union {
    #[pyo3(get)]
    inputs: Vec<usize>,
    #[pyo3(get)]
    options: Option<(i64, usize)>,
}
#[pyclass]
/// Horizontal concatenation of multiple plans
pub struct HConcat {
    #[pyo3(get)]
    inputs: Vec<usize>,
    #[pyo3(get)]
    options: (),
}
#[pyclass]
/// This allows expressions to access other tables
pub struct ExtContext {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    contexts: Vec<usize>,
}

#[pyclass]
pub struct Sink {
    #[pyo3(get)]
    input: usize,
    #[pyo3(get)]
    payload: PyObject,
}

pub(crate) fn into_py(py: Python<'_>, plan: &IR) -> PyResult<PyObject> {
    let result = match plan {
        IR::PythonScan { options } => {
            let python_src = match options.python_source {
                PythonScanSource::Pyarrow => "pyarrow",
                PythonScanSource::Cuda => "cuda",
                PythonScanSource::IOPlugin => "io_plugin",
            };

            PythonScan {
                options: (
                    options
                        .scan_fn
                        .as_ref()
                        .map_or_else(|| py.None(), |s| s.0.clone()),
                    options
                        .with_columns
                        .as_ref()
                        .map_or_else(|| py.None(), |cols| cols.to_object(py)),
                    python_src,
                    match &options.predicate {
                        PythonPredicate::None => py.None(),
                        PythonPredicate::PyArrow(s) => ("pyarrow", s).to_object(py),
                        PythonPredicate::Polars(e) => ("polars", e.node().0).to_object(py),
                    },
                    options
                        .n_rows
                        .map_or_else(|| py.None(), |s| s.to_object(py)),
                )
                    .to_object(py),
            }
            .into_py(py)
        },
        IR::Slice { input, offset, len } => Slice {
            input: input.0,
            offset: *offset,
            len: *len,
        }
        .into_py(py),
        IR::Filter { input, predicate } => Filter {
            input: input.0,
            predicate: predicate.into(),
        }
        .into_py(py),
        IR::Scan {
            hive_parts: Some(_),
            ..
        } => {
            return Err(PyNotImplementedError::new_err(
                "scan with hive partitioning",
            ))
        },
        IR::Scan {
            paths,
            file_info: _,
            hive_parts: _,
            predicate,
            output_schema: _,
            scan_type,
            file_options,
        } => Scan {
            paths: paths.to_object(py),
            // TODO: file info
            file_info: py.None(),
            predicate: predicate.as_ref().map(|e| e.into()),
            file_options: PyFileOptions {
                inner: file_options.clone(),
            },
            scan_type: match scan_type {
                #[cfg(feature = "csv")]
                FileScan::Csv {
                    options,
                    cloud_options,
                } => {
                    // Since these options structs are serializable,
                    // we just use the serde json representation
                    let options = serde_json::to_string(options)
                        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
                    let cloud_options = serde_json::to_string(cloud_options)
                        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
                    ("csv", options, cloud_options).into_py(py)
                },
                #[cfg(feature = "parquet")]
                FileScan::Parquet {
                    options,
                    cloud_options,
                    ..
                } => {
                    let options = serde_json::to_string(options)
                        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
                    let cloud_options = serde_json::to_string(cloud_options)
                        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
                    ("parquet", options, cloud_options).into_py(py)
                },
                #[cfg(feature = "ipc")]
                FileScan::Ipc { .. } => return Err(PyNotImplementedError::new_err("ipc scan")),
                #[cfg(feature = "json")]
                FileScan::NDJson { options, .. } => {
                    // TODO: Also pass cloud_options
                    let options = serde_json::to_string(options)
                        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
                    ("ndjson", options).into_py(py)
                },
                FileScan::Anonymous { .. } => {
                    return Err(PyNotImplementedError::new_err("anonymous scan"))
                },
            },
        }
        .into_py(py),
        IR::DataFrameScan {
            df,
            schema: _,
            output_schema,
            filter: selection,
        } => DataFrameScan {
            df: PyDataFrame::new((**df).clone()),
            projection: output_schema.as_ref().map_or_else(
                || py.None(),
                |s| {
                    s.iter_names()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .to_object(py)
                },
            ),
            selection: selection.as_ref().map(|e| e.into()),
        }
        .into_py(py),
        IR::SimpleProjection { input, columns: _ } => {
            SimpleProjection { input: input.0 }.into_py(py)
        },
        IR::Select {
            input,
            expr,
            schema: _,
            options,
        } => Select {
            expr: expr.iter().map(|e| e.into()).collect(),
            input: input.0,
            should_broadcast: options.should_broadcast,
        }
        .into_py(py),
        IR::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => Sort {
            input: input.0,
            by_column: by_column.iter().map(|e| e.into()).collect(),
            sort_options: (
                sort_options.maintain_order,
                sort_options.nulls_last.clone(),
                sort_options.descending.clone(),
            ),
            slice: *slice,
        }
        .into_py(py),
        IR::Cache {
            input,
            id,
            cache_hits,
        } => Cache {
            input: input.0,
            id_: *id,
            cache_hits: *cache_hits,
        }
        .into_py(py),
        IR::GroupBy {
            input,
            keys,
            aggs,
            schema: _,
            apply,
            maintain_order,
            options,
        } => GroupBy {
            input: input.0,
            keys: keys.iter().map(|e| e.into()).collect(),
            aggs: aggs.iter().map(|e| e.into()).collect(),
            apply: apply.as_ref().map_or(Ok(()), |_| {
                Err(PyNotImplementedError::new_err(format!(
                    "apply inside GroupBy {:?}",
                    plan
                )))
            })?,
            maintain_order: *maintain_order,
            // TODO: dynamic options
            options: PyGroupbyOptions::new(options.as_ref().clone()).into_py(py),
        }
        .into_py(py),
        IR::Join {
            input_left,
            input_right,
            schema: _,
            left_on,
            right_on,
            options,
        } => Join {
            input_left: input_left.0,
            input_right: input_right.0,
            left_on: left_on.iter().map(|e| e.into()).collect(),
            right_on: right_on.iter().map(|e| e.into()).collect(),
            options: (
                match options.args.how {
                    JoinType::Left => "left",
                    JoinType::Right => "right",
                    JoinType::Inner => "inner",
                    JoinType::Full => "full",
                    #[cfg(feature = "asof_join")]
                    JoinType::AsOf(_) => return Err(PyNotImplementedError::new_err("asof join")),
                    JoinType::Cross => "cross",
                    JoinType::Semi => "leftsemi",
                    JoinType::Anti => "leftanti",
                },
                options.args.join_nulls,
                options.args.slice,
                options.args.suffix.clone(),
                options.args.coalesce.coalesce(&options.args.how),
            )
                .to_object(py),
        }
        .into_py(py),
        IR::HStack {
            input,
            exprs,
            schema: _,
            options,
        } => HStack {
            input: input.0,
            exprs: exprs.iter().map(|e| e.into()).collect(),
            should_broadcast: options.should_broadcast,
        }
        .into_py(py),
        IR::Reduce {
            input,
            exprs,
            schema: _,
        } => Reduce {
            input: input.0,
            exprs: exprs.iter().map(|e| e.into()).collect(),
        }
        .into_py(py),
        IR::Distinct { input, options } => Distinct {
            input: input.0,
            options: (
                match options.keep_strategy {
                    UniqueKeepStrategy::First => "first",
                    UniqueKeepStrategy::Last => "last",
                    UniqueKeepStrategy::None => "none",
                    UniqueKeepStrategy::Any => "any",
                },
                options.subset.as_ref().map_or_else(
                    || py.None(),
                    |f| {
                        f.iter()
                            .map(|s| s.as_ref())
                            .collect::<Vec<&str>>()
                            .to_object(py)
                    },
                ),
                options.maintain_order,
                options.slice,
            )
                .to_object(py),
        }
        .into_py(py),
        IR::MapFunction { input, function } => MapFunction {
            input: input.0,
            function: match function {
                FunctionIR::OpaquePython(_) => {
                    return Err(PyNotImplementedError::new_err("opaque python mapfunction"))
                },
                FunctionIR::Opaque {
                    function: _,
                    schema: _,
                    predicate_pd: _,
                    projection_pd: _,
                    streamable: _,
                    fmt_str: _,
                } => return Err(PyNotImplementedError::new_err("opaque rust mapfunction")),
                FunctionIR::Pipeline {
                    function: _,
                    schema: _,
                    original: _,
                } => return Err(PyNotImplementedError::new_err("pipeline mapfunction")),
                FunctionIR::Unnest { columns } => (
                    "unnest",
                    columns.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                )
                    .to_object(py),
                FunctionIR::Rechunk => ("rechunk",).to_object(py),
                #[cfg(feature = "merge_sorted")]
                FunctionIR::MergeSorted { column } => {
                    ("merge_sorted", column.to_string()).to_object(py)
                },
                FunctionIR::Rename {
                    existing,
                    new,
                    swapping,
                    schema: _,
                } => (
                    "rename",
                    existing.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    new.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    *swapping,
                )
                    .to_object(py),
                FunctionIR::Explode { columns, schema: _ } => (
                    "explode",
                    columns.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                )
                    .to_object(py),
                FunctionIR::Unpivot { args, schema: _ } => (
                    "unpivot",
                    args.index.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    args.on.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    args.variable_name
                        .as_ref()
                        .map_or_else(|| py.None(), |s| s.as_str().to_object(py)),
                    args.value_name
                        .as_ref()
                        .map_or_else(|| py.None(), |s| s.as_str().to_object(py)),
                )
                    .to_object(py),
                FunctionIR::RowIndex {
                    name,
                    schema: _,
                    offset,
                } => ("row_index", name.to_string(), offset.unwrap_or(0)).to_object(py),
                FunctionIR::FastCount {
                    paths: _,
                    scan_type: _,
                    alias: _,
                } => return Err(PyNotImplementedError::new_err("function count")),
            },
        }
        .into_py(py),
        IR::Union { inputs, options } => Union {
            inputs: inputs.iter().map(|n| n.0).collect(),
            // TODO: rest of options
            options: options.slice,
        }
        .into_py(py),
        IR::HConcat {
            inputs,
            schema: _,
            options: _,
        } => HConcat {
            inputs: inputs.iter().map(|n| n.0).collect(),
            options: (),
        }
        .into_py(py),
        IR::ExtContext {
            input,
            contexts,
            schema: _,
        } => ExtContext {
            input: input.0,
            contexts: contexts.iter().map(|n| n.0).collect(),
        }
        .into_py(py),
        IR::Sink {
            input: _,
            payload: _,
        } => {
            return Err(PyNotImplementedError::new_err(
                "Not expecting to see a Sink node",
            ))
        },
        IR::Invalid => return Err(PyNotImplementedError::new_err("Invalid")),
    };
    Ok(result)
}
