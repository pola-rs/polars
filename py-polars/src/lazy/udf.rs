//! UDF (User-Defined Functions) definitions for use in polars_lazy Expr.

use polars::prelude::*;
use polars_core::utils::get_supertype;
use polars_lazy::dsl::{ApplyOptions, FunctionOptions};
use polars_lazy::udf_registry::ErasedSerialize;
use pyo3::prelude::*;
use pyo3::types::PyFloat;
use serde::{Deserialize, Serialize};

use super::utils::py_exprs_to_exprs;
use super::*;
use crate::conversion::Wrap;
use crate::lazy::dsl::PyExpr;
use crate::py_modules::POLARS;
use crate::series::PySeries;

pub(crate) struct PyLambda(pub(crate) PyObject);

/// This enum describes how a python UDF should be called.
#[cfg_attr(feature = "json", derive(Serialize, Deserialize))]
enum UdfLambdaOp {
    // fn(a) -> r, flat
    MapSingle {
        agg_list: bool,
    },
    // fn(...args) -> r, flat or groups
    MapMultiple {
        apply_groups: bool,
        returns_scalar: bool,
    },
    // fn(a, b) -> r, groups
    Fold,
    Reduce,
    CumFold {
        include_init: bool,
    },
    CumReduce,
}

#[cfg_attr(feature = "json", derive(Serialize, Deserialize))]
pub(crate) struct PyUdfLambda {
    lambda: PyLambda,
    output_type: Option<DataType>,
    op: UdfLambdaOp,
}

impl polars_lazy::dsl::SeriesUdf for PyUdfLambda {
    #[cfg(feature = "json")]
    fn as_serialize(&self) -> Option<(&str, &dyn ErasedSerialize)> {
        Some(("py-lambda", self))
    }
    fn call_udf(&self, series: &mut [Series]) -> PolarsResult<Series> {
        let output_type = self.output_type.as_ref().unwrap_or(&DataType::Unknown);
        Python::with_gil(|py| {
            let res = match &self.op {
                UdfLambdaOp::MapSingle { .. } => {
                    let series = &series[0];

                    call_lambda_series_unary(py, series.clone(), &self.lambda.0, &POLARS)
                        .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?
                        .to_series(py, &POLARS, series.name())
                }
                UdfLambdaOp::MapMultiple { apply_groups, .. } => {
                    let out = call_lambda_series_slice(py, series, &self.lambda.0, &POLARS);

                    // we return an error, because that will become a null value polars lazy apply list
                    if *apply_groups && out.is_none(py) {
                        return Err(PolarsError::NoData("".into()));
                    }

                    out.to_series(py, &POLARS, "")
                }
                UdfLambdaOp::Fold => {
                    let mut series = series.to_vec();
                    let mut acc = series.pop().unwrap(); // last argument is the accumulator

                    for s in series {
                        acc = call_lambda_series_binary(py, acc, s, &self.lambda.0, &POLARS)
                            .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?
                            .to_series(py, &POLARS, "")
                    }
                    acc
                }
                UdfLambdaOp::Reduce => {
                    let mut s = series.to_vec();
                    let mut s_iter = s.drain(..);

                    match s_iter.next() {
                        Some(mut acc) => {
                            for s in s_iter {
                                acc = call_lambda_series_binary(py, acc, s, &self.lambda.0, &POLARS)
                                    .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?
                                    .to_series(py, &POLARS, "")
                            }
                            acc
                        }
                        None => {
                            return Err(PolarsError::ComputeError(
                                "Reduce did not have any expressions to fold".into(),
                            ))
                        }
                    }
                }
                UdfLambdaOp::CumFold { include_init } => {
                    let mut series = series.to_vec();
                    let mut acc = series.pop().unwrap();

                    let mut result = vec![];
                    if *include_init {
                        result.push(acc.clone())
                    }

                    for s in series {
                        let name = s.name().to_string();
                        acc = call_lambda_series_binary(py, acc, s, &self.lambda.0, &POLARS)
                            .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?
                            .to_series(py, &POLARS, &name);
                        acc.rename(&name);
                        result.push(acc.clone());
                    }

                    StructChunked::new(acc.name(), &result).map(|ca| ca.into_series())?
                }
                UdfLambdaOp::CumReduce => {
                    let mut s = series.to_vec();
                    let mut s_iter = s.drain(..);

                    match s_iter.next() {
                        Some(mut acc) => {
                            let mut result = vec![acc.clone()];

                            for s in s_iter {
                                let name = s.name().to_string();
                                acc =
                                    call_lambda_series_binary(py, acc, s, &self.lambda.0, &POLARS)
                                        .map_err(|e| {
                                            PolarsError::ComputeError(format!("{e}").into())
                                        })?
                                        .to_series(py, &POLARS, &name);
                                result.push(acc.clone());
                            }

                            StructChunked::new(acc.name(), &result).map(|ca| ca.into_series())?
                        }
                        None => {
                            return Err(PolarsError::ComputeError(
                                "Reduce did not have any expressions to fold".into(),
                            ))
                        }
                    }
                }
            };

            if !matches!(output_type, DataType::Unknown) && res.dtype() != output_type {
                Err(PolarsError::SchemaMisMatch(
                    format!("Expected output type: '{:?}', but got '{:?}'. Set 'return_dtype' to the proper datatype.", output_type, res.dtype()).into()))
            } else {
                Ok(res)
            }
        })
    }
}

impl polars_lazy::dsl::FunctionOutputField for PyUdfLambda {
    #[cfg(feature = "json")]
    fn as_serialize(&self) -> Option<(&str, &dyn ErasedSerialize)> {
        Some(("py-lambda", self))
    }
    fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: polars_lazy::dsl::Context,
        fields: &[Field],
    ) -> Field {
        match &self.op {
            UdfLambdaOp::MapSingle { .. } => {
                let fld = &fields[0];
                match self.output_type {
                    Some(ref dt) => Field::new(fld.name(), dt.clone()),
                    None => {
                        let mut fld = fld.clone();
                        fld.coerce(DataType::Unknown);
                        fld
                    }
                }
            }
            UdfLambdaOp::MapMultiple { .. } => {
                let fld = &fields[0];
                match self.output_type {
                    Some(ref dt) => Field::new(fld.name(), dt.clone()),
                    None => fld.clone(),
                }
            }
            UdfLambdaOp::Fold | UdfLambdaOp::Reduce => {
                // super_type
                let mut fld = fields[0].clone();
                let mut new_type = fields[0].data_type().clone();
                for f in &fields[1..] {
                    new_type = get_supertype(&new_type, f.data_type()).unwrap();
                }
                fld.coerce(new_type);
                fld
            }
            UdfLambdaOp::CumFold { .. } | UdfLambdaOp::CumReduce => {
                let mut st = fields[0].dtype.clone();
                for fld in &fields[1..] {
                    st = get_supertype(&st, &fld.dtype).unwrap();
                }
                Field::new(
                    &fields[0].name,
                    DataType::Struct(
                        fields
                            .iter()
                            .map(|fld| Field::new(fld.name(), st.clone()))
                            .collect(),
                    ),
                )
            }
        }
    }
}

impl PyUdfLambda {
    fn into_expr(self, inputs: Vec<Expr>) -> Expr {
        let opt = match self.op {
            UdfLambdaOp::MapSingle { agg_list: false } => FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                fmt_str: "map",
                ..Default::default()
            },
            UdfLambdaOp::MapSingle { agg_list: true } => FunctionOptions {
                collect_groups: ApplyOptions::ApplyList,
                fmt_str: "map_list",
                ..Default::default()
            },
            UdfLambdaOp::MapMultiple {
                apply_groups: true,
                returns_scalar,
            } => FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                auto_explode: returns_scalar,
                fmt_str: "apply_multiple",
                ..Default::default()
            },
            UdfLambdaOp::MapMultiple {
                apply_groups: false,
                ..
            } => FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                fmt_str: "map_multiple",
                ..Default::default()
            },
            UdfLambdaOp::Fold => FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: true,
                auto_explode: true,
                fmt_str: "fold",
                ..Default::default()
            },
            UdfLambdaOp::Reduce => FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: true,
                auto_explode: true,
                fmt_str: "reduce",
                ..Default::default()
            },
            UdfLambdaOp::CumFold { .. } => FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: true,
                auto_explode: true,
                fmt_str: "cumfold",
                ..Default::default()
            },
            UdfLambdaOp::CumReduce => FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: true,
                auto_explode: true,
                fmt_str: "cumreduce",
                ..Default::default()
            },
        };

        make_udf_expr(inputs, Arc::new(self), opt)
    }
}

#[pymethods]
impl PyExpr {
    // todo: make this serializable
    pub fn rolling_apply(
        &self,
        py: Python,
        lambda: PyObject,
        window_size: usize,
        weights: Option<Vec<f64>>,
        min_periods: usize,
        center: bool,
    ) -> PyExpr {
        let options = RollingOptionsFixedWindow {
            window_size,
            weights,
            min_periods,
            center,
        };
        // get the pypolars module
        // do the import outside of the function.
        let pypolars = PyModule::import(py, "polars").unwrap().to_object(py);

        let function = move |s: &Series| {
            Python::with_gil(|py| {
                let out = call_lambda_series_unary(py, s.clone(), &lambda, &pypolars)
                    .expect("python function failed");
                match out.getattr(py, "_s") {
                    Ok(pyseries) => {
                        let pyseries = pyseries.extract::<PySeries>(py).unwrap();
                        pyseries.series
                    }
                    Err(_) => {
                        let obj = out;
                        let is_float = obj.as_ref(py).is_instance_of::<PyFloat>().unwrap();

                        let dtype = s.dtype();

                        use DataType::*;
                        let result = match dtype {
                            UInt8 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(UInt8Chunked::from_slice("", &[v as u8]).into_series())
                                } else {
                                    obj.extract::<u8>(py)
                                        .map(|v| UInt8Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            UInt16 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(UInt16Chunked::from_slice("", &[v as u16]).into_series())
                                } else {
                                    obj.extract::<u16>(py)
                                        .map(|v| UInt16Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            UInt32 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(UInt32Chunked::from_slice("", &[v as u32]).into_series())
                                } else {
                                    obj.extract::<u32>(py)
                                        .map(|v| UInt32Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            UInt64 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(UInt64Chunked::from_slice("", &[v as u64]).into_series())
                                } else {
                                    obj.extract::<u64>(py)
                                        .map(|v| UInt64Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            Int8 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int8Chunked::from_slice("", &[v as i8]).into_series())
                                } else {
                                    obj.extract::<i8>(py)
                                        .map(|v| Int8Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            Int16 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int16Chunked::from_slice("", &[v as i16]).into_series())
                                } else {
                                    obj.extract::<i16>(py)
                                        .map(|v| Int16Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            Int32 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int32Chunked::from_slice("", &[v as i32]).into_series())
                                } else {
                                    obj.extract::<i32>(py)
                                        .map(|v| Int32Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            Int64 => {
                                if is_float {
                                    let v = obj.extract::<f64>(py).unwrap();
                                    Ok(Int64Chunked::from_slice("", &[v as i64]).into_series())
                                } else {
                                    obj.extract::<i64>(py)
                                        .map(|v| Int64Chunked::from_slice("", &[v]).into_series())
                                }
                            }
                            Float32 => obj
                                .extract::<f32>(py)
                                .map(|v| Float32Chunked::from_slice("", &[v]).into_series()),
                            Float64 => obj
                                .extract::<f64>(py)
                                .map(|v| Float64Chunked::from_slice("", &[v]).into_series()),
                            dt => panic!("{dt:?} not implemented"),
                        };

                        match result {
                            Ok(s) => s,
                            Err(e) => {
                                panic!("{e:?}")
                            }
                        }
                    }
                }
            })
        };
        self.clone()
            .inner
            .rolling_apply(Arc::new(function), GetOutput::same_type(), options)
            .with_fmt("rolling_apply")
            .into()
    }

    pub fn map(
        &self,
        lambda: PyObject,
        output_type: Option<Wrap<DataType>>,
        agg_list: bool,
    ) -> PyExpr {
        PyUdfLambda {
            lambda: PyLambda(lambda),
            op: UdfLambdaOp::MapSingle { agg_list },
            output_type: output_type.map(|wrap| wrap.0),
        }
        .into_expr(vec![self.inner.clone()])
        .into()
    }

    // todo: make this serializable
    pub fn map_alias(&self, lambda: PyObject) -> PyExpr {
        self.inner
            .clone()
            .map_alias(move |name| {
                let out = Python::with_gil(|py| lambda.call1(py, (name,)));
                match out {
                    Ok(out) => Ok(out.to_string()),
                    Err(e) => Err(PolarsError::ComputeError(
                        format!("Python function in 'map_alias' produced an error: {e}.").into(),
                    )),
                }
            })
            .into()
    }

    // todo: make this serializable
    fn lst_to_struct(
        &self,
        width_strat: Wrap<ListToStructWidthStrategy>,
        name_gen: Option<PyObject>,
        upper_bound: usize,
    ) -> PyResult<Self> {
        let name_gen = name_gen.map(|lambda| {
            Arc::new(move |idx: usize| {
                Python::with_gil(|py| {
                    let out = lambda.call1(py, (idx,)).unwrap();
                    out.extract::<String>(py).unwrap()
                })
            }) as NameGenerator
        });

        Ok(self
            .inner
            .clone()
            .arr()
            .to_struct(width_strat.0, name_gen, upper_bound)
            .into())
    }
}

pub fn map_mul(
    pyexpr: Vec<PyExpr>,
    lambda: PyObject,
    output_type: Option<Wrap<DataType>>,
    apply_groups: bool,
    returns_scalar: bool,
) -> PyExpr {
    PyUdfLambda {
        lambda: PyLambda(lambda),
        op: UdfLambdaOp::MapMultiple {
            apply_groups,
            returns_scalar,
        },
        output_type: output_type.map(|wrap| wrap.0),
    }
    .into_expr(py_exprs_to_exprs(pyexpr))
    .into()
}

pub fn fold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    let mut exprs = py_exprs_to_exprs(exprs);
    exprs.push(acc.inner);

    PyUdfLambda {
        lambda: PyLambda(lambda),
        op: UdfLambdaOp::Fold,
        output_type: None,
    }
    .into_expr(exprs)
    .into()
}

pub fn reduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    PyUdfLambda {
        lambda: PyLambda(lambda),
        op: UdfLambdaOp::Reduce,
        output_type: None,
    }
    .into_expr(py_exprs_to_exprs(exprs))
    .into()
}

pub fn cumfold(acc: PyExpr, lambda: PyObject, exprs: Vec<PyExpr>, include_init: bool) -> PyExpr {
    let mut exprs = py_exprs_to_exprs(exprs);
    exprs.push(acc.inner);

    PyUdfLambda {
        lambda: PyLambda(lambda),
        op: UdfLambdaOp::CumFold { include_init },
        output_type: None,
    }
    .into_expr(exprs)
    .into()
}

pub fn cumreduce(lambda: PyObject, exprs: Vec<PyExpr>) -> PyExpr {
    PyUdfLambda {
        lambda: PyLambda(lambda),
        op: UdfLambdaOp::CumReduce,
        output_type: None,
    }
    .into_expr(py_exprs_to_exprs(exprs))
    .into()
}
