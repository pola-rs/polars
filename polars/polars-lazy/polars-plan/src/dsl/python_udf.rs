use std::ptr::null;
use std::sync::Arc;

use once_cell::sync::Lazy;
use polars_arrow::error::PolarsResult;
use polars_core::datatypes::{DataType, Field};
use polars_core::error::*;
use polars_core::prelude::Series;
use pyo3::{PyObject, Python};
use serde::{Deserialize, Serialize};

use super::expr_dyn_fn::*;
use crate::constants::MAP_LIST_NAME;
use crate::prelude::*;

pub static mut CALL_LAMBDA: Option<fn(s: Series, lambda: &PyObject) -> PolarsResult<Series>> = None;

pub struct PythonFunction {
    lambda: PyObject,
    output_type: Option<DataType>,
}

impl PythonFunction {
    pub fn new(lambda: PyObject, output_type: Option<DataType>) -> Self {
        Self {
            lambda,
            output_type,
        }
    }
}

impl SeriesUdf for PythonFunction {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        let func = unsafe { CALL_LAMBDA.unwrap() };

        let output_type = self.output_type.clone().unwrap_or(DataType::Unknown);
        let out = func(s[0].clone(), &self.lambda)?;

        polars_ensure!(
            matches!(output_type, DataType::Unknown) || out.dtype() == &output_type,
            SchemaMismatch:
            "expected output type '{:?}', got '{:?}'; set `return_dtype` to the proper datatype",
            output_type, out.dtype(),
        );
        Ok(Some(out))
    }
}

// impl<T: Serialize> Serialize for SeriesUdf {
//     fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
//         where
//             S: Serializer,
//     {
//         self.0.serialize(serializer)
//     }
// }

// #[cfg(feature = "serde")]
// impl<'a, T: Deserialize<'a>> Deserialize<'a> for SpecialEq<T> {
//     fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
//         where
//             D: Deserializer<'a>,
//     {
//         let t = T::deserialize(deserializer)?;
//         Ok(SpecialEq(t))
//     }
// }

impl Expr {
    pub fn map_python(self, func: PythonFunction, agg_list: bool) -> Expr {
        let (collect_groups, name) = if agg_list {
            (ApplyOptions::ApplyList, MAP_LIST_NAME)
        } else {
            (ApplyOptions::ApplyFlat, "python_udf")
        };

        let return_dtype = func.output_type.clone();
        let output_type = GetOutput::map_field(move |fld| match return_dtype {
            Some(ref dt) => Field::new(fld.name(), dt.clone()),
            None => {
                let mut fld = fld.clone();
                fld.coerce(DataType::Unknown);
                fld
            }
        });

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(func)),
            output_type,
            options: FunctionOptions {
                collect_groups,
                fmt_str: name,
                ..Default::default()
            },
        }
    }
}
