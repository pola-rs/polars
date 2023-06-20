use std::ptr::null;
use pyo3::{PyObject, Python};
use serde::{
    Serialize, Deserialize
};
use polars_arrow::error::PolarsResult;
use polars_core::datatypes::DataType;
use polars_core::prelude::Series;
use super::expr_dyn_fn::*;
use once_cell::sync::Lazy;
use polars_core::error::*;

pub static mut CALL_LAMBDA: Option<fn(s: &Series, lambda: &PyObject) -> PolarsResult<Series>> = None;

pub struct PythonFunction {
    lambda: PyObject,
    output_type: Option<DataType>,
    agg_list: bool
}

impl PythonFunction {
    pub fn new(lambda: PyObject, output_type: Option<DataType>, agg_list: bool)  -> Self {
        Self {
            lambda,
            output_type,
            agg_list
        }
    }
}


impl SeriesUdf for PythonFunction {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        let func = unsafe {
            CALL_LAMBDA.unwrap()
        };

        Python::with_gil(|py| {
            let output_type = self.output_type.clone().unwrap_or(DataType::Unknown);
            let out = func(&s[0], &self.lambda)?;
            polars_ensure!(
                matches!(output_type, DataType::Unknown) || out.dtype() == &output_type,
                SchemaMismatch:
                "expected output type '{:?}', got '{:?}'; set `return_dtype` to the proper datatype",
                output_type, out.dtype(),
            );
            Ok(Some(out))
        })
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
