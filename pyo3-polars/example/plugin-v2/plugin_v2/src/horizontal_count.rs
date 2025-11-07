use polars::error::{polars_bail, PolarsResult};
use polars::prelude::{ArithmeticChunked, ChunkCast, DataType, Field, IdxCa, Schema};
use polars::series::{IntoSeries, Series};
use pyo3_polars::polars_plugin_expr_info;
use pyo3_polars::v1::{self, PolarsPluginExprInfo};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct HorizontalCount;

impl v1::elementwise::PolarsElementwisePlugin for HorizontalCount {
    fn to_field(&self, _fields: &Schema) -> PolarsResult<Field> {
        Ok(Field::new("horizontal_count".into(), DataType::IDX_DTYPE))
    }

    fn evaluate(&self, inputs: &[Series]) -> PolarsResult<Series> {
        let mut length = 1;
        for i in inputs {
            if i.len() == 1 {
                continue;
            }
            if length != 1 && i.len() != length {
                polars_bail!(length_mismatch = "horizontal_count", length, i.len());
            }
            length = i.len();
        }

        let mut acc = IdxCa::new_vec("horizontal_count".into(), vec![0; length]);
        for i in inputs {
            let i = i.is_not_null();
            let i = i.cast(&DataType::IDX_DTYPE).unwrap();
            let i = i.idx()?;
            acc = acc.wrapping_add(i.clone());
        }

        Ok(acc.into_series())
    }
}

#[pyo3::pyfunction]
pub fn horizontal_count() -> PolarsPluginExprInfo {
    polars_plugin_expr_info!(
        "horizontal_count",
        v1::elementwise::Plugin(HorizontalCount),
        v1::elementwise::Plugin<HorizontalCount>
    )
}
