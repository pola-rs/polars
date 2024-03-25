use polars::frame::row::{rows_to_schema_supertypes, Row};
use pyo3::prelude::*;

use super::*;
use crate::arrow_interop;
use crate::conversion::Wrap;

#[pymethods]
impl PyDataFrame {
    #[staticmethod]
    pub fn from_rows(
        py: Python,
        rows: Vec<Wrap<Row>>,
        infer_schema_length: Option<usize>,
        schema: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        // SAFETY: Wrap<T> is transparent.
        let rows = unsafe { std::mem::transmute::<Vec<Wrap<Row>>, Vec<Row>>(rows) };
        py.allow_threads(move || {
            finish_from_rows(rows, schema.map(|wrap| wrap.0), None, infer_schema_length)
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, schema, schema_overrides, strict=false, infer_schema_length=None))]
    pub fn from_dicts(
        py: Python,
        data: &PyAny,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        strict: bool,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        // If given, read dict fields in schema order.
        let mut schema_columns = PlIndexSet::new();
        if let Some(s) = &schema {
            schema_columns.extend(s.0.iter_names().map(|n| n.to_string()))
        }

        let (rows, names) = dicts_to_rows(data, infer_schema_length, schema_columns)?;

        py.allow_threads(move || {
            let mut schema_overrides_by_idx: Vec<(usize, DataType)> = Vec::new();
            if let Some(overrides) = schema_overrides {
                for (idx, name) in names.iter().enumerate() {
                    if let Some(dtype) = overrides.0.get(name) {
                        schema_overrides_by_idx.push((idx, dtype.clone()));
                    }
                }
            }
            let mut pydf = finish_from_rows(
                rows,
                schema.map(|wrap| wrap.0),
                Some(schema_overrides_by_idx),
                infer_schema_length,
            )?;
            unsafe {
                for (s, name) in pydf.df.get_columns_mut().iter_mut().zip(&names) {
                    s.rename(name);
                }
            }
            let length = names.len();
            if names.into_iter().collect::<PlHashSet<_>>().len() != length {
                let err = PolarsError::Duplicate("duplicate column names found".into());
                Err(PyPolarsErr::Polars(err))?;
            }

            Ok(pydf)
        })
    }

    #[staticmethod]
    pub fn from_arrow_record_batches(rb: Vec<&PyAny>) -> PyResult<Self> {
        let df = arrow_interop::to_rust::to_rust_df(&rb)?;
        Ok(Self::from(df))
    }
}

fn finish_from_rows(
    rows: Vec<Row>,
    schema: Option<Schema>,
    schema_overrides_by_idx: Option<Vec<(usize, DataType)>>,
    infer_schema_length: Option<usize>,
) -> PyResult<PyDataFrame> {
    // Object builder must be registered, this is done on import.
    let mut final_schema =
        rows_to_schema_supertypes(&rows, infer_schema_length.map(|n| std::cmp::max(1, n)))
            .map_err(PyPolarsErr::from)?;

    // Erase scale from inferred decimals.
    for dtype in final_schema.iter_dtypes_mut() {
        if let DataType::Decimal(_, _) = dtype {
            *dtype = DataType::Decimal(None, None)
        }
    }

    // Integrate explicit/inferred schema.
    if let Some(schema) = schema {
        for (i, (name, dtype)) in schema.into_iter().enumerate() {
            if let Some((name_, dtype_)) = final_schema.get_at_index_mut(i) {
                *name_ = name;

                // If schema dtype is Unknown, overwrite with inferred datatype.
                if !matches!(dtype, DataType::Unknown) {
                    *dtype_ = dtype;
                }
            } else {
                final_schema.with_column(name, dtype);
            }
        }
    }

    // Optional per-field overrides; these supersede default/inferred dtypes.
    if let Some(overrides) = schema_overrides_by_idx {
        for (i, dtype) in overrides {
            if let Some((_, dtype_)) = final_schema.get_at_index_mut(i) {
                if !matches!(dtype, DataType::Unknown) {
                    *dtype_ = dtype;
                }
            }
        }
    }
    let df = DataFrame::from_rows_and_schema(&rows, &final_schema).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

fn dicts_to_rows(
    records: &PyAny,
    infer_schema_len: Option<usize>,
    schema_columns: PlIndexSet<String>,
) -> PyResult<(Vec<Row>, Vec<String>)> {
    let infer_schema_len = infer_schema_len
        .map(|n| std::cmp::max(1, n))
        .unwrap_or(usize::MAX);
    let len = records.len()?;

    let key_names = {
        if !schema_columns.is_empty() {
            schema_columns
        } else {
            let mut inferred_keys = PlIndexSet::new();
            for d in records.iter()?.take(infer_schema_len) {
                let d = d?;
                let d = d.downcast::<PyDict>()?;
                let keys = d.keys();
                for name in keys {
                    let name = name.extract::<String>()?;
                    inferred_keys.insert(name);
                }
            }
            inferred_keys
        }
    };
    let mut rows = Vec::with_capacity(len);

    for d in records.iter()? {
        let d = d?;
        let d = d.downcast::<PyDict>()?;

        let mut row = Vec::with_capacity(key_names.len());
        for k in key_names.iter() {
            let val = match d.get_item(k)? {
                None => AnyValue::Null,
                Some(val) => val.extract::<Wrap<AnyValue>>()?.0,
            };
            row.push(val)
        }
        rows.push(Row(row))
    }
    Ok((rows, key_names.into_iter().collect()))
}
