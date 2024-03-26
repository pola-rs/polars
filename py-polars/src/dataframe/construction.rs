use polars::frame::row::{rows_to_schema_supertypes, rows_to_supertypes, Row};
use pyo3::prelude::*;

use super::*;
use crate::arrow_interop;
use crate::conversion::{vec_extract_wrapped, Wrap};

#[pymethods]
impl PyDataFrame {
    #[staticmethod]
    pub fn from_rows(
        py: Python,
        data: Vec<Wrap<Row>>,
        schema: Option<Wrap<Schema>>,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let data = vec_extract_wrapped(data);
        let schema = schema.map(|wrap| wrap.0);
        py.allow_threads(move || finish_from_rows(data, schema, None, infer_schema_length))
    }

    #[staticmethod]
    #[pyo3(signature = (data, schema=None, schema_overrides=None, infer_schema_length=None))]
    pub fn from_dicts(
        py: Python,
        data: &PyAny,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let schema = schema.map(|wrap| wrap.0);
        let schema_overrides = schema_overrides.map(|wrap| wrap.0);

        // If given, read dict fields in schema order.
        let mut schema_columns = PlIndexSet::new();
        if let Some(ref s) = schema {
            schema_columns.extend(s.iter_names().map(|n| n.to_string()))
        }

        let (rows, names) = dicts_to_rows(data, infer_schema_length, schema_columns)?;

        let schema = schema.or_else(|| {
            Some(columns_names_to_empty_schema(
                names.iter().map(String::as_str),
            ))
        });

        py.allow_threads(move || {
            finish_from_rows(rows, schema, schema_overrides, infer_schema_length)
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
    schema_overrides: Option<Schema>,
    infer_schema_length: Option<usize>,
) -> PyResult<PyDataFrame> {
    let schema = if let Some(mut schema) = schema {
        resolve_schema_overrides(&mut schema, schema_overrides)?;
        update_schema_from_rows(&mut schema, &rows, infer_schema_length)?;
        schema
    } else {
        rows_to_schema_supertypes(&rows, infer_schema_length).map_err(PyPolarsErr::from)?
    };

    let df = DataFrame::from_rows_and_schema(&rows, &schema).map_err(PyPolarsErr::from)?;
    Ok(df.into())
}

fn update_schema_from_rows(
    schema: &mut Schema,
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PyResult<()> {
    let schema_is_complete = schema.iter_dtypes().all(|dtype| dtype.is_known());
    if schema_is_complete {
        return Ok(());
    }

    // TODO: Only infer dtypes for columns with an unknown dtype
    let inferred_dtypes =
        rows_to_supertypes(rows, infer_schema_length).map_err(PyPolarsErr::from)?;
    let inferred_dtypes_slice = inferred_dtypes.as_slice();

    for (i, dtype) in schema.iter_dtypes_mut().enumerate() {
        if !dtype.is_known() {
            *dtype = inferred_dtypes_slice.get(i).ok_or_else(|| {
                polars_err!(SchemaMismatch: "the number of columns in the schema does not match the data")
            })
            .map_err(PyPolarsErr::from)?
            .clone();
        }
    }
    Ok(())
}

/// Override the data type of certain schema fields.
fn resolve_schema_overrides(schema: &mut Schema, schema_overrides: Option<Schema>) -> PyResult<()> {
    if let Some(overrides) = schema_overrides {
        for (name, dtype) in overrides.into_iter() {
            schema.set_dtype(name.as_str(), dtype).ok_or_else(|| {
                polars_err!(SchemaMismatch: "nonexistent column specified in `schema_overrides`: {name}")
            }).map_err(PyPolarsErr::from)?;
        }
    }
    Ok(())
}

fn columns_names_to_empty_schema<'a, I>(column_names: I) -> Schema
where
    I: IntoIterator<Item = &'a str>,
{
    let fields = column_names
        .into_iter()
        .map(|c| Field::new(c, DataType::Unknown));
    Schema::from_iter(fields)
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
