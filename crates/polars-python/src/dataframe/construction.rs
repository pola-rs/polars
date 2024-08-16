use polars::frame::row::{rows_to_schema_supertypes, rows_to_supertypes, Row};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::PyDataFrame;
use crate::conversion::any_value::py_object_to_any_value;
use crate::conversion::{vec_extract_wrapped, Wrap};
use crate::error::PyPolarsErr;
use crate::interop;

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
    #[pyo3(signature = (data, schema=None, schema_overrides=None, strict=true, infer_schema_length=None))]
    pub fn from_dicts(
        py: Python,
        data: &Bound<PyAny>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        strict: bool,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let schema = schema.map(|wrap| wrap.0);
        let schema_overrides = schema_overrides.map(|wrap| wrap.0);

        let names = get_schema_names(data, schema.as_ref(), infer_schema_length)?;
        let rows = dicts_to_rows(data, &names, strict)?;

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
    pub fn from_arrow_record_batches(rb: Vec<Bound<PyAny>>) -> PyResult<Self> {
        let df = interop::arrow::to_rust::to_rust_df(&rb)?;
        Ok(Self::from(df))
    }
}

fn finish_from_rows(
    rows: Vec<Row>,
    schema: Option<Schema>,
    schema_overrides: Option<Schema>,
    infer_schema_length: Option<usize>,
) -> PyResult<PyDataFrame> {
    let mut schema = if let Some(mut schema) = schema {
        resolve_schema_overrides(&mut schema, schema_overrides);
        update_schema_from_rows(&mut schema, &rows, infer_schema_length)?;
        schema
    } else {
        rows_to_schema_supertypes(&rows, infer_schema_length).map_err(PyPolarsErr::from)?
    };

    // TODO: Remove this step when Decimals are supported properly.
    // Erasing the decimal precision/scale here will just require us to infer it again later.
    // https://github.com/pola-rs/polars/issues/14427
    erase_decimal_precision_scale(&mut schema);

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
///
/// Overrides for nonexistent columns are ignored.
fn resolve_schema_overrides(schema: &mut Schema, schema_overrides: Option<Schema>) {
    if let Some(overrides) = schema_overrides {
        for (name, dtype) in overrides.into_iter() {
            schema.set_dtype(name.as_str(), dtype);
        }
    }
}

/// Erase precision/scale information from Decimal types.
fn erase_decimal_precision_scale(schema: &mut Schema) {
    for dtype in schema.iter_dtypes_mut() {
        if let DataType::Decimal(_, _) = dtype {
            *dtype = DataType::Decimal(None, None)
        }
    }
}

fn columns_names_to_empty_schema<'a, I>(column_names: I) -> Schema
where
    I: IntoIterator<Item = &'a str>,
{
    let fields = column_names
        .into_iter()
        .map(|c| Field::new(c, DataType::Unknown(Default::default())));
    Schema::from_iter(fields)
}

fn dicts_to_rows<'a>(
    data: &Bound<'a, PyAny>,
    names: &'a [String],
    strict: bool,
) -> PyResult<Vec<Row<'a>>> {
    let len = data.len()?;
    let mut rows = Vec::with_capacity(len);
    for d in data.iter()? {
        let d = d?;
        let d = d.downcast::<PyDict>()?;

        let mut row = Vec::with_capacity(names.len());
        for k in names.iter() {
            let val = match d.get_item(k)? {
                None => AnyValue::Null,
                Some(val) => py_object_to_any_value(&val.as_borrowed(), strict)?,
            };
            row.push(val)
        }
        rows.push(Row(row))
    }
    Ok(rows)
}

/// Either read the given schema, or infer the schema names from the data.
fn get_schema_names(
    data: &Bound<PyAny>,
    schema: Option<&Schema>,
    infer_schema_length: Option<usize>,
) -> PyResult<Vec<String>> {
    if let Some(schema) = schema {
        Ok(schema.iter_names().map(|n| n.to_string()).collect())
    } else {
        infer_schema_names_from_data(data, infer_schema_length)
    }
}

/// Infer schema names from an iterable of dictionaries.
///
/// The resulting schema order is determined by the order in which the names are encountered in
/// the data.
fn infer_schema_names_from_data(
    data: &Bound<PyAny>,
    infer_schema_length: Option<usize>,
) -> PyResult<Vec<String>> {
    let data_len = data.len()?;
    let infer_schema_length = infer_schema_length
        .map(|n| std::cmp::max(1, n))
        .unwrap_or(data_len);

    let mut names = PlIndexSet::new();
    for d in data.iter()?.take(infer_schema_length) {
        let d = d?;
        let d = d.downcast::<PyDict>()?;
        let keys = d.keys();
        for name in keys {
            let name = name.extract::<String>()?;
            names.insert(name);
        }
    }
    Ok(names.into_iter().collect())
}
