use polars::frame::row::{
    AnyValueBufferBatched, Row, rows_to_schema_supertypes, rows_to_supertypes,
};
use polars::prelude::*;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PySequence, PyString};

use super::PyDataFrame;
use crate::conversion::Wrap;
use crate::conversion::any_value::py_object_to_any_value;
use crate::error::PyPolarsErr;
use crate::interop;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PyDataFrame {
    #[staticmethod]
    #[pyo3(signature = (data, schema=None, infer_schema_length=None))]
    pub fn from_rows(
        py: Python<'_>,
        data: &Bound<PyAny>,
        schema: Option<Wrap<Schema>>,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let data = data.cast::<PySequence>()?;
        let schema = schema.map(|wrap| wrap.0);
        let extract = |record: &Bound<PyAny>| PyResult::Ok(record.extract::<Wrap<Row>>()?.0);

        // the leading records infer the schema (none are needed if it is complete)
        let schema_is_complete = schema
            .as_ref()
            .is_some_and(|s| s.iter_values().all(|dtype| dtype.is_known()));
        let n_infer = match infer_schema_length {
            _ if schema_is_complete => 0,
            Some(n) => n.max(1),
            None => usize::MAX,
        };
        let mut records = data.try_iter()?;
        let leading = records
            .by_ref()
            .take(n_infer)
            .collect::<PyResult<Vec<_>>>()?;
        let rows = leading.iter().map(extract).collect::<PyResult<Vec<_>>>()?;
        let schema = match schema {
            Some(mut schema) => {
                update_schema_from_rows(&mut schema, &rows, infer_schema_length)?;
                schema
            },
            None => {
                rows_to_schema_supertypes(&rows, infer_schema_length).map_err(PyPolarsErr::from)?
            },
        };

        // values move into the buffers; a rejected value is read again, for the error
        let capacity = data.len()?;
        let mut buffers: Vec<_> = schema
            .iter_values()
            .map(|dtype| AnyValueBufferBatched::new(dtype, capacity))
            .collect();
        let read = |record: &Bound<PyAny>, i: usize| {
            PyResult::Ok(record.get_item(i)?.extract::<Wrap<AnyValue>>()?.0)
        };
        let mut width = None;
        let mut height = 0;
        let mut push_row = |record: &Bound<PyAny>, row: Row<'static>| {
            // rows are as wide as the first (any further schema columns are null)
            let expected = *width.get_or_insert(row.0.len().min(buffers.len()));
            if row.0.len() != expected {
                return Err(PyPolarsErr::from(polars_err!(
                    ShapeMismatch: "row at index {} has length {} (expected {})",
                    height, row.0.len(), expected
                ))
                .into());
            }
            for (i, (buffer, value)) in buffers.iter_mut().zip(row.0).enumerate() {
                push(py, buffer, value, || read(record, i))?;
            }
            height += 1;
            PyResult::Ok(())
        };
        for (record, row) in leading.iter().zip(rows) {
            push_row(record, row)?;
        }
        for record in records {
            let record = record?;
            push_row(&record, extract(&record)?)?;
        }

        py.enter_polars_df(move || {
            let columns = buffers
                .into_iter()
                .zip(schema.iter_names())
                .map(|(buffer, name)| {
                    let series = buffer.into_series()?;
                    Ok(if series.is_empty() {
                        Column::full_null(name.clone(), height, series.dtype())
                    } else {
                        series.with_name(name.clone()).into()
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            DataFrame::new(height, columns)
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, schema=None, schema_overrides=None, strict=true, infer_schema_length=None))]
    pub fn from_dicts(
        py: Python<'_>,
        data: &Bound<PyAny>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        strict: bool,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let schema = schema.map(|wrap| wrap.0);
        let schema_overrides = schema_overrides.map(|wrap| wrap.0);
        let dtype_hint = |name: &str| {
            schema_overrides
                .as_ref()
                .and_then(|s| s.get(name))
                .or_else(|| schema.as_ref().and_then(|s| s.get(name)))
                .cloned()
        };

        // the leading records infer the schema (none are needed if it is complete)
        let schema_is_complete = schema.as_ref().is_some_and(|s| {
            s.iter_names()
                .all(|name| dtype_hint(name).is_some_and(|dtype| dtype.is_known()))
        });
        let n_infer = match infer_schema_length {
            _ if schema_is_complete => 0,
            Some(n) => n.max(1),
            None => usize::MAX,
        };
        let mut records = data.try_iter()?;
        let leading = records
            .by_ref()
            .take(n_infer)
            .map(|record| Record::new(record?))
            .collect::<PyResult<Vec<_>>>()?;

        let names: Vec<String> = match &schema {
            Some(schema) => schema.iter_names().map(|name| name.to_string()).collect(),
            None => {
                // in order of appearance
                let mut names = PlIndexSet::default();
                for record in &leading {
                    record.add_names(&mut names)?;
                }
                names.into_iter().collect()
            },
        };
        let keys: Vec<Bound<PyString>> = names
            .iter()
            .map(|name| PyString::intern(py, name))
            .collect();
        let hints: Vec<Option<DataType>> = names.iter().map(|name| dtype_hint(name)).collect();
        let read = |record: &Record, i: usize| record.value(&keys[i], hints[i].as_ref(), strict);
        let rows = leading
            .iter()
            .map(|record| {
                (0..names.len())
                    .map(|i| read(record, i))
                    .collect::<PyResult<_>>()
                    .map(Row)
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut schema = schema
            .unwrap_or_else(|| columns_names_to_empty_schema(names.iter().map(String::as_str)));
        resolve_schema_overrides(&mut schema, schema_overrides);
        update_schema_from_rows(&mut schema, &rows, infer_schema_length)?;

        // values move into the buffers; a rejected value is read again, for the error
        let capacity = data.len()?;
        let mut buffers: Vec<_> = schema
            .iter_values()
            .map(|dtype| AnyValueBufferBatched::new(dtype, capacity))
            .collect();
        let mut height = leading.len();
        for (record, row) in leading.iter().zip(rows) {
            for (i, (buffer, value)) in buffers.iter_mut().zip(row.0).enumerate() {
                push(py, buffer, value, || read(record, i))?;
            }
        }
        for record in records {
            let record = Record::new(record?)?;
            for (i, buffer) in buffers.iter_mut().enumerate() {
                push(py, buffer, read(&record, i)?, || read(&record, i))?;
            }
            height += 1;
        }

        py.enter_polars_df(move || {
            let columns = buffers
                .into_iter()
                .zip(schema.iter_names())
                .map(|(buffer, name)| Ok(buffer.into_series()?.with_name(name.clone()).into()))
                .collect::<PolarsResult<Vec<_>>>()?;
            DataFrame::new(height, columns)
        })
    }

    #[staticmethod]
    pub fn from_arrow_record_batches(
        py: Python<'_>,
        rb: Vec<Bound<PyAny>>,
        schema: Bound<PyAny>,
    ) -> PyResult<Self> {
        let df = interop::arrow::to_rust::to_rust_df(py, &rb, schema)?;
        Ok(Self::from(df))
    }
}

/// Add a value, or else `reread` it (owned) for the error; full batches convert without the GIL.
#[inline]
fn push(
    py: Python<'_>,
    buffer: &mut AnyValueBufferBatched<'static>,
    value: AnyValue<'static>,
    reread: impl FnOnce() -> PyResult<AnyValue<'static>>,
) -> PyResult<()> {
    if buffer.add(value).is_none() {
        buffer.add_fallible(&reread()?).map_err(PyPolarsErr::from)?;
    }
    if buffer.needs_flush() {
        py.enter_polars(|| buffer.flush())?;
    }
    Ok(())
}

fn update_schema_from_rows(
    schema: &mut Schema,
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PyResult<()> {
    let schema_is_complete = schema.iter_values().all(|dtype| dtype.is_known());
    if schema_is_complete {
        return Ok(());
    }

    // TODO: Only infer dtypes for columns with an unknown dtype
    let inferred_dtypes =
        rows_to_supertypes(rows, infer_schema_length).map_err(PyPolarsErr::from)?;
    let inferred_dtypes_slice = inferred_dtypes.as_slice();

    for (i, dtype) in schema.iter_values_mut().enumerate() {
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

fn columns_names_to_empty_schema<'a, I>(column_names: I) -> Schema
where
    I: IntoIterator<Item = &'a str>,
{
    let fields = column_names
        .into_iter()
        .map(|c| Field::new(c.into(), DataType::Unknown(Default::default())));
    Schema::from_iter(fields)
}

/// A dict (or, slower, any mapping) record; `None` is a record of nulls.
enum Record<'py> {
    Null,
    Dict(Bound<'py, PyDict>),
    Mapping(Bound<'py, PyMapping>),
}

impl<'py> Record<'py> {
    fn new(record: Bound<'py, PyAny>) -> PyResult<Self> {
        if record.is_none() {
            return Ok(Self::Null);
        }
        Ok(match record.cast_into::<PyDict>() {
            Ok(dict) => Self::Dict(dict),
            Err(err) => Self::Mapping(err.into_inner().cast_into::<PyMapping>()?),
        })
    }

    /// Add the record's keys to the (ordered) schema names.
    fn add_names(&self, names: &mut PlIndexSet<String>) -> PyResult<()> {
        let keys = match self {
            Self::Null => return Ok(()),
            Self::Dict(dict) => dict.keys(),
            Self::Mapping(mapping) => mapping.keys()?,
        };
        for key in keys {
            let key = key.cast::<PyString>()?.to_str()?;
            if !names.contains(key) {
                names.insert(key.to_owned());
            }
        }
        Ok(())
    }

    #[inline]
    fn value(
        &self,
        key: &Bound<'py, PyString>,
        dtype: Option<&DataType>,
        strict: bool,
    ) -> PyResult<AnyValue<'static>> {
        let value = match self {
            Self::Null => None,
            Self::Dict(dict) => dict.get_item(key)?,
            Self::Mapping(mapping) => match mapping.get_item(key) {
                Err(err) if err.is_instance_of::<PyKeyError>(mapping.py()) => None,
                value => Some(value?),
            },
        };
        match value {
            Some(value) if !value.is_none() => py_object_to_any_value(&value, strict, true, dtype),
            _ => Ok(AnyValue::Null),
        }
    }
}
