use polars::frame::row::{AnyValueBuffer, Row, rows_to_schema_supertypes, rows_to_supertypes};
use polars::prelude::*;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyMapping, PyString};

use super::PyDataFrame;
use crate::conversion::any_value::py_object_to_any_value;
use crate::conversion::{Wrap, vec_extract_wrapped};
use crate::error::PyPolarsErr;
use crate::interop;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PyDataFrame {
    #[staticmethod]
    #[pyo3(signature = (data, schema=None, infer_schema_length=None))]
    pub fn from_rows(
        py: Python<'_>,
        data: Vec<Wrap<Row>>,
        schema: Option<Wrap<Schema>>,
        infer_schema_length: Option<usize>,
    ) -> PyResult<Self> {
        let data = vec_extract_wrapped(data);
        let schema = schema.map(|wrap| wrap.0);
        py.enter_polars(move || finish_from_rows(data, schema, None, infer_schema_length))
    }

    #[staticmethod]
    #[pyo3(signature = (data, schema=None, schema_overrides=None, strict=true, infer_schema_length=None, index_name=None))]
    pub fn from_dicts(
        py: Python<'_>,
        data: &Bound<PyAny>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        strict: bool,
        infer_schema_length: Option<usize>,
        index_name: Option<String>,
    ) -> PyResult<Self> {
        let schema = schema.map(|wrap| wrap.0);
        let schema_overrides = schema_overrides.map(|wrap| wrap.0);
        // A dict value needs its target dtype to be read as a Map rather than a Struct.
        let dtype_hint = |name: &str| {
            schema_overrides
                .as_ref()
                .and_then(|s| s.get(name))
                .or_else(|| schema.as_ref().and_then(|s| s.get(name)))
                .cloned()
        };

        // stream the records; indexed data (`{key: [record, ...]}`) pairs each
        // record with the key that fills its index value, if it has none
        let mut records: Box<dyn Iterator<Item = PyResult<KeyedRecord>>> = match index_name
            .as_deref()
        {
            None => Box::new(data.try_iter()?.map(|record| Ok((record?, None)))),
            Some(name) => {
                let use_key = schema.as_ref().is_none_or(|s| s.contains(name));
                let key_dtype = dtype_hint(name);
                let mut groups = data.call_method0("items")?.try_iter()?;
                let mut group: Option<(Option<AnyValue>, Bound<PyIterator>)> = None;
                let mut next = move || -> PyResult<Option<_>> {
                    loop {
                        if let Some((key, records)) = &mut group
                            && let Some(record) = records.next()
                        {
                            return Ok(Some((record?, key.clone())));
                        }
                        let Some(item) = groups.next() else {
                            return Ok(None);
                        };
                        let (key, records) = item?.extract::<(Bound<PyAny>, Bound<PyAny>)>()?;
                        let key = use_key
                            .then(|| py_object_to_any_value(&key, strict, true, key_dtype.as_ref()))
                            .transpose()?;
                        group = Some((key, records.try_iter()?));
                    }
                };
                Box::new(std::iter::from_fn(move || next().transpose()))
            },
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
        let leading = records
            .by_ref()
            .take(n_infer)
            .map(|item| item.and_then(|(record, key)| Ok((Record::new(record)?, key))))
            .collect::<PyResult<Vec<_>>>()?;

        let names: Vec<String> = match &schema {
            Some(schema) => schema.iter_names().map(|name| name.to_string()).collect(),
            None => {
                // in order of appearance, after the index column
                let mut names = PlIndexSet::from_iter(index_name.clone());
                for (record, _) in &leading {
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
        let index_pos = index_name
            .as_deref()
            .and_then(|index| names.iter().position(|name| name == index));
        let read = |record: &Record, key: &Option<AnyValue<'static>>, i: usize| {
            let value = record.value(&keys[i], hints[i].as_ref(), strict)?;
            PyResult::Ok(match key {
                Some(key) if value.is_null() && index_pos == Some(i) => key.clone(),
                _ => value,
            })
        };
        let rows = leading
            .iter()
            .map(|(record, key)| (0..names.len()).map(|i| read(record, key, i)).collect())
            .map(|row: PyResult<_>| row.map(Row))
            .collect::<PyResult<Vec<_>>>()?;

        let has_schema = schema.is_some();
        let mut schema = schema
            .unwrap_or_else(|| columns_names_to_empty_schema(names.iter().map(String::as_str)));
        resolve_schema_overrides(&mut schema, schema_overrides);
        if rows.is_empty() && has_schema {
            // no records, so unknown dtypes are null
            for dtype in schema.iter_values_mut().filter(|dtype| !dtype.is_known()) {
                *dtype = DataType::Null;
            }
        } else {
            update_schema_from_rows(&mut schema, &rows, infer_schema_length)?;
        }

        // values move into the buffers; a rejected value is read again, for the error
        let capacity = match index_name {
            None => data.len()?,
            // an unsized group counts as one record
            Some(_) => data
                .call_method0("values")?
                .try_iter()?
                .map(|group| group.map(|group| group.len().unwrap_or(1)))
                .sum::<PyResult<usize>>()?,
        };
        let mut buffers: Vec<AnyValueBuffer> = schema
            .iter_values()
            .map(|dtype| AnyValueBuffer::new(dtype, capacity))
            .collect();
        let push = |buffer: &mut AnyValueBuffer<'static>, value, record: &Record, key: &_, i| {
            if buffer.add(value).is_none() {
                buffer
                    .add_fallible(&read(record, key, i)?)
                    .map_err(PyPolarsErr::from)?;
            }
            PyResult::Ok(())
        };
        let mut height = leading.len();
        for ((record, key), row) in leading.iter().zip(rows) {
            for (i, (buffer, value)) in buffers.iter_mut().zip(row.0).enumerate() {
                push(buffer, value, record, key, i)?;
            }
        }
        for item in records {
            let (record, key) = item?;
            let record = Record::new(record)?;
            for (i, buffer) in buffers.iter_mut().enumerate() {
                push(buffer, read(&record, &key, i)?, &record, &key, i)?;
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

fn finish_from_rows(
    rows: Vec<Row>,
    schema: Option<Schema>,
    schema_overrides: Option<Schema>,
    infer_schema_length: Option<usize>,
) -> PyResult<PyDataFrame> {
    let schema = if let Some(mut schema) = schema {
        resolve_schema_overrides(&mut schema, schema_overrides);
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

/// A record, with the key that fills its index value (indexed data only).
type KeyedRecord<'py> = (Bound<'py, PyAny>, Option<AnyValue<'static>>);

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
