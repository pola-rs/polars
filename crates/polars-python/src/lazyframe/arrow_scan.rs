//! Arrow C Stream scan support for streaming LazyFrame execution.
//!
//! This module provides an `AnonymousScan` implementation that consumes Arrow C Streams
//! via the Arrow PyCapsule Interface for streaming execution in the Polars engine.

use std::any::Any;
use std::sync::Arc;

use arrow::ffi::{ArrowArrayStream, ArrowArrayStreamReader};
use parking_lot::Mutex;
use polars_core::prelude::*;
use polars_plan::plans::{AnonymousScan, AnonymousScanArgs};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Validates that a PyCapsule has the expected name for Arrow C Stream.
fn validate_stream_capsule(capsule: &Bound<PyCapsule>) -> PyResult<()> {
    let name = capsule.name()?;
    if let Some(name) = name {
        let name_cstr = unsafe { name.as_cstr() };
        if name_cstr.to_str() != Ok("arrow_array_stream") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected PyCapsule with name 'arrow_array_stream', got '{name_cstr:?}'"
            )));
        }
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected PyCapsule to have name 'arrow_array_stream' set",
        ));
    }
    Ok(())
}

/// Internal state for the Arrow C Stream reader.
struct ArrowCStreamState {
    reader: Option<ArrowArrayStreamReader<Box<ArrowArrayStream>>>,
    schema: SchemaRef,
}

/// An AnonymousScan implementation that reads from an Arrow C Stream.
///
/// This struct wraps an Arrow C Stream (obtained via the Arrow PyCapsule Interface)
/// and provides streaming batch access for the Polars streaming engine.
pub struct ArrowCStreamScan {
    state: Arc<Mutex<ArrowCStreamState>>,
}

impl ArrowCStreamScan {
    /// Creates a new ArrowCStreamScan from a Python object implementing `__arrow_c_stream__`.
    ///
    /// # Arguments
    /// * `source` - Python object with `__arrow_c_stream__` method
    /// * `schema` - Optional schema; if None, will be inferred from the stream
    pub fn new(source: &Bound<PyAny>, schema: Option<SchemaRef>) -> PyResult<Self> {
        // Get the stream capsule from __arrow_c_stream__
        let capsule: Bound<PyCapsule> = if source.hasattr("__arrow_c_stream__")? {
            source.getattr("__arrow_c_stream__")?.call0()?.extract()?
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected an object with __arrow_c_stream__ method",
            ));
        };

        validate_stream_capsule(&capsule)?;

        // Import the stream from the capsule
        let reader = unsafe {
            #[allow(deprecated)]
            let stream_ptr = Box::new(std::ptr::replace(
                capsule.pointer() as *mut ArrowArrayStream,
                ArrowArrayStream::empty(),
            ));
            ArrowArrayStreamReader::try_new(stream_ptr)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        };

        // Derive schema from stream field if not provided
        let schema = match schema {
            Some(s) => s,
            None => {
                let field = reader.field();
                let struct_fields = match &field.dtype {
                    ArrowDataType::Struct(fields) => fields,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Arrow stream schema must be a struct type",
                        ));
                    },
                };
                Arc::new(Schema::from_iter(struct_fields.iter().map(Field::from)))
            },
        };

        Ok(Self {
            state: Arc::new(Mutex::new(ArrowCStreamState {
                reader: Some(reader),
                schema,
            })),
        })
    }

    /// Returns the schema of the Arrow C Stream.
    pub fn schema(&self) -> SchemaRef {
        self.state.lock().schema.clone()
    }
}

impl AnonymousScan for ArrowCStreamScan {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn scan(&self, scan_opts: AnonymousScanArgs) -> PolarsResult<DataFrame> {
        // For non-streaming execution, collect all batches into a single DataFrame
        let mut state = self.state.lock();
        let schema = state.schema.clone();
        let reader = state
            .reader
            .as_mut()
            .ok_or_else(|| polars_err!(ComputeError: "Arrow C Stream has already been consumed"))?;

        let mut chunks: Vec<DataFrame> = Vec::new();
        while let Some(array_result) = unsafe { reader.next() } {
            let array = array_result?;
            let df = array_to_dataframe(array, &schema)?;
            chunks.push(df);
        }

        // Mark as consumed
        state.reader = None;

        let mut result = if chunks.is_empty() {
            DataFrame::empty_with_schema(&schema)
        } else {
            polars_core::utils::accumulate_dataframes_vertical(chunks)?
        };

        // Apply projection pushdown if columns are specified
        if let Some(with_columns) = &scan_opts.with_columns {
            result = result.select(with_columns.iter().cloned())?;
        }

        // Apply n_rows limit if specified
        if let Some(n_rows) = scan_opts.n_rows {
            result = result.head(Some(n_rows));
        }

        Ok(result)
    }

    fn schema(&self, _infer_schema_length: Option<usize>) -> PolarsResult<SchemaRef> {
        Ok(self.state.lock().schema.clone())
    }

    fn allows_projection_pushdown(&self) -> bool {
        true
    }

    fn allows_predicate_pushdown(&self) -> bool {
        // Predicate pushdown is not supported for Arrow C Stream because
        // the predicate expression cannot be evaluated during batch reading.
        // Predicates are applied by the query engine after reading.
        false
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn next_batch(&self) -> PolarsResult<Option<DataFrame>> {
        let mut state = self.state.lock();
        let reader: &mut ArrowArrayStreamReader<Box<ArrowArrayStream>> = match state.reader.as_mut()
        {
            Some(r) => r,
            None => return Ok(None), // Already exhausted
        };

        match unsafe { reader.next() } {
            Some(Ok(array)) => {
                let df = array_to_dataframe(array, &state.schema)?;
                Ok(Some(df))
            },
            Some(Err(e)) => Err(e),
            None => {
                state.reader = None; // Mark as consumed
                Ok(None)
            },
        }
    }
}

/// Converts an Arrow array (expected to be a StructArray) to a DataFrame.
fn array_to_dataframe(
    array: Box<dyn arrow::array::Array>,
    schema: &Schema,
) -> PolarsResult<DataFrame> {
    let struct_array = array
        .as_any()
        .downcast_ref::<arrow::array::StructArray>()
        .ok_or_else(|| polars_err!(ComputeError: "Expected StructArray from Arrow C Stream"))?;

    let columns: Vec<Column> = struct_array
        .values()
        .iter()
        .zip(schema.iter())
        .map(|(arr, (name, _dtype))| {
            let series = unsafe {
                Series::_try_from_arrow_unchecked(name.clone(), vec![arr.clone()], arr.dtype())?
            };
            Ok(series.into_column())
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    DataFrame::new_infer_height(columns)
}
