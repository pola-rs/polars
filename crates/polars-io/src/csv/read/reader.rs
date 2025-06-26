use std::fs::File;
use std::path::PathBuf;

use polars_core::prelude::*;

use super::options::CsvReadOptions;
use super::read_impl::CoreReader;
use super::read_impl::batched::to_batched_owned;
use super::{BatchedCsvReader, OwnedBatchedCsvReader};
use crate::mmap::MmapBytesReader;
use crate::path_utils::resolve_homedir;
use crate::predicates::PhysicalIoExpr;
use crate::shared::SerReader;
use crate::utils::get_reader_bytes;

/// Create a new DataFrame by reading a csv file.
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::prelude::*;
/// use std::fs::File;
///
/// fn example() -> PolarsResult<DataFrame> {
///     CsvReadOptions::default()
///             .with_has_header(true)
///             .try_into_reader_with_file_path(Some("iris.csv".into()))?
///             .finish()
/// }
/// ```
#[must_use]
pub struct CsvReader<R>
where
    R: MmapBytesReader,
{
    /// File or Stream object.
    reader: R,
    /// Options for the CSV reader.
    options: CsvReadOptions,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
}

impl<R> CsvReader<R>
where
    R: MmapBytesReader,
{
    pub fn _with_predicate(mut self, predicate: Option<Arc<dyn PhysicalIoExpr>>) -> Self {
        self.predicate = predicate;
        self
    }

    // TODO: Investigate if we can remove this
    pub(crate) fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.options.schema = Some(schema);
        self
    }
}

impl CsvReadOptions {
    /// Creates a CSV reader using a file path.
    ///
    /// # Panics
    /// If both self.path and the path parameter are non-null. Only one of them is
    /// to be non-null.
    pub fn try_into_reader_with_file_path(
        mut self,
        path: Option<PathBuf>,
    ) -> PolarsResult<CsvReader<File>> {
        if self.path.is_some() {
            assert!(
                path.is_none(),
                "impl error: only 1 of self.path or the path parameter is to be non-null"
            );
        } else {
            self.path = path;
        };

        assert!(
            self.path.is_some(),
            "impl error: either one of self.path or the path parameter is to be non-null"
        );

        let path = resolve_homedir(self.path.as_ref().unwrap());
        let reader = polars_utils::open_file(&path)?;
        let options = self;

        Ok(CsvReader {
            reader,
            options,
            predicate: None,
        })
    }

    /// Creates a CSV reader using a file handle.
    pub fn into_reader_with_file_handle<R: MmapBytesReader>(self, reader: R) -> CsvReader<R> {
        let options = self;

        CsvReader {
            reader,
            options,
            predicate: Default::default(),
        }
    }
}

impl<R: MmapBytesReader> CsvReader<R> {
    fn core_reader(&mut self) -> PolarsResult<CoreReader> {
        let reader_bytes = get_reader_bytes(&mut self.reader)?;

        let parse_options = self.options.get_parse_options();

        CoreReader::new(
            reader_bytes,
            parse_options,
            self.options.n_rows,
            self.options.skip_rows,
            self.options.skip_lines,
            self.options.projection.clone().map(|x| x.as_ref().clone()),
            self.options.infer_schema_length,
            self.options.has_header,
            self.options.ignore_errors,
            self.options.schema.clone(),
            self.options.columns.clone(),
            self.options.n_threads,
            self.options.schema_overwrite.clone(),
            self.options.dtype_overwrite.clone(),
            self.options.chunk_size,
            self.predicate.clone(),
            self.options.fields_to_cast.clone(),
            self.options.skip_rows_after_header,
            self.options.row_index.clone(),
            self.options.raise_if_empty,
        )
    }

    pub fn batched_borrowed(&mut self) -> PolarsResult<BatchedCsvReader> {
        let csv_reader = self.core_reader()?;
        csv_reader.batched()
    }
}

impl CsvReader<Box<dyn MmapBytesReader>> {
    pub fn batched(mut self, schema: Option<SchemaRef>) -> PolarsResult<OwnedBatchedCsvReader> {
        if let Some(schema) = schema {
            self = self.with_schema(schema);
        }

        to_batched_owned(self)
    }
}

impl<R> SerReader<R> for CsvReader<R>
where
    R: MmapBytesReader,
{
    /// Create a new CsvReader from a file/stream using default read options. To
    /// use non-default read options, first construct [CsvReadOptions] and then use
    /// any of the `(try)_into_` methods.
    fn new(reader: R) -> Self {
        CsvReader {
            reader,
            options: Default::default(),
            predicate: None,
        }
    }

    /// Read the file and create the DataFrame.
    fn finish(mut self) -> PolarsResult<DataFrame> {
        let rechunk = self.options.rechunk;
        let low_memory = self.options.low_memory;

        let csv_reader = self.core_reader()?;
        let mut df = csv_reader.finish()?;

        // Important that this rechunk is never done in parallel.
        // As that leads to great memory overhead.
        if rechunk && df.first_col_n_chunks() > 1 {
            if low_memory {
                df.as_single_chunk();
            } else {
                df.as_single_chunk_par();
            }
        }

        Ok(df)
    }
}

impl<R: MmapBytesReader> CsvReader<R> {
    /// Sets custom CSV read options.
    pub fn with_options(mut self, options: CsvReadOptions) -> Self {
        self.options = options;
        self
    }
}

/// Splits datatypes that cannot be natively read into a `fields_to_cast` for
/// post-read casting.
///
/// # Returns
/// `has_categorical`
pub fn prepare_csv_schema(
    schema: &mut SchemaRef,
    fields_to_cast: &mut Vec<Field>,
) -> PolarsResult<bool> {
    // This branch we check if there are dtypes we cannot parse.
    // We only support a few dtypes in the parser and later cast to the required dtype
    let mut _has_categorical = false;

    let mut changed = false;

    let new_schema = schema
        .iter_fields()
        .map(|mut fld| {
            use DataType::*;

            let mut matched = true;

            let out = match fld.dtype() {
                Time => {
                    fields_to_cast.push(fld.clone());
                    fld.coerce(String);
                    PolarsResult::Ok(fld)
                },
                #[cfg(feature = "dtype-categorical")]
                Categorical(_, _) => {
                    _has_categorical = true;
                    PolarsResult::Ok(fld)
                },
                #[cfg(feature = "dtype-decimal")]
                Decimal(precision, scale) => match (precision, scale) {
                    (_, Some(_)) => {
                        fields_to_cast.push(fld.clone());
                        fld.coerce(String);
                        PolarsResult::Ok(fld)
                    },
                    _ => Err(PolarsError::ComputeError(
                        "'scale' must be set when reading csv column as Decimal".into(),
                    )),
                },
                _ => {
                    matched = false;
                    PolarsResult::Ok(fld)
                },
            }?;

            changed |= matched;

            PolarsResult::Ok(out)
        })
        .collect::<PolarsResult<Schema>>()?;

    if changed {
        *schema = Arc::new(new_schema);
    }

    Ok(_has_categorical)
}
