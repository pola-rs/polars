use std::fs::File;
use std::path::PathBuf;

use polars_core::prelude::*;
#[cfg(feature = "temporal")]
use polars_time::prelude::*;
#[cfg(feature = "temporal")]
use rayon::prelude::*;

use super::options::CsvReadOptions;
use super::read_impl::batched::to_batched_owned;
use super::read_impl::CoreReader;
use super::{infer_file_schema, BatchedCsvReader, OwnedBatchedCsvReader};
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

    // TODO: Investigate if we can remove this
    pub(crate) fn get_schema(&self) -> Option<SchemaRef> {
        self.options.schema.clone()
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
    fn core_reader(
        &mut self,
        schema: Option<SchemaRef>,
        to_cast: Vec<Field>,
    ) -> PolarsResult<CoreReader> {
        let reader_bytes = get_reader_bytes(&mut self.reader)?;

        let parse_options = self.options.get_parse_options();

        CoreReader::new(
            reader_bytes,
            self.options.n_rows,
            self.options.skip_rows,
            self.options.projection.clone().map(|x| x.as_ref().clone()),
            self.options.infer_schema_length,
            Some(parse_options.separator),
            self.options.has_header,
            self.options.ignore_errors,
            self.options.schema.clone(),
            self.options.columns.clone(),
            parse_options.encoding,
            self.options.n_threads,
            schema,
            self.options.dtype_overwrite.clone(),
            self.options.sample_size,
            self.options.chunk_size,
            self.options.low_memory,
            parse_options.comment_prefix.clone(),
            parse_options.quote_char,
            parse_options.eol_char,
            parse_options.null_values.clone(),
            parse_options.missing_is_null,
            self.predicate.clone(),
            to_cast,
            self.options.skip_rows_after_header,
            self.options.row_index.clone(),
            parse_options.try_parse_dates,
            self.options.raise_if_empty,
            parse_options.truncate_ragged_lines,
            parse_options.decimal_comma,
        )
    }

    // TODO:
    // * Move this step outside of the reader so that we don't do it multiple times
    //   when we read a file list.
    // * See if we can avoid constructing a filtered schema.
    fn prepare_schema_overwrite(
        &self,
        overwriting_schema: &Schema,
    ) -> PolarsResult<(Schema, Vec<Field>, bool)> {
        // This branch we check if there are dtypes we cannot parse.
        // We only support a few dtypes in the parser and later cast to the required dtype
        let mut to_cast = Vec::with_capacity(overwriting_schema.len());

        let mut _has_categorical = false;
        let mut _err: Option<PolarsError> = None;

        #[allow(unused_mut)]
        let schema = overwriting_schema
            .iter_fields()
            .filter_map(|mut fld| {
                use DataType::*;
                match fld.data_type() {
                    Time => {
                        to_cast.push(fld);
                        // let inference decide the column type
                        None
                    },
                    #[cfg(feature = "dtype-categorical")]
                    Categorical(_, _) => {
                        _has_categorical = true;
                        Some(fld)
                    },
                    #[cfg(feature = "dtype-decimal")]
                    Decimal(precision, scale) => match (precision, scale) {
                        (_, Some(_)) => {
                            to_cast.push(fld.clone());
                            fld.coerce(String);
                            Some(fld)
                        },
                        _ => {
                            _err = Some(PolarsError::ComputeError(
                                "'scale' must be set when reading csv column as Decimal".into(),
                            ));
                            None
                        },
                    },
                    _ => Some(fld),
                }
            })
            .collect::<Schema>();

        if let Some(err) = _err {
            Err(err)
        } else {
            Ok((schema, to_cast, _has_categorical))
        }
    }

    pub fn batched_borrowed(&mut self) -> PolarsResult<BatchedCsvReader> {
        if let Some(schema) = self.options.schema_overwrite.as_deref() {
            let (schema, to_cast, has_cat) = self.prepare_schema_overwrite(schema)?;
            let schema = Arc::new(schema);

            let csv_reader = self.core_reader(Some(schema), to_cast)?;
            csv_reader.batched(has_cat)
        } else {
            let csv_reader = self.core_reader(self.options.schema.clone(), vec![])?;
            csv_reader.batched(false)
        }
    }
}

impl CsvReader<Box<dyn MmapBytesReader>> {
    pub fn batched(mut self, schema: Option<SchemaRef>) -> PolarsResult<OwnedBatchedCsvReader> {
        match schema {
            Some(schema) => Ok(to_batched_owned(self.with_schema(schema))),
            None => {
                let parse_options = self.options.get_parse_options();
                let reader_bytes = get_reader_bytes(&mut self.reader)?;

                let (inferred_schema, _, _) = infer_file_schema(
                    &reader_bytes,
                    parse_options.separator,
                    self.options.infer_schema_length,
                    self.options.has_header,
                    None,
                    self.options.skip_rows,
                    self.options.skip_rows_after_header,
                    parse_options.comment_prefix.as_ref(),
                    parse_options.quote_char,
                    parse_options.eol_char,
                    parse_options.null_values.as_ref(),
                    parse_options.try_parse_dates,
                    self.options.raise_if_empty,
                    &mut self.options.n_threads,
                    parse_options.decimal_comma,
                )?;
                let schema = Arc::new(inferred_schema);
                Ok(to_batched_owned(self.with_schema(schema)))
            },
        }
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
        let schema_overwrite = self.options.schema_overwrite.clone();
        let low_memory = self.options.low_memory;

        #[cfg(feature = "dtype-categorical")]
        let mut _cat_lock = None;

        let mut df = if let Some(schema) = schema_overwrite.as_deref() {
            let (schema, to_cast, _has_cat) = self.prepare_schema_overwrite(schema)?;

            #[cfg(feature = "dtype-categorical")]
            if _has_cat {
                _cat_lock = Some(polars_core::StringCacheHolder::hold())
            }

            let mut csv_reader = self.core_reader(Some(Arc::new(schema)), to_cast)?;
            csv_reader.as_df()?
        } else {
            #[cfg(feature = "dtype-categorical")]
            {
                let has_cat = self
                    .options
                    .schema
                    .clone()
                    .map(|schema| {
                        schema
                            .iter_dtypes()
                            .any(|dtype| matches!(dtype, DataType::Categorical(_, _)))
                    })
                    .unwrap_or(false);
                if has_cat {
                    _cat_lock = Some(polars_core::StringCacheHolder::hold())
                }
            }
            let mut csv_reader = self.core_reader(self.options.schema.clone(), vec![])?;
            csv_reader.as_df()?
        };

        // Important that this rechunk is never done in parallel.
        // As that leads to great memory overhead.
        if rechunk && df.n_chunks() > 1 {
            if low_memory {
                df.as_single_chunk();
            } else {
                df.as_single_chunk_par();
            }
        }

        #[cfg(feature = "temporal")]
        {
            let parse_options = self.options.get_parse_options();

            // only needed until we also can parse time columns in place
            if parse_options.try_parse_dates {
                // determine the schema that's given by the user. That should not be changed
                let fixed_schema = match (schema_overwrite, self.options.dtype_overwrite) {
                    (Some(schema), _) => schema,
                    (None, Some(dtypes)) => {
                        let schema = dtypes
                            .iter()
                            .zip(df.get_column_names())
                            .map(|(dtype, name)| Field::new(name, dtype.clone()))
                            .collect::<Schema>();

                        Arc::new(schema)
                    },
                    _ => Arc::default(),
                };
                df = parse_dates(df, &fixed_schema)
            }
        }

        Ok(df)
    }
}

#[cfg(feature = "temporal")]
fn parse_dates(mut df: DataFrame, fixed_schema: &Schema) -> DataFrame {
    use polars_core::POOL;

    let cols = unsafe { std::mem::take(df.get_columns_mut()) }
        .into_par_iter()
        .map(|s| {
            match s.dtype() {
                DataType::String => {
                    let ca = s.str().unwrap();
                    // don't change columns that are in the fixed schema.
                    if fixed_schema.index_of(s.name()).is_some() {
                        return s;
                    }

                    #[cfg(feature = "dtype-time")]
                    if let Ok(ca) = ca.as_time(None, false) {
                        return ca.into_series();
                    }
                    s
                },
                _ => s,
            }
        });
    let cols = POOL.install(|| cols.collect::<Vec<_>>());

    unsafe { DataFrame::new_no_checks(cols) }
}
