use crate::file::*;
use crate::prelude::*;
use crate::series::JsSeries;
use napi::JsUnknown;
use polars::frame::row::{infer_schema, Row};
use polars::io::RowCount;
use std::borrow::Borrow;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor};

#[napi]
#[repr(transparent)]
#[derive(Clone)]
pub struct JsDataFrame {
    pub(crate) df: DataFrame,
}

impl JsDataFrame {
    pub(crate) fn new(df: DataFrame) -> JsDataFrame {
        JsDataFrame { df }
    }
}
impl From<DataFrame> for JsDataFrame {
    fn from(s: DataFrame) -> JsDataFrame {
        JsDataFrame::new(s)
    }
}

pub(crate) fn to_series_collection(ps: Array) -> Vec<Series> {
    let len = ps.len();
    (0..len)
        .map(|idx| {
            let item: &JsSeries = ps.get(idx).unwrap().unwrap();
            item.series.clone()
        })
        .collect()
}
pub(crate) fn to_jsseries_collection(s: Vec<Series>) -> Vec<JsSeries> {
    let mut s = std::mem::ManuallyDrop::new(s);

    let p = s.as_mut_ptr() as *mut JsSeries;
    let len = s.len();
    let cap = s.capacity();

    unsafe { Vec::from_raw_parts(p, len, cap) }
}

#[napi(object)]
pub struct ReadCsvOptions {
    pub infer_schema_length: Option<u32>,
    pub chunk_size: u32,
    pub has_header: bool,
    pub ignore_errors: bool,
    pub n_rows: Option<u32>,
    pub skip_rows: u32,
    pub sep: String,
    pub rechunk: bool,
    pub columns: Option<Vec<String>>,
    pub encoding: String,
    pub n_threads: Option<u32>,
    pub null_values: Option<Wrap<NullValues>>,
    pub path: Option<String>,
    pub low_memory: bool,
    pub comment_char: Option<String>,
    pub quote_char: Option<String>,
    pub parse_dates: bool,
    pub skip_rows_after_header: u32,
    pub row_count: Option<JsRowCount>,
}

#[napi]
pub fn read_csv(
    path_or_buffer: Either<String, Buffer>,
    options: ReadCsvOptions,
) -> napi::Result<JsDataFrame> {
    let infer_schema_length = options.infer_schema_length.map(|i| i as usize);
    let n_threads = options.n_threads.map(|i| i as usize);
    let n_rows = options.n_rows.map(|i| i as usize);
    let skip_rows = options.skip_rows as usize;
    let chunk_size = options.chunk_size as usize;
    let null_values = options.null_values.map(|w| w.0);
    let comment_char = options.comment_char.map(|s| s.as_bytes()[0]);
    let row_count = options.row_count.map(RowCount::from);

    let quote_char = if let Some(s) = options.quote_char {
        if s.is_empty() {
            None
        } else {
            Some(s.as_bytes()[0])
        }
    } else {
        None
    };

    let encoding = match options.encoding.as_ref() {
        "utf8" => CsvEncoding::Utf8,
        "utf8-lossy" => CsvEncoding::LossyUtf8,
        e => return Err(JsPolarsErr::Other(format!("encoding not {} not implemented.", e)).into()),
    };
    let df = match path_or_buffer {
        Either::A(path) => CsvReader::from_path(path)
            .expect("unable to read file")
            .infer_schema(infer_schema_length)
            .has_header(options.has_header)
            .with_n_rows(n_rows)
            .with_delimiter(options.sep.as_bytes()[0])
            .with_skip_rows(skip_rows)
            .with_ignore_parser_errors(options.ignore_errors)
            .with_rechunk(options.rechunk)
            .with_chunk_size(chunk_size)
            .with_encoding(encoding)
            .with_columns(options.columns)
            .with_n_threads(n_threads)
            .low_memory(options.low_memory)
            .with_comment_char(comment_char)
            .with_null_values(null_values)
            .with_parse_dates(options.parse_dates)
            .with_quote_char(quote_char)
            .with_row_count(row_count)
            .finish()
            .map_err(JsPolarsErr::from)?,
        Either::B(buffer) => {
            let cursor = Cursor::new(buffer.as_ref());
            CsvReader::new(cursor)
                .infer_schema(infer_schema_length)
                .has_header(options.has_header)
                .with_n_rows(n_rows)
                .with_delimiter(options.sep.as_bytes()[0])
                .with_skip_rows(skip_rows)
                .with_ignore_parser_errors(options.ignore_errors)
                .with_rechunk(options.rechunk)
                .with_chunk_size(chunk_size)
                .with_encoding(encoding)
                .with_columns(options.columns)
                .with_n_threads(n_threads)
                .low_memory(options.low_memory)
                .with_comment_char(comment_char)
                .with_null_values(null_values)
                .with_parse_dates(options.parse_dates)
                .with_quote_char(quote_char)
                .with_row_count(row_count)
                .finish()
                .map_err(JsPolarsErr::from)?
        }
    };
    Ok(df.into())
}
#[napi(object)]
pub struct ReadJsonOptions {
    pub infer_schema_length: Option<u32>,
    pub batch_size: Option<u32>,
    pub format: Option<String>,
}

#[napi(object)]
pub struct WriteJsonOptions {
    pub format: String,
}

#[napi]
pub fn read_json_lines(
    path_or_buffer: Either<String, Buffer>,
    options: ReadJsonOptions,
) -> napi::Result<JsDataFrame> {
    let infer_schema_length = options.infer_schema_length.unwrap_or(100) as usize;
    let batch_size = options.batch_size.unwrap_or(10000) as usize;

    let df = match path_or_buffer {
        Either::A(path) => JsonLineReader::from_path(path)
            .expect("unable to read file")
            .infer_schema_len(Some(infer_schema_length))
            .finish()
            .map_err(JsPolarsErr::from)?,
        Either::B(buf) => {
            let cursor = Cursor::new(buf.as_ref());
            JsonLineReader::new(cursor)
                .infer_schema_len(Some(infer_schema_length))
                .finish()
                .map_err(JsPolarsErr::from)?
        }
    };
    Ok(df.into())
}
#[napi]
pub fn read_json(
    path_or_buffer: Either<String, Buffer>,
    options: ReadJsonOptions,
) -> napi::Result<JsDataFrame> {
    let infer_schema_length = options.infer_schema_length.unwrap_or(100) as usize;
    let batch_size = options.batch_size.unwrap_or(10000) as usize;
    let format: JsonFormat = options
        .format
        .map(|s| match s.as_ref() {
            "lines" => Ok(JsonFormat::JsonLines),
            "json" => Ok(JsonFormat::Json),
            _ => {
                return Err(napi::Error::from_reason(
                    "format must be 'json' or `lines'".to_owned(),
                ))
            }
        })
        .unwrap()?;
    let df = match path_or_buffer {
        Either::A(path) => {
            let f = File::open(&path)?;
            let reader = BufReader::new(f);
            JsonReader::new(reader)
                .infer_schema_len(Some(infer_schema_length))
                .with_batch_size(batch_size)
                .with_json_format(format)
                .finish()
                .map_err(JsPolarsErr::from)?
        }
        Either::B(buf) => {
            let cursor = Cursor::new(buf.as_ref());
            JsonReader::new(cursor)
                .infer_schema_len(Some(infer_schema_length))
                .with_batch_size(batch_size)
                .with_json_format(format)
                .finish()
                .map_err(JsPolarsErr::from)?
        }
    };
    Ok(df.into())
}

#[napi(object)]
pub struct ReadParquetOptions {
    pub columns: Option<Vec<String>>,
    pub projection: Option<Vec<i64>>,
    pub n_rows: Option<i64>,
    pub parallel: Option<bool>,
    pub row_count: Option<JsRowCount>,
}

#[napi]
pub fn read_parquet(
    path_or_buffer: Either<String, Buffer>,
    options: ReadParquetOptions,
) -> napi::Result<JsDataFrame> {
    let columns = options.columns;

    let projection = options
        .projection
        .map(|projection| projection.into_iter().map(|p| p as usize).collect());
    let row_count = options.row_count.map(|rc| rc.into());
    let n_rows = options.n_rows.map(|nr| nr as usize);
    let parallel = options.parallel.unwrap_or(true);

    let result = match path_or_buffer {
        Either::A(path) => {
            let f = File::open(&path)?;
            let reader = BufReader::new(f);
            ParquetReader::new(reader)
                .with_projection(projection)
                .with_columns(columns)
                .read_parallel(parallel)
                .with_n_rows(n_rows)
                .with_row_count(row_count)
                .finish()
        }
        Either::B(buf) => {
            let cursor = Cursor::new(buf.as_ref());
            ParquetReader::new(cursor)
                .with_projection(projection)
                .with_columns(columns)
                .read_parallel(parallel)
                .with_n_rows(n_rows)
                .with_row_count(row_count)
                .finish()
        }
    };
    let df = result.map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
}

#[napi(object)]
pub struct ReadIpcOptions {
    pub columns: Option<Vec<String>>,
    pub projection: Option<Vec<i64>>,
    pub n_rows: Option<i64>,
    pub row_count: Option<JsRowCount>,
}

#[napi]
pub fn read_ipc(
    path_or_buffer: Either<String, Buffer>,
    options: ReadIpcOptions,
) -> napi::Result<JsDataFrame> {
    let columns = options.columns;
    let projection = options
        .projection
        .map(|projection| projection.into_iter().map(|p| p as usize).collect());
    let row_count = options.row_count.map(|rc| rc.into());
    let n_rows = options.n_rows.map(|nr| nr as usize);

    let result = match path_or_buffer {
        Either::A(path) => {
            let f = File::open(&path)?;
            let reader = BufReader::new(f);
            IpcReader::new(reader)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .with_row_count(row_count)
                .finish()
        }
        Either::B(buf) => {
            let cursor = Cursor::new(buf.as_ref());
            IpcReader::new(cursor)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .with_row_count(row_count)
                .finish()
        }
    };
    let df = result.map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
}

#[napi(object)]
pub struct ReadAvroOptions {
    pub columns: Option<Vec<String>>,
    pub projection: Option<Vec<i64>>,
    pub n_rows: Option<i64>,
}

#[napi]
pub fn read_avro(
    path_or_buffer: Either<String, Buffer>,
    options: ReadAvroOptions,
) -> napi::Result<JsDataFrame> {
    use polars::io::avro::AvroReader;
    let columns = options.columns;
    let projection = options
        .projection
        .map(|projection| projection.into_iter().map(|p| p as usize).collect());
    let n_rows = options.n_rows.map(|nr| nr as usize);

    let result = match path_or_buffer {
        Either::A(path) => {
            let f = File::open(&path)?;
            let reader = BufReader::new(f);
            AvroReader::new(reader)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .finish()
        }
        Either::B(buf) => {
            let cursor = Cursor::new(buf.as_ref());
            AvroReader::new(cursor)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .finish()
        }
    };
    let df = result.map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
}

#[napi]
pub fn from_rows(
    rows: Array,
    schema: Option<Wrap<Schema>>,
    infer_schema_length: Option<u32>,
    env: Env,
) -> napi::Result<JsDataFrame> {
    let schema = match schema {
        Some(s) => s.0,
        None => {
            let infer_schema_length = infer_schema_length.unwrap_or(100) as usize;
            let pairs = obj_to_pairs(&rows, infer_schema_length);
            infer_schema(pairs, infer_schema_length)
        }
    };
    let len = rows.len();
    let it: Vec<Row> = (0..len)
        .into_iter()
        .map(|idx| {
            let obj = rows
                .get::<Object>(idx as u32)
                .unwrap_or(None)
                .unwrap_or(env.create_object().unwrap());
            Row(schema
                .iter_fields()
                .map(|fld| {
                    let dtype = fld.data_type().clone();
                    let key = fld.name();
                    if let Ok(unknown) = obj.get(key) {
                        let av = match unknown {
                            Some(unknown) => unsafe {
                                coerce_js_anyvalue(unknown, dtype).unwrap_or(AnyValue::Null)
                            },
                            None => AnyValue::Null,
                        };
                        av
                    } else {
                        AnyValue::Null
                    }
                })
                .collect())
        })
        .collect();
    let df = DataFrame::from_rows_and_schema(&it, &schema).map_err(JsPolarsErr::from)?;
    Ok(df.into())
}

#[napi]
impl JsDataFrame {
    #[napi]
    pub fn to_js(&self, env: Env) -> napi::Result<napi::JsUnknown> {
        env.to_js_value(&self.df)
    }

    #[napi]
    pub fn serialize(&self, format: String) -> napi::Result<Buffer> {
        let buf = match format.as_ref() {
            "bincode" => bincode::serialize(&self.df)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            "json" => serde_json::to_vec(&self.df)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            _ => {
                return Err(napi::Error::from_reason(
                    "unexpected format. \n supportd options are 'json', 'bincode'".to_owned(),
                ))
            }
        };
        Ok(Buffer::from(buf))
    }

    #[napi(factory)]
    pub fn deserialize(buf: Buffer, format: String) -> napi::Result<JsDataFrame> {
        let df: DataFrame = match format.as_ref() {
            "bincode" => bincode::deserialize(&buf)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            "json" => serde_json::from_slice(&buf)
                .map_err(|err| napi::Error::from_reason(format!("{:?}", err)))?,
            _ => {
                return Err(napi::Error::from_reason(
                    "unexpected format. \n supportd options are 'json', 'bincode'".to_owned(),
                ))
            }
        };
        Ok(df.into())
    }
    #[napi(constructor)]
    pub fn from_columns(columns: Array) -> napi::Result<JsDataFrame> {
        let len = columns.len();
        let cols: Vec<Series> = (0..len)
            .map(|idx| {
                let item: &JsSeries = columns.get(idx).unwrap().unwrap();
                item.series.clone()
            })
            .collect();

        let df = DataFrame::new(cols).map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }

    #[napi]
    pub fn estimated_size(&self) -> u32 {
        self.df.estimated_size() as u32
    }

    #[napi]
    pub fn to_string(&self) -> String {
        format!("{:?}", self.df)
    }

    #[napi]
    pub fn add(&self, s: &JsSeries) -> napi::Result<JsDataFrame> {
        let df = (&self.df + &s.series).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn sub(&self, s: &JsSeries) -> napi::Result<JsDataFrame> {
        let df = (&self.df - &s.series).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn div(&self, s: &JsSeries) -> napi::Result<JsDataFrame> {
        let df = (&self.df / &s.series).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn mul(&self, s: &JsSeries) -> napi::Result<JsDataFrame> {
        let df = (&self.df * &s.series).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn rem(&self, s: &JsSeries) -> napi::Result<JsDataFrame> {
        let df = (&self.df % &s.series).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn add_df(&self, s: &JsDataFrame) -> napi::Result<JsDataFrame> {
        let df = (&self.df + &s.df).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn sub_df(&self, s: &JsDataFrame) -> napi::Result<JsDataFrame> {
        let df = (&self.df - &s.df).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn div_df(&self, s: &JsDataFrame) -> napi::Result<JsDataFrame> {
        let df = (&self.df / &s.df).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn mul_df(&self, s: &JsDataFrame) -> napi::Result<JsDataFrame> {
        let df = (&self.df * &s.df).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn rem_df(&self, s: &JsDataFrame) -> napi::Result<JsDataFrame> {
        let df = (&self.df % &s.df).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn rechunk(&mut self) -> JsDataFrame {
        self.df.agg_chunks().into()
    }
    #[napi]
    pub fn fill_null(&self, strategy: Wrap<FillNullStrategy>) -> napi::Result<JsDataFrame> {
        let df = self.df.fill_null(strategy.0).map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }
    #[napi]
    pub fn join(
        &self,
        other: &JsDataFrame,
        left_on: Vec<&str>,
        right_on: Vec<&str>,
        how: String,
        suffix: String,
    ) -> napi::Result<JsDataFrame> {
        let how = match how.as_ref() {
            "left" => JoinType::Left,
            "inner" => JoinType::Inner,
            "outer" => JoinType::Outer,
            "semi" => JoinType::Semi,
            "anti" => JoinType::Anti,
            "asof" => JoinType::AsOf(AsOfOptions {
                strategy: AsofStrategy::Backward,
                left_by: None,
                right_by: None,
                tolerance: None,
                tolerance_str: None,
            }),
            "cross" => JoinType::Cross,
            _ => panic!("not supported"),
        };

        let df = self
            .df
            .join(&other.df, left_on, right_on, how, Some(suffix))
            .map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }

    #[napi]
    pub fn get_columns(&self) -> Vec<JsSeries> {
        let cols = self.df.get_columns().clone();
        to_jsseries_collection(cols)
    }

    /// Get column names
    #[napi(getter)]
    pub fn columns(&self) -> Vec<&str> {
        self.df.get_column_names()
    }

    #[napi(setter, js_name = "columns")]
    pub fn set_columns(&mut self, names: Vec<&str>) -> napi::Result<()> {
        self.df
            .set_column_names(&names)
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }

    #[napi]
    pub fn with_column(&mut self, s: &JsSeries) -> napi::Result<JsDataFrame> {
        let mut df = self.df.clone();
        df.with_column(s.series.clone())
            .map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    /// Get datatypes
    #[napi]
    pub fn dtypes(&self) -> Vec<JsDataType> {
        self.df.iter().map(|s| s.dtype().into()).collect()
    }
    #[napi]
    pub fn n_chunks(&self) -> napi::Result<u32> {
        let n = self.df.n_chunks().map_err(JsPolarsErr::from)?;
        Ok(n as u32)
    }

    #[napi(getter)]
    pub fn shape(&self) -> Shape {
        self.df.shape().into()
    }
    #[napi(getter)]
    pub fn height(&self) -> i64 {
        self.df.height() as i64
    }
    #[napi(getter)]
    pub fn width(&self) -> i64 {
        self.df.width() as i64
    }
    #[napi]
    pub fn hstack_mut(&mut self, columns: Array) -> napi::Result<()> {
        let columns = to_series_collection(columns);
        self.df.hstack_mut(&columns).map_err(JsPolarsErr::from)?;
        Ok(())
    }
    #[napi]
    pub fn hstack(&self, columns: Array) -> napi::Result<JsDataFrame> {
        let columns = to_series_collection(columns);
        let df = self.df.hstack(&columns).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }
    #[napi]
    pub fn extend(&mut self, df: &JsDataFrame) -> napi::Result<()> {
        self.df.extend(&df.df).map_err(JsPolarsErr::from)?;
        Ok(())
    }
    #[napi]
    pub fn vstack_mut(&mut self, df: &JsDataFrame) -> napi::Result<()> {
        self.df.vstack_mut(&df.df).map_err(JsPolarsErr::from)?;
        Ok(())
    }
    #[napi]
    pub fn vstack(&mut self, df: &JsDataFrame) -> napi::Result<JsDataFrame> {
        let df = self.df.vstack(&df.df).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }
    #[napi]
    pub fn drop_in_place(&mut self, name: String) -> napi::Result<JsSeries> {
        let s = self.df.drop_in_place(&name).map_err(JsPolarsErr::from)?;
        Ok(JsSeries { series: s })
    }
    #[napi]
    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> napi::Result<JsDataFrame> {
        let df = self
            .df
            .drop_nulls(subset.as_ref().map(|s| s.as_ref()))
            .map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn drop(&self, name: String) -> napi::Result<JsDataFrame> {
        let df = self.df.drop(&name).map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }
    #[napi]
    pub fn select_at_idx(&self, idx: i64) -> Option<JsSeries> {
        self.df
            .select_at_idx(idx as usize)
            .map(|s| JsSeries::new(s.clone()))
    }

    #[napi]
    pub fn find_idx_by_name(&self, name: String) -> Option<i64> {
        self.df.find_idx_by_name(&name).map(|i| i as i64)
    }
    #[napi]
    pub fn column(&self, name: String) -> napi::Result<JsSeries> {
        let series = self
            .df
            .column(&name)
            .map(|s| JsSeries::new(s.clone()))
            .map_err(JsPolarsErr::from)?;
        Ok(series)
    }
    #[napi]
    pub fn select(&self, selection: Vec<&str>) -> napi::Result<JsDataFrame> {
        let df = self.df.select(&selection).map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }
    #[napi]
    pub fn filter(&self, mask: &JsSeries) -> napi::Result<JsDataFrame> {
        let filter_series = &mask.series;
        if let Ok(ca) = filter_series.bool() {
            let df = self.df.filter(ca).map_err(JsPolarsErr::from)?;
            Ok(JsDataFrame::new(df))
        } else {
            Err(napi::Error::from_reason(
                "Expected a boolean mask".to_owned(),
            ))
        }
    }
    #[napi]
    pub fn take(&self, indices: Vec<u32>) -> napi::Result<JsDataFrame> {
        let indices = UInt32Chunked::from_vec("", indices);
        let df = self.df.take(&indices).map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }
    #[napi]
    pub fn take_with_series(&self, indices: &JsSeries) -> napi::Result<JsDataFrame> {
        let idx = indices.series.u32().map_err(JsPolarsErr::from)?;
        let df = self.df.take(idx).map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }
    #[napi]
    pub fn sort(
        &self,
        by_column: String,
        reverse: bool,
        nulls_last: bool,
    ) -> napi::Result<JsDataFrame> {
        let df = self
            .df
            .sort_with_options(
                &by_column,
                SortOptions {
                    descending: reverse,
                    nulls_last,
                },
            )
            .map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }
    #[napi]
    pub fn sort_in_place(&mut self, by_column: String, reverse: bool) -> napi::Result<()> {
        self.df
            .sort_in_place([&by_column], reverse)
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }
    #[napi]
    pub fn replace(&mut self, column: String, new_col: &JsSeries) -> napi::Result<()> {
        self.df
            .replace(&column, new_col.series.clone())
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }

    #[napi]
    pub fn rename(&mut self, column: String, new_col: String) -> napi::Result<()> {
        self.df
            .rename(&column, &new_col)
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }

    #[napi]
    pub fn replace_at_idx(&mut self, index: f64, new_col: &JsSeries) -> napi::Result<()> {
        self.df
            .replace_at_idx(index as usize, new_col.series.clone())
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }

    #[napi]
    pub fn insert_at_idx(&mut self, index: f64, new_col: &JsSeries) -> napi::Result<()> {
        self.df
            .insert_at_idx(index as usize, new_col.series.clone())
            .map_err(JsPolarsErr::from)?;
        Ok(())
    }

    #[napi]
    pub fn slice(&self, offset: i64, length: i64) -> JsDataFrame {
        let df = self.df.slice(offset as i64, length as usize);
        df.into()
    }

    #[napi]
    pub fn head(&self, length: Option<i64>) -> JsDataFrame {
        let length = length.map(|l| l as usize);
        let df = self.df.head(length);
        JsDataFrame::new(df)
    }
    #[napi]
    pub fn tail(&self, length: Option<i64>) -> JsDataFrame {
        let length = length.map(|l| l as usize);
        let df = self.df.tail(length);
        JsDataFrame::new(df)
    }
    #[napi]
    pub fn is_unique(&self) -> napi::Result<JsSeries> {
        let mask = self.df.is_unique().map_err(JsPolarsErr::from)?;
        Ok(mask.into_series().into())
    }
    #[napi]
    pub fn is_duplicated(&self) -> napi::Result<JsSeries> {
        let mask = self.df.is_duplicated().map_err(JsPolarsErr::from)?;
        Ok(mask.into_series().into())
    }
    #[napi]
    pub fn frame_equal(&self, other: &JsDataFrame, null_equal: bool) -> bool {
        if null_equal {
            self.df.frame_equal_missing(&other.df)
        } else {
            self.df.frame_equal(&other.df)
        }
    }
    #[napi]
    pub fn with_row_count(&self, name: String, offset: Option<u32>) -> napi::Result<JsDataFrame> {
        let df = self
            .df
            .with_row_count(&name, offset)
            .map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }
    #[napi]
    pub fn groupby(
        &self,
        by: Vec<&str>,
        select: Option<Vec<String>>,
        agg: String,
    ) -> napi::Result<JsDataFrame> {
        let gb = self.df.groupby(&by).map_err(JsPolarsErr::from)?;
        let selection = match select.as_ref() {
            Some(s) => gb.select(s),
            None => gb,
        };
        finish_groupby(selection, &agg)
    }

    #[napi]
    pub fn pivot(
        &self,
        by: Vec<String>,
        pivot_column: Vec<String>,
        values_column: Vec<String>,
        agg: String,
    ) -> napi::Result<JsDataFrame> {
        let mut gb = self.df.groupby(&by).map_err(JsPolarsErr::from)?;
        let pivot = gb.pivot(pivot_column, values_column);
        let df = match agg.as_ref() {
            "first" => pivot.first(),
            "min" => pivot.min(),
            "max" => pivot.max(),
            "mean" => pivot.mean(),
            "median" => pivot.median(),
            "sum" => pivot.sum(),
            "count" => pivot.count(),
            "last" => pivot.last(),
            a => Err(PolarsError::ComputeError(
                format!("agg fn {} does not exists", a).into(),
            )),
        };
        let df = df.map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }
    #[napi]
    pub fn clone(&self) -> JsDataFrame {
        JsDataFrame::new(self.df.clone())
    }
    #[napi]
    pub fn melt(
        &self,
        id_vars: Vec<String>,
        value_vars: Vec<String>,
        value_name: Option<String>,
        variable_name: Option<String>,
    ) -> napi::Result<JsDataFrame> {
        let args = MeltArgs {
            id_vars,
            value_vars,
            value_name,
            variable_name,
        };

        let df = self.df.melt2(args).map_err(JsPolarsErr::from)?;
        Ok(JsDataFrame::new(df))
    }

    #[napi]
    pub fn partition_by(
        &self,
        groups: Vec<String>,
        stable: bool,
    ) -> napi::Result<Vec<JsDataFrame>> {
        let out = if stable {
            self.df.partition_by_stable(groups)
        } else {
            self.df.partition_by(groups)
        }
        .map_err(JsPolarsErr::from)?;
        // Safety:
        // Repr mem layout
        Ok(unsafe { std::mem::transmute::<Vec<DataFrame>, Vec<JsDataFrame>>(out) })
    }

    #[napi]
    pub fn shift(&self, periods: i64) -> JsDataFrame {
        self.df.shift(periods).into()
    }
    #[napi]
    pub fn unique(
        &self,
        maintain_order: bool,
        subset: Option<Vec<String>>,
        keep: Wrap<UniqueKeepStrategy>,
    ) -> napi::Result<JsDataFrame> {
        let subset = subset.as_ref().map(|v| v.as_ref());
        let df = match maintain_order {
            true => self.df.unique_stable(subset, keep.0),
            false => self.df.unique(subset, keep.0),
        }
        .map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn lazy(&self) -> crate::lazy::dataframe::JsLazyFrame {
        self.df.clone().lazy().into()
    }

    #[napi]
    pub fn max(&self) -> JsDataFrame {
        self.df.max().into()
    }
    #[napi]
    pub fn min(&self) -> JsDataFrame {
        self.df.min().into()
    }
    #[napi]
    pub fn sum(&self) -> JsDataFrame {
        self.df.sum().into()
    }
    #[napi]
    pub fn mean(&self) -> JsDataFrame {
        self.df.mean().into()
    }
    #[napi]
    pub fn std(&self) -> JsDataFrame {
        self.df.std().into()
    }
    #[napi]
    pub fn var(&self) -> JsDataFrame {
        self.df.var().into()
    }
    #[napi]
    pub fn median(&self) -> JsDataFrame {
        self.df.median().into()
    }

    #[napi]
    pub fn hmean(&self, null_strategy: Wrap<NullStrategy>) -> napi::Result<Option<JsSeries>> {
        let s = self.df.hmean(null_strategy.0).map_err(JsPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }
    #[napi]
    pub fn hmax(&self) -> napi::Result<Option<JsSeries>> {
        let s = self.df.hmax().map_err(JsPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    #[napi]
    pub fn hmin(&self) -> napi::Result<Option<JsSeries>> {
        let s = self.df.hmin().map_err(JsPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    #[napi]
    pub fn hsum(&self, null_strategy: Wrap<NullStrategy>) -> napi::Result<Option<JsSeries>> {
        let s = self.df.hsum(null_strategy.0).map_err(JsPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }
    #[napi]
    pub fn quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
    ) -> napi::Result<JsDataFrame> {
        let df = self
            .df
            .quantile(quantile, interpolation.0)
            .map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn to_dummies(&self) -> napi::Result<JsDataFrame> {
        let df = self.df.to_dummies().map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }

    #[napi]
    pub fn null_count(&self) -> JsDataFrame {
        let df = self.df.null_count();
        df.into()
    }
    #[napi]
    pub fn shrink_to_fit(&mut self) {
        self.df.shrink_to_fit();
    }
    #[napi]
    pub fn hash_rows(
        &self,
        k0: Wrap<u64>,
        k1: Wrap<u64>,
        k2: Wrap<u64>,
        k3: Wrap<u64>,
    ) -> napi::Result<JsSeries> {
        let hb = ahash::RandomState::with_seeds(k0.0, k1.0, k2.0, k3.0);
        let hash = self.df.hash_rows(Some(hb)).map_err(JsPolarsErr::from)?;
        Ok(hash.into_series().into())
    }

    #[napi]
    pub fn transpose(&self, include_header: bool, names: String) -> napi::Result<JsDataFrame> {
        let mut df = self.df.transpose().map_err(JsPolarsErr::from)?;
        if include_header {
            let s = Utf8Chunked::from_iter_values(
                &names,
                self.df.get_columns().iter().map(|s| s.name()),
            )
            .into_series();
            df.insert_at_idx(0, s).unwrap();
        }
        Ok(df.into())
    }

    #[napi]
    pub fn sample_n(
        &self,
        n: i64,
        with_replacement: bool,
        seed: Option<i64>,
    ) -> napi::Result<JsDataFrame> {
        todo!()
        // let df = self
        //     .df
        //     .sample_n(n as usize, with_replacement, seed.map(|s| s as u64))
        //     .map_err(JsPolarsErr::from)?;
        // Ok(df.into())
    }

    #[napi]
    pub fn sample_frac(
        &self,
        frac: f64,
        with_replacement: bool,
        seed: Option<i64>,
    ) -> napi::Result<JsDataFrame> {
        todo!()
        // let df = self
        //     .df
        //     .sample_frac(frac, with_replacement, seed.map(|s| s as u64))
        //     .map_err(JsPolarsErr::from)?;
        // Ok(df.into())
    }

    #[napi]
    pub fn upsample(
        &self,
        by: Vec<String>,
        index_column: String,
        every: String,
        offset: String,
        stable: bool,
    ) -> napi::Result<JsDataFrame> {
        let out = if stable {
            self.df.upsample_stable(
                by,
                &index_column,
                Duration::parse(&every),
                Duration::parse(&offset),
            )
        } else {
            self.df.upsample(
                by,
                &index_column,
                Duration::parse(&every),
                Duration::parse(&offset),
            )
        };
        let out = out.map_err(JsPolarsErr::from)?;
        Ok(out.into())
    }
    #[napi]
    pub fn to_struct(&self, name: String) -> JsSeries {
        let s = self.df.clone().into_struct(&name);
        s.into_series().into()
    }
    #[napi]
    pub fn unnest(&self, names: Vec<String>) -> napi::Result<JsDataFrame> {
        let df = self.df.unnest(names).map_err(JsPolarsErr::from)?;
        Ok(df.into())
    }
    #[napi]
    pub fn to_row(&self, idx: f64, env: Env) -> napi::Result<Array> {
        let idx = idx as i64;

        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };

        let width = self.df.width();
        let mut row = env.create_array(width as u32)?;

        for (i, col) in self.df.get_columns().iter().enumerate() {
            let val = col.get(idx);
            row.set(i as u32, Wrap(val))?;
        }
        Ok(row)
    }

    #[napi]
    pub fn to_rows(&self, env: Env) -> napi::Result<Array> {
        let (height, width) = self.df.shape();

        let mut rows = env.create_array(height as u32)?;
        for idx in 0..height {
            let mut row = env.create_array(width as u32)?;
            for (i, col) in self.df.get_columns().iter().enumerate() {
                let val = col.get(idx);
                row.set(i as u32, Wrap(val))?;
            }
            rows.set(idx as u32, row)?;
        }
        Ok(rows)
    }
    #[napi]
    pub fn to_rows_cb(&self, callback: napi::JsFunction, env: Env) -> napi::Result<()> {
        use napi::threadsafe_function::*;
        use polars_core::utils::rayon::prelude::*;
        let (height, _) = self.df.shape();
        let tsfn: ThreadsafeFunction<
            Either<Vec<JsAnyValue>, napi::JsNull>,
            ErrorStrategy::CalleeHandled,
        > = callback.create_threadsafe_function(
            0,
            |ctx: ThreadSafeCallContext<Either<Vec<JsAnyValue>, napi::JsNull>>| Ok(vec![ctx.value]),
        )?;

        polars_core::POOL.install(|| {
            (0..height).into_par_iter().for_each(|idx| {
                let tsfn = tsfn.clone();
                let values = self
                    .df
                    .get_columns()
                    .iter()
                    .map(|s| {
                        let av: JsAnyValue = s.get(idx).into();
                        av
                    })
                    .collect::<Vec<_>>();

                tsfn.call(
                    Ok(Either::A(values)),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            });
        });
        tsfn.call(
            Ok(Either::B(env.get_null().unwrap())),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }
    #[napi]
    pub fn to_row_obj(&self, idx: Either<i64, f64>, env: Env) -> napi::Result<Object> {
        let idx = match idx {
            Either::A(a) => a,
            Either::B(b) => b as i64,
        };

        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };

        let mut row = env.create_object()?;

        for col in self.df.get_columns() {
            let key = col.name();
            let val = col.get(idx);
            row.set(key, Wrap(val))?;
        }
        Ok(row)
    }
    #[napi]
    pub fn to_objects(&self, env: Env) -> napi::Result<Array> {
        let (height, _) = self.df.shape();

        let mut rows = env.create_array(height as u32)?;
        for idx in 0..height {
            let mut row = env.create_object()?;
            for col in self.df.get_columns() {
                let key = col.name();
                let val = col.get(idx);
                row.set(key, Wrap(val))?;
            }
            rows.set(idx as u32, row)?;
        }
        Ok(rows)
    }
    #[napi]
    pub fn to_objects_cb(&self, callback: napi::JsFunction, env: Env) -> napi::Result<()> {
        use napi::threadsafe_function::*;
        use polars_core::utils::rayon::prelude::*;
        use std::collections::HashMap;
        let (height, _) = self.df.shape();
        let tsfn: ThreadsafeFunction<
            Either<HashMap<String, JsAnyValue>, napi::JsNull>,
            ErrorStrategy::CalleeHandled,
        > = callback.create_threadsafe_function(
            0,
            |ctx: ThreadSafeCallContext<Either<HashMap<String, JsAnyValue>, napi::JsNull>>| {
                Ok(vec![ctx.value])
            },
        )?;

        polars_core::POOL.install(|| {
            (0..height).into_par_iter().for_each(|idx| {
                let tsfn = tsfn.clone();
                let values = self
                    .df
                    .get_columns()
                    .iter()
                    .map(|s| {
                        let key = s.name().to_owned();
                        let av: JsAnyValue = s.get(idx).into();
                        (key, av)
                    })
                    .collect::<HashMap<_, _>>();

                tsfn.call(
                    Ok(Either::A(values)),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            });
        });
        tsfn.call(
            Ok(Either::B(env.get_null().unwrap())),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    #[napi]
    pub fn write_csv(
        &mut self,
        path_or_buffer: Either<String, napi::JsObject>,
        options: WriteCsvOptions,
        env: Env,
    ) -> napi::Result<()> {
        let has_header = options.has_header.unwrap_or(true);
        let sep = options.sep.unwrap_or(",".to_owned());
        let sep = sep.as_bytes()[0];
        let quote = options.quote.unwrap_or(",".to_owned());
        let quote = quote.as_bytes()[0];

        match path_or_buffer {
            Either::A(path) => {
                let f = std::fs::File::create(path).unwrap();
                let f = BufWriter::new(f);
                CsvWriter::new(f)
                    .has_header(has_header)
                    .with_delimiter(sep)
                    .with_quoting_char(quote)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
            Either::B(inner) => {
                let writeable = JsWriteStream { inner, env: &env };

                CsvWriter::new(writeable)
                    .has_header(has_header)
                    .with_delimiter(sep)
                    .with_quoting_char(quote)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
        };
        Ok(())
    }

    #[napi]
    pub fn write_parquet(
        &mut self,
        path_or_buffer: Either<String, napi::JsObject>,
        compression: Wrap<ParquetCompression>,
        env: Env,
    ) -> napi::Result<()> {
        let compression = compression.0;

        match path_or_buffer {
            Either::A(path) => {
                let f = std::fs::File::create(path).unwrap();
                let f = BufWriter::new(f);
                ParquetWriter::new(f)
                    .with_compression(compression)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
            Either::B(inner) => {
                let writeable = JsWriteStream { inner, env: &env };

                ParquetWriter::new(writeable)
                    .with_compression(compression)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
        };
        Ok(())
    }
    #[napi]
    pub fn write_ipc(
        &mut self,
        path_or_buffer: Either<String, napi::JsObject>,
        compression: Wrap<Option<IpcCompression>>,
        env: Env,
    ) -> napi::Result<()> {
        let compression = compression.0;

        match path_or_buffer {
            Either::A(path) => {
                let f = std::fs::File::create(path).unwrap();
                let f = BufWriter::new(f);
                IpcWriter::new(f)
                    .with_compression(compression)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
            Either::B(inner) => {
                let writeable = JsWriteStream { inner, env: &env };
                IpcWriter::new(writeable)
                    .with_compression(compression)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
        };
        Ok(())
    }
    #[napi]
    pub fn write_json(
        &mut self,
        path_or_buffer: Either<String, napi::JsObject>,
        options: WriteJsonOptions,
        env: Env,
    ) -> napi::Result<()> {
        let json_format = options.format;
        let json_format = match json_format.as_ref() {
            "json" => JsonFormat::Json,
            "lines" => JsonFormat::JsonLines,
            _ => {
                return Err(napi::Error::from_reason(
                    "format must be 'json' or `lines'".to_owned(),
                ))
            }
        };

        match path_or_buffer {
            Either::A(path) => {
                let f = std::fs::File::create(path).unwrap();
                let f = BufWriter::new(f);
                JsonWriter::new(f)
                    .with_json_format(json_format)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
            Either::B(inner) => {
                let writeable = JsWriteStream { inner, env: &env };
                JsonWriter::new(writeable)
                    .with_json_format(json_format)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)
                    .unwrap()
            }
        };
        Ok(())
    }
    #[napi]
    pub fn write_avro(
        &mut self,
        path_or_buffer: Either<String, napi::JsObject>,
        compression: String,
        env: Env,
    ) -> napi::Result<()> {
        use polars::io::avro::{AvroCompression, AvroWriter};
        let compression = match compression.as_ref() {
            "uncompressed" => None,
            "snappy" => Some(AvroCompression::Snappy),
            "deflate" => Some(AvroCompression::Deflate),
            s => return Err(JsPolarsErr::Other(format!("compression {} not supported", s)).into()),
        };

        match path_or_buffer {
            Either::A(path) => {
                let f = std::fs::File::create(path).unwrap();
                let f = BufWriter::new(f);
                AvroWriter::new(f)
                    .with_compression(compression)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
            Either::B(inner) => {
                let writeable = JsWriteStream { inner, env: &env };

                AvroWriter::new(writeable)
                    .with_compression(compression)
                    .finish(&mut self.df)
                    .map_err(JsPolarsErr::from)?;
            }
        };
        Ok(())
    }
}

fn finish_groupby(gb: GroupBy, agg: &str) -> napi::Result<JsDataFrame> {
    let df = match agg {
        "min" => gb.min(),
        "max" => gb.max(),
        "mean" => gb.mean(),
        "first" => gb.first(),
        "last" => gb.last(),
        "sum" => gb.sum(),
        "count" => gb.count(),
        "n_unique" => gb.n_unique(),
        "median" => gb.median(),
        "agg_list" => gb.agg_list(),
        "groups" => gb.groups(),
        "std" => gb.std(),
        "var" => gb.var(),
        a => Err(PolarsError::ComputeError(
            format!("agg fn {} does not exists", a).into(),
        )),
    };

    let df = df.map_err(JsPolarsErr::from)?;
    Ok(JsDataFrame::new(df))
}

fn coerce_data_type<A: Borrow<DataType>>(datatypes: &[A]) -> DataType {
    use DataType::*;

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }

    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());

    return match (lhs, rhs) {
        (lhs, rhs) if lhs == rhs => lhs.clone(),
        (List(lhs), List(rhs)) => {
            let inner = coerce_data_type(&[lhs.as_ref(), rhs.as_ref()]);
            List(Box::new(inner))
        }
        (scalar, List(list)) => {
            let inner = coerce_data_type(&[scalar, list.as_ref()]);
            List(Box::new(inner))
        }
        (List(list), scalar) => {
            let inner = coerce_data_type(&[scalar, list.as_ref()]);
            List(Box::new(inner))
        }
        (Float64, UInt64) => Float64,
        (UInt64, Float64) => Float64,
        (UInt64, Boolean) => UInt64,
        (Boolean, UInt64) => UInt64,
        (_, _) => Utf8,
    };
}

fn obj_to_pairs(rows: &Array, len: usize) -> impl '_ + Iterator<Item = Vec<(String, DataType)>> {
    let len = std::cmp::min(len, rows.len() as usize);
    (0..len).map(move |idx| {
        let obj = rows.get::<Object>(idx as u32).unwrap().unwrap();

        let keys = Object::keys(&obj).unwrap();
        keys.iter()
            .map(|key| {
                let value = obj.get::<_, napi::JsUnknown>(&key).unwrap().unwrap();
                let ty = value.get_type().unwrap();
                let dtype = match ty {
                    ValueType::Boolean => DataType::Boolean,
                    ValueType::Number => DataType::Float64,
                    ValueType::String => DataType::Utf8,
                    ValueType::Object => {
                        if value.is_array().unwrap() {
                            let arr: napi::JsObject = unsafe { value.cast() };
                            let len = arr.get_array_length().unwrap();
                            // dont compare too many items, as it could be expensive
                            let max_take = std::cmp::min(len as usize, 10);
                            let mut dtypes: Vec<DataType> = Vec::with_capacity(len as usize);

                            for idx in 0..max_take {
                                let item: napi::JsUnknown = arr.get_element(idx as u32).unwrap();
                                let ty = item.get_type().unwrap();
                                let dt: Wrap<DataType> = ty.into();
                                dtypes.push(dt.0)
                            }
                            let dtype = coerce_data_type(&dtypes);

                            DataType::List(dtype.into())
                        } else if value.is_date().unwrap() {
                            DataType::Datetime(TimeUnit::Milliseconds, None)
                        } else {
                            DataType::Struct(vec![])
                        }
                    }
                    ValueType::BigInt => DataType::UInt64,
                    _ => DataType::Null,
                };
                (key.to_owned(), dtype)
            })
            .collect()
    })
}

unsafe fn coerce_js_anyvalue<'a>(val: JsUnknown, dtype: DataType) -> JsResult<AnyValue<'a>> {
    use DataType::*;
    let vtype = val.get_type().unwrap();
    match (vtype, dtype) {
        (ValueType::Null | ValueType::Undefined | ValueType::Unknown, _) => Ok(AnyValue::Null),
        (ValueType::String, Utf8) => AnyValue::from_js(val),
        (_, Utf8) => {
            let s = val.coerce_to_string()?.into_unknown();
            AnyValue::from_js(s)
        }
        (ValueType::Boolean, Boolean) => bool::from_js(val).map(AnyValue::Boolean),
        (_, Boolean) => val.coerce_to_bool().map(|b| {
            let b: bool = b.try_into().unwrap();
            AnyValue::Boolean(b)
        }),
        (ValueType::BigInt | ValueType::Number, UInt64) => u64::from_js(val).map(AnyValue::UInt64),
        (_, UInt64) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_int64().unwrap();
            AnyValue::UInt64(n as u64)
        }),
        (ValueType::BigInt | ValueType::Number, Int64) => i64::from_js(val).map(AnyValue::Int64),
        (_, Int64) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_int64().unwrap();
            AnyValue::Int64(n)
        }),
        (ValueType::Number, Float64) => f64::from_js(val).map(AnyValue::Float64),
        (_, Float64) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_double().unwrap();
            AnyValue::Float64(n)
        }),
        (ValueType::Number, Float32) => f32::from_js(val).map(AnyValue::Float32),
        (_, Float32) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_double().unwrap();
            AnyValue::Float32(n as f32)
        }),
        (ValueType::Number, Int32) => i32::from_js(val).map(AnyValue::Int32),
        (_, Int32) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_int32().unwrap();
            AnyValue::Int32(n)
        }),
        (ValueType::Number, UInt32) => u32::from_js(val).map(AnyValue::UInt32),
        (_, UInt32) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_uint32().unwrap();
            AnyValue::UInt32(n)
        }),
        (ValueType::Number, Int16) => i16::from_js(val).map(AnyValue::Int16),
        (_, Int16) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_int32().unwrap();
            AnyValue::Int16(n as i16)
        }),
        (ValueType::Number, UInt16) => u16::from_js(val).map(AnyValue::UInt16),
        (_, UInt16) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_uint32().unwrap();
            AnyValue::UInt16(n as u16)
        }),
        (ValueType::Number, Int8) => i8::from_js(val).map(AnyValue::Int8),
        (_, Int8) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_int32().unwrap();
            AnyValue::Int8(n as i8)
        }),
        (ValueType::Number, UInt8) => u8::from_js(val).map(AnyValue::UInt8),
        (_, UInt8) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_uint32().unwrap();
            AnyValue::UInt8(n as u8)
        }),
        (ValueType::Number, Date) => i32::from_js(val).map(AnyValue::Date),
        (_, Date) => val.coerce_to_number().map(|js_num| {
            let n = js_num.get_int32().unwrap();
            AnyValue::Date(n)
        }),
        (ValueType::BigInt | ValueType::Number, Datetime(_, _)) => {
            i64::from_js(val).map(|d| AnyValue::Datetime(d, TimeUnit::Milliseconds, &None))
        }
        (ValueType::Object, DataType::Datetime(_, _)) => {
            if val.is_date()? {
                let d: napi::JsDate = val.cast();
                let d = d.value_of()?;
                Ok(AnyValue::Datetime(d as i64, TimeUnit::Milliseconds, &None))
            } else {
                Ok(AnyValue::Null)
            }
        }
        (ValueType::Object, DataType::List(_)) => {
            let s = val.to_series();
            Ok(AnyValue::List(s))
        }
        _ => Ok(AnyValue::Null),
    }
}
