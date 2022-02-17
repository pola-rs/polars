use crate::conversion::prelude::*;
use crate::error::JsPolarsEr;
use crate::file::JsWriteStream;
use crate::prelude::JsResult;
use napi::{CallContext, JsExternal, JsObject, JsString, JsUndefined, JsUnknown, ValueType};
use polars::frame::row::{rows_to_schema, Row};
use polars::io::RowCount;
use polars::prelude::*;
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};

#[js_function(1)]
pub(crate) fn read_columns(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let columns: JsObject = params.0.get_named_property("columns")?;
    let len = columns.get_array_length()?;
    let cols: Vec<Series> = (0..len)
        .map(|idx| {
            let item: JsExternal = columns.get_element(idx).expect("Out of bounds");
            let series: &Series = cx
                .env
                .get_value_external(&item)
                .expect("item is not 'series'");
            series.to_owned()
        })
        .collect();

    DataFrame::new(cols)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

// ------
// CSV
// ------
#[js_function(1)]
pub(crate) fn read_csv_buffer(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;

    let chunk_size: usize = params.get_as("batchSize")?;
    let columns: Option<Vec<String>> = params.get_as("columns")?;
    let comment_char: Option<&str> = params.get_as("commentChar")?;
    let encoding: &str = params.get_as("encoding")?;
    let has_header: bool = params.get_as("hasHeader")?;
    let ignore_errors: bool = params.get_as("ignoreErrors")?;
    let infer_schema_length: Option<usize> = params.get_as("inferSchemaLength")?;
    let low_memory: bool = params.get_as("lowMemory")?;
    let n_threads: Option<usize> = params.get_as("numThreads")?;
    let null_values: Option<Wrap<NullValues>> = params.get_as("nullValues")?;
    let parse_dates: bool = params.get_as("parseDates")?;
    let projection: Option<Vec<usize>> = params.get_as("projection")?;
    let quote_char: Option<&str> = params.get_as("quoteChar")?;
    let rechunk: bool = params.get_as("rechunk")?;
    let sep: &str = params.get_as("sep")?;
    let skip_rows: usize = params.get_as("startRows")?;
    let stop_after_n_rows: Option<usize> = params.get_as("endRows")?;
    let null_values = null_values.map(|w| w.0);
    let comment_char = comment_char.map(|s| s.as_bytes()[0]);
    let buff = params.get::<napi::JsBuffer>("buff")?;
    let buffer_value = buff.into_value()?;
    let row_count = params.get_as::<Option<RowCount>>("rowCount")?;

    let cursor = Cursor::new(buffer_value.as_ref());

    let quote_char = if let Some(s) = quote_char {
        if s.is_empty() {
            None
        } else {
            Some(s.as_bytes()[0])
        }
    } else {
        None
    };
    let encoding = match encoding {
        "utf8" => CsvEncoding::Utf8,
        "utf8-lossy" => CsvEncoding::LossyUtf8,
        e => return Err(JsPolarsEr::Other(format!("encoding not {} not implemented.", e)).into()),
    };

    CsvReader::new(cursor)
        .infer_schema(infer_schema_length)
        .has_header(has_header)
        .with_n_rows(stop_after_n_rows)
        .with_delimiter(sep.as_bytes()[0])
        .with_skip_rows(skip_rows)
        .with_ignore_parser_errors(ignore_errors)
        .with_projection(projection)
        .with_rechunk(rechunk)
        .with_chunk_size(chunk_size)
        .with_encoding(encoding)
        .with_columns(columns)
        .with_n_threads(n_threads)
        .low_memory(low_memory)
        .with_comment_char(comment_char)
        .with_null_values(null_values)
        .with_parse_dates(parse_dates)
        .with_quote_char(quote_char)
        .with_row_count(row_count)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn read_csv_path(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;

    let chunk_size: usize = params.get_as("batchSize")?;
    let columns: Option<Vec<String>> = params.get_as("columns")?;
    let comment_char: Option<&str> = params.get_as("commentChar")?;
    let encoding: &str = params.get_as("encoding")?;
    let has_header: bool = params.get_as("hasHeader")?;
    let ignore_errors: bool = params.get_as("ignoreErrors")?;
    let infer_schema_length: Option<usize> = params.get_as("inferSchemaLength")?;
    let low_memory: bool = params.get_as("lowMemory")?;
    let n_threads: Option<usize> = params.get_as("numThreads")?;
    let null_values: Option<Wrap<NullValues>> = params.get_as("nullValues")?;
    let parse_dates: bool = params.get_as("parseDates")?;
    let path = params.get_as::<String>("path")?;
    let projection: Option<Vec<usize>> = params.get_as("projection")?;
    let quote_char: Option<&str> = params.get_as("quoteChar")?;
    let rechunk: bool = params.get_as("rechunk")?;
    let sep: &str = params.get_as("sep")?;
    let skip_rows: usize = params.get_as("startRows")?;
    let stop_after_n_rows: Option<usize> = params.get_as("endRows")?;
    let null_values = null_values.map(|w| w.0);
    let comment_char = comment_char.map(|s| s.as_bytes()[0]);
    let row_count = params.get_as::<Option<RowCount>>("rowCount")?;

    let quote_char = if let Some(s) = quote_char {
        if s.is_empty() {
            None
        } else {
            Some(s.as_bytes()[0])
        }
    } else {
        None
    };
    let encoding = match encoding {
        "utf8" => CsvEncoding::Utf8,
        "utf8-lossy" => CsvEncoding::LossyUtf8,
        e => return Err(JsPolarsEr::Other(format!("encoding not {} not implemented.", e)).into()),
    };

    CsvReader::from_path(path)
        .expect("unable to read file")
        .infer_schema(infer_schema_length)
        .has_header(has_header)
        .with_n_rows(stop_after_n_rows)
        .with_delimiter(sep.as_bytes()[0])
        .with_skip_rows(skip_rows)
        .with_ignore_parser_errors(ignore_errors)
        .with_projection(projection)
        .with_rechunk(rechunk)
        .with_chunk_size(chunk_size)
        .with_encoding(encoding)
        .with_columns(columns)
        .with_n_threads(n_threads)
        .low_memory(low_memory)
        .with_comment_char(comment_char)
        .with_null_values(null_values)
        .with_parse_dates(parse_dates)
        .with_quote_char(quote_char)
        .with_row_count(row_count)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn write_csv_stream(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let has_headers: bool = params.get_as("hasHeader")?;

    let sep: String = params.get_as("sep")?;
    let sep = sep.chars().next().unwrap();

    let stream = params.get::<JsObject>("writeStream")?;
    let writeable = JsWriteStream {
        inner: stream,
        env: cx.env,
    };

    CsvWriter::new(writeable)
        .has_header(has_headers)
        .with_delimiter(sep as u8)
        .finish(df)
        .map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}

#[js_function(1)]
pub(crate) fn write_csv_path(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let has_headers: bool = params.get_as("hasHeader")?;

    let sep: String = params.get_as("sep")?;
    let sep = sep.chars().next().unwrap();

    let path = params.get_as::<String>("path")?;
    let p = std::path::Path::new(&path);
    let p = resolve_homedir(p);
    let f = File::create(&p)?;

    CsvWriter::new(f)
        .has_header(has_headers)
        .with_delimiter(sep as u8)
        .finish(df)
        .map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}

// ------
// PARQUET
// ------
#[js_function(1)]
pub(crate) fn read_parquet_path(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let path = params.get_as::<String>("path")?;
    let columns: Option<Vec<String>> = params.get_as("columns")?;
    let projection: Option<Vec<usize>> = params.get_as("projection")?;
    let n_rows: Option<usize> = params.get_as("numRows")?;
    let parallel: bool = params.get_or("parallel", true)?;
    let rechunk: bool = params.get_or("rechunk", true)?;
    let row_count = params.get_as::<Option<RowCount>>("rowCount")?;

    let f = File::open(&path)?;

    ParquetReader::new(f)
        .with_projection(projection)
        .with_columns(columns)
        .read_parallel(parallel)
        .with_n_rows(n_rows)
        .with_row_count(row_count)
        .set_rechunk(rechunk)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn read_parquet_buffer(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let columns: Option<Vec<String>> = params.get_as("columns")?;
    let projection: Option<Vec<usize>> = params.get_as("projection")?;
    let n_rows: Option<usize> = params.get_as("numRows")?;
    let parallel: bool = params.get_or("parallel", true)?;
    let rechunk: bool = params.get_or("rechunk", true)?;
    let row_count = params.get_as::<Option<RowCount>>("rowCount")?;

    let buff = params.get::<napi::JsBuffer>("buff")?;
    let buffer_value = buff.into_value()?;

    let cursor = Cursor::new(buffer_value.as_ref());

    ParquetReader::new(cursor)
        .with_projection(projection)
        .with_columns(columns)
        .read_parallel(parallel)
        .with_n_rows(n_rows)
        .with_row_count(row_count)
        .set_rechunk(rechunk)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn write_parquet_path(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let compression = params.get_as::<String>("compression")?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let path = params.get_as::<String>("path")?;
    let compression = match compression.as_str() {
        "uncompressed" => ParquetCompression::Uncompressed,
        "snappy" => ParquetCompression::Snappy,
        "gzip" => ParquetCompression::Gzip,
        "lzo" => ParquetCompression::Lzo,
        "brotli" => ParquetCompression::Brotli,
        "lz4" => ParquetCompression::Lz4,
        "zstd" => ParquetCompression::Zstd,
        s => return Err(JsPolarsEr::Other(format!("compression {} not supported", s)).into()),
    };
    let f = File::create(&path)?;

    ParquetWriter::new(f)
        .with_compression(compression)
        .finish(df)
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}

#[js_function(1)]
pub(crate) fn write_parquet_stream(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let stream = params.get::<JsObject>("writeStream")?;
    let writeable = JsWriteStream {
        inner: stream,
        env: cx.env,
    };
    let compression = params.get_as::<String>("compression")?;
    let compression = match compression.as_str() {
        "uncompressed" => ParquetCompression::Uncompressed,
        "snappy" => ParquetCompression::Snappy,
        "gzip" => ParquetCompression::Gzip,
        "lzo" => ParquetCompression::Lzo,
        "brotli" => ParquetCompression::Brotli,
        "lz4" => ParquetCompression::Lz4,
        "zstd" => ParquetCompression::Zstd,
        s => return Err(JsPolarsEr::Other(format!("compression {} not supported", s)).into()),
    };

    ParquetWriter::new(writeable)
        .with_compression(compression)
        .finish(df)
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}

// ------
// IPC/ARROW
// ------

#[js_function(1)]
pub(crate) fn read_ipc_path(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let path = params.get_as::<String>("path")?;
    let columns: Option<Vec<String>> = params.get_as("columns")?;
    let projection: Option<Vec<usize>> = params.get_as("projection")?;
    let n_rows: Option<usize> = params.get_as("numRows")?;
    let row_count = params.get_as::<Option<RowCount>>("rowCount")?;

    let f = File::open(&path)?;

    IpcReader::new(f)
        .with_projection(projection)
        .with_columns(columns)
        .with_n_rows(n_rows)
        .with_row_count(row_count)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn read_ipc_buffer(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let columns: Option<Vec<String>> = params.get_as("columns")?;
    let projection: Option<Vec<usize>> = params.get_as("projection")?;
    let n_rows: Option<usize> = params.get_as("numRows")?;
    let row_count = params.get_as::<Option<RowCount>>("rowCount")?;

    let buff = params.get::<napi::JsBuffer>("buff")?;
    let buffer_value = buff.into_value()?;

    let cursor = Cursor::new(buffer_value.as_ref());

    IpcReader::new(cursor)
        .with_projection(projection)
        .with_columns(columns)
        .with_n_rows(n_rows)
        .with_row_count(row_count)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn write_ipc_path(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let path = params.get_as::<String>("path")?;
    let compression = params.get_as::<String>("compression")?;
    let compression = match compression.as_str() {
        "uncompressed" => None,
        "lz4" => Some(IpcCompression::LZ4),
        "zstd" => Some(IpcCompression::ZSTD),
        s => return Err(JsPolarsEr::Other(format!("compression {} not supported", s)).into()),
    };
    let f = File::create(&path)?;

    IpcWriter::new(f)
        .with_compression(compression)
        .finish(df)
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}
#[js_function(1)]
pub(crate) fn write_ipc_stream(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let stream = params.get::<JsObject>("writeStream")?;
    let writeable = JsWriteStream {
        inner: stream,
        env: cx.env,
    };
    let compression = params.get_as::<String>("compression")?;

    let compression = match compression.as_str() {
        "uncompressed" => None,
        "lz4" => Some(IpcCompression::LZ4),
        "zstd" => Some(IpcCompression::ZSTD),
        s => return Err(JsPolarsEr::Other(format!("compression {} not supported", s)).into()),
    };

    IpcWriter::new(writeable)
        .with_compression(compression)
        .finish(df)
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}

// ------
// JSON
// ------

#[js_function(1)]
pub(crate) fn read_json_buffer(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let infer_schema_length = params.get_as::<Option<usize>>("inferSchemaLength")?;
    let batch_size = params.get_as::<usize>("batchSize")?;

    let buff = params.get::<napi::JsBuffer>("buff")?;
    let buffer_value = buff.into_value()?;
    let reader = Cursor::new(buffer_value.as_ref());

    JsonReader::new(reader)
        .infer_schema_len(infer_schema_length)
        .with_batch_size(batch_size)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn read_json_path(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let infer_schema_length = params.get_as::<Option<usize>>("inferSchemaLength")?;
    let batch_size = params.get_as::<usize>("batchSize")?;

    let path = params.get_as::<&str>("path")?;
    let f = File::open(&path)?;
    let reader = BufReader::new(f);
    JsonReader::new(reader)
        .infer_schema_len(infer_schema_length)
        .with_batch_size(batch_size)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn to_json(cx: CallContext) -> JsResult<napi::JsBuffer> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let byte_array = serde_json::to_vec(df)?;
    let buff_val = cx.env.create_buffer_with_data(byte_array)?;
    Ok(buff_val.into_raw())
}

#[js_function(1)]
pub(crate) fn to_js(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    cx.env.to_js_value(df)
}

#[js_function(1)]
pub(crate) fn write_json_stream(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let multiline: bool = params.get_or("multiline", false)?;
    let stream = params.get::<JsObject>("writeStream")?;
    let writeable = JsWriteStream {
        inner: stream,
        env: cx.env,
    };
    let json_fmt = match multiline {
        true => JsonFormat::JsonLines,
        false => JsonFormat::Json,
    };
    JsonWriter::new(writeable)
        .with_json_format(json_fmt)
        .finish(df)
        .map_err(|e| JsPolarsEr::Other(format!("{:?}", e)))?;
    cx.env.get_undefined()
}

#[js_function(1)]
pub(crate) fn write_json_path(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let path = params.get_as::<String>("path")?;
    let multiline: bool = params.get_or("multiline", false)?;
    let json_fmt = match multiline {
        true => JsonFormat::JsonLines,
        false => JsonFormat::Json,
    };

    let p = std::path::Path::new(&path);
    let p = resolve_homedir(p);
    let f = File::create(&p)?;
    JsonWriter::new(f)
        .with_json_format(json_fmt)
        .finish(df)
        .map_err(|e| JsPolarsEr::Other(format!("{:?}", e)))?;
    cx.env.get_undefined()
}

// ------
// Rows
// ------

#[js_function(1)]
pub(crate) fn to_rows(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mut arr = cx.env.create_array()?;
    for idx in 0..df.height() {
        let mut arr_row = cx.env.create_array()?;
        for (i, col) in df.get_columns().iter().enumerate() {
            let val: Wrap<AnyValue> = col.get(idx).into();
            let jsv = val.into_js(&cx);
            arr_row.set_element(i as u32, jsv)?;
        }
        arr.set_element(idx as u32, arr_row)?;
    }
    Ok(arr)
}

#[js_function(1)]
pub(crate) fn to_row(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<i64>("idx")?;
    let idx = if idx < 0 {
        (df.height() as i64 + idx) as usize
    } else {
        idx as usize
    };

    let mut row = cx.env.create_array()?;
    for (i, col) in df.get_columns().iter().enumerate() {
        let val: Wrap<AnyValue> = col.get(idx).into();
        let jsv = val.into_js(&cx);
        row.set_element(i as u32, jsv)?;
    }
    Ok(row)
}

#[js_function(1)]
pub(crate) fn to_row_objects(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mut arr = cx.env.create_array()?;
    for idx in 0..df.height() {
        let mut obj_row = cx.env.create_object()?;
        for col in df.get_columns().iter() {
            let col_name = col.name();
            let col_name_js = cx.env.create_string(col_name)?;
            let val: Wrap<AnyValue> = col.get(idx).into();
            let jsv = val.into_js(&cx);
            obj_row.set_property(col_name_js, jsv)?;
        }
        arr.set_element(idx as u32, obj_row)?;
    }
    Ok(arr)
}

#[js_function(1)]
pub(crate) fn to_row_object(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<i64>("idx")?;
    let idx = if idx < 0 {
        (df.height() as i64 + idx) as usize
    } else {
        idx as usize
    };

    let mut obj = cx.env.create_object()?;
    for col in df.get_columns().iter() {
        let val: Wrap<AnyValue> = col.get(idx).into();
        let jsv = val.into_js(&cx);
        let col_name = col.name();
        let col_name_js = cx.env.create_string(col_name)?;
        obj.set_property(col_name_js, jsv)?;
    }
    Ok(obj)
}

#[js_function(1)]
pub(crate) fn read_rows(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let rows = params.get::<JsObject>("rows")?;
    let schema: Schema = infer_schema(&rows)?;

    let len = rows.get_array_length()?;
    let keys = rows
        .get_element_unchecked::<JsObject>(0)?
        .get_property_names()?
        .into_unknown();

    let keys = Vec::<String>::from_js(keys)?;
    (0..len).for_each(|idx| {
        let obj: JsObject = rows.get_element_unchecked(idx).unwrap();

        let v: Vec<AnyValue> = schema
            .fields()
            .iter()
            .map(|fld| {
                let dtype = fld.data_type();
                let key = fld.name();
                let value: JsUnknown = obj.get_named_property(key).unwrap();
                let value = coerce_to_dtype(value, dtype.clone());
                AnyValue::from_js(value).unwrap()
            })
            .collect();
        println!("v={:#?}", v);
    });

    let rows: Vec<Row> = (0..len)
        .map(|idx| {
            let obj: JsObject = rows.get_element_unchecked(idx).unwrap();
            let keys = obj.get_property_names().unwrap();
            let keys_len = keys.get_array_length_unchecked().unwrap();

            Row((0..keys_len)
                .map(|key_idx| {
                    let key: JsString = keys.get_element_unchecked(key_idx).unwrap();
                    let value: JsUnknown = obj.get_property(key).unwrap();
                    AnyValue::from_js(value).unwrap()
                })
                .collect())
        })
        .collect();
    let mut df = finish_from_rows(rows)?;
    df.set_column_names(&keys).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}
use std::collections::{HashMap, HashSet};

type Tracker = HashMap<String, HashSet<DataType>>;

fn infer_schema(rows: &JsObject) -> JsResult<Schema> {
    let mut values: Tracker = Tracker::new();
    let len = rows.get_array_length()?;

    let max_infer = std::cmp::min(len, 50);
    (0..max_infer).for_each(|idx| {
        let obj: JsObject = rows.get_element_unchecked(idx).unwrap();
        let keys = obj.get_property_names().unwrap();
        let keys_len = keys.get_array_length_unchecked().unwrap();
        for key_idx in (0..keys_len) {
            let key: JsString = keys.get_element_unchecked(key_idx).unwrap();

            let value: JsUnknown = obj.get_property(key).unwrap();
            let dtype = DataType::from_js(value).unwrap();

            let key = key.into_utf8().unwrap().into_owned().unwrap();
            add_or_insert(&mut values, &key, dtype);
        }
    });
    Ok(Schema::new(resolve_fields(values)))
}

fn add_or_insert(values: &mut Tracker, key: &str, data_type: DataType) {
    if data_type == DataType::Null {
        return;
    }

    if values.contains_key(key) {
        let x = values.get_mut(key).unwrap();
        x.insert(data_type);
    } else {
        // create hashset and add value type
        let mut hs = HashSet::new();
        hs.insert(data_type);
        values.insert(key.to_string(), hs);
    }
}

fn resolve_fields(spec: Tracker) -> Vec<Field> {
    spec.iter()
        .map(|(k, hs)| {
            let v: Vec<&DataType> = hs.iter().collect();
            Field::new(k, coerce_data_type(&v))
        })
        .collect()
}
use std::borrow::Borrow;

fn coerce_to_dtype(val: JsUnknown, dtype: DataType) -> JsUnknown {
    use DataType::*;
    let actual: Wrap<DataType> = val.get_type().unwrap().into();
    if actual.0 == dtype {
        val
    } else {
        match dtype {
            Boolean => val.coerce_to_bool().unwrap().into_unknown(),
            Float32 => val.coerce_to_number().unwrap().into_unknown(),
            Float64 => val.coerce_to_number().unwrap().into_unknown(),
            Int8 => val.coerce_to_number().unwrap().into_unknown(),
            UInt8 => val.coerce_to_number().unwrap().into_unknown(),
            Int16 => val.coerce_to_number().unwrap().into_unknown(),
            UInt16 => val.coerce_to_number().unwrap().into_unknown(),
            Int32 => val.coerce_to_number().unwrap().into_unknown(),
            UInt32 => val.coerce_to_number().unwrap().into_unknown(),
            _ => val.coerce_to_string().unwrap().into_unknown(),
        }
    }
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

#[js_function(1)]
pub(crate) fn read_array_rows(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let rows = params.get::<JsObject>("data")?;
    let len = rows.get_array_length()?;
    let err_message = "There was an error while processing rows, \ndata must be an array of arrays";
    let rows: Vec<Row> = (0..len)
        .map(|idx| {
            let arr: JsObject = rows.get_element_unchecked(idx).expect(err_message);
            let arr_len = arr.get_array_length().expect(err_message);
            Row((0..arr_len)
                .map(|arr_idx| {
                    let value: JsUnknown = arr.get_element(arr_idx).expect(err_message);
                    AnyValue::from_js(value).expect("unable to cast value")
                })
                .collect())
        })
        .collect();
    finish_from_rows(rows)?.try_into_js(&cx)
}

fn resolve_homedir(path: &Path) -> PathBuf {
    // replace "~" with home directory
    if path.starts_with("~") {
        if let Some(homedir) = dirs::home_dir() {
            return homedir.join(path.strip_prefix("~").unwrap());
        }
    }

    path.into()
}

fn finish_from_rows(rows: Vec<Row>) -> JsResult<DataFrame> {
    let schema = rows_to_schema(&rows);
    let fields = schema
        .fields()
        .iter()
        .map(|fld| match fld.data_type() {
            DataType::Null => Field::new(fld.name(), DataType::Boolean),
            _ => fld.clone(),
        })
        .collect();
    let schema = Schema::new(fields);

    DataFrame::from_rows_and_schema(&rows, &schema).map_err(|err| JsPolarsEr::from(err).into())
}
