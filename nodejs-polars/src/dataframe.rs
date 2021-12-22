use crate::conversion::prelude::*;
use crate::datatypes::JsDataType;
use crate::error::JsPolarsEr;
use crate::file::JsFileLike;
use crate::prelude::JsResult;
use napi::{
    CallContext, Either, JsBoolean, JsExternal, JsNumber, JsObject, JsString, JsUndefined,
    JsUnknown,
};
use polars::frame::groupby::GroupBy;
use polars::frame::row::{rows_to_schema, Row};
use polars::prelude::*;
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};

pub struct JsDataFrame {}

impl IntoJs<JsExternal> for DataFrame {
    fn try_into_js(self, cx: &CallContext) -> JsResult<JsExternal> {
        cx.env.create_external(self, None)
    }
}

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

#[js_function(1)]
pub(crate) fn read_csv(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;

    let chunk_size: usize = params.get_as("batchSize")?;
    let columns: Option<Vec<String>> = params.get_as("columns")?;
    let comment_char: Option<&str> = params.get_as("commentChar")?;
    let encoding: &str = params.get_as("encoding")?;
    let has_header: bool = params.get_as("hasHeader")?;
    let ignore_errors: bool = params.get_as("ignoreErrors")?;
    let infer_schema_length: Option<usize> = params.get_as("inferSchemaLength")?;
    let inline: Option<bool> = params.get_as("inline")?;
    let low_memory: bool = params.get_as("lowMemory")?;
    let n_threads: Option<usize> = params.get_as("numThreads")?;
    let null_values: Option<Wrap<NullValues>> = params.get_as("nullValues")?;
    let parse_dates: bool = params.get_as("parseDates")?;
    let path = params.get_as::<String>("file")?;
    let projection: Option<Vec<usize>> = params.get_as("projection")?;
    let quote_char: Option<&str> = params.get_as("quoteChar")?;
    let rechunk: bool = params.get_as("rechunk")?;
    let sep: &str = params.get_as("sep")?;
    let skip_rows: usize = params.get_as("startRows")?;
    let stop_after_n_rows: Option<usize> = params.get_as("endRows")?;
    let null_values = null_values.map(|w| w.0);
    let comment_char = comment_char.map(|s| s.as_bytes()[0]);

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
    let df = if inline.unwrap_or(false) {
        let data: JsString = params.0.get_named_property("file")?;
        let utf = data.into_utf8()?;
        let string_slice = utf.as_slice();
        let c = Cursor::new(string_slice);
        CsvReader::new(c)
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
            .finish()
            .map_err(JsPolarsEr::from)?
    } else {
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
            .finish()
            .map_err(JsPolarsEr::from)?
    };
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn read_parquet(_cx: CallContext) -> JsResult<JsExternal> {
    todo!()
}

#[js_function(1)]
#[cfg(feature = "ipc")]
pub(crate) fn read_ipc(_cx: CallContext) -> JsResult<JsExternal> {
    todo!()
}

#[js_function(1)]
pub(crate) fn read_json(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let inline = params.get_as::<Option<bool>>("inline")?;
    let infer_schema_length = params.get_as::<Option<usize>>("inferSchemaLength")?;
    let batch_size = params.get_as::<usize>("batchSize")?;
    let df = if inline.unwrap_or(false) {
        let data: JsString = params.0.get_named_property("file")?;
        let utf = data.into_utf8()?;
        let string_slice = utf.as_slice();
        let c = Cursor::new(string_slice);
        let reader = BufReader::new(c);

        JsonReader::new(reader)
            .infer_schema(infer_schema_length)
            .with_batch_size(batch_size)
            .finish()
            .unwrap()
    } else {
        let path = params.get_as::<&str>("file")?;
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        JsonReader::new(reader)
            .infer_schema(infer_schema_length)
            .with_batch_size(batch_size)
            .finish()
            .unwrap()
    };

    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn to_json(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let stream = params.get::<JsObject>("writeStream")?;
    let writeable = JsFileLike {
        inner: stream,
        env: cx.env,
    };
    serde_json::to_writer(writeable, df)?;

    cx.env.get_undefined()
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
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let stream = params.get::<JsObject>("writeStream")?;
    let writeable = JsFileLike {
        inner: stream,
        env: cx.env,
    };
    let w = JsonWriter::new(writeable);
    w.finish(df).unwrap();
    cx.env.get_undefined()
}

#[js_function(1)]
pub(crate) fn write_json(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let path = params.get_as::<String>("path")?;
    let p = std::path::Path::new(&path);
    let p = resolve_homedir(p);
    let f = File::create(&p)?;
    let w = JsonWriter::new(f);
    w.finish(df).unwrap();
    cx.env.get_undefined()
}

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
pub(crate) fn read_rows(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let rows = params.get::<JsObject>("rows")?;
    let len = rows.get_array_length()?;
    let keys = rows
        .get_element_unchecked::<JsObject>(0)?
        .get_property_names()?
        .into_unknown();

    let keys = Vec::<String>::from_js(keys)?;

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

#[js_function(1)]
pub(crate) fn write_csv_stream(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let has_headers: bool = params.get_as("hasHeader")?;

    let sep: String = params.get_as("sep")?;
    let sep = sep.chars().next().unwrap();

    let stream = params.get::<JsObject>("writeStream")?;
    let writeable = JsFileLike {
        inner: stream,
        env: cx.env,
    };

    CsvWriter::new(writeable)
        .has_header(has_headers)
        .with_delimiter(sep as u8)
        .finish(&df)
        .map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}

#[js_function(1)]
pub(crate) fn write_csv(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
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
        .finish(&df)
        .map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}

#[js_function(1)]
pub(crate) fn add(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df + s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}
#[js_function(1)]
pub(crate) fn sub(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df - s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn div(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df / s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn mul(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df * s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn rem(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df % s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn sample_n(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let n = params.get_as::<usize>("n")?;
    let with_replacement = params.get_as::<bool>("withReplacement")?;
    df.sample_n(n, with_replacement, 0)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn sample_frac(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let frac = params.get_as::<f64>("frac")?;
    let with_replacement = params.get_as::<bool>("withReplacement")?;
    df.sample_frac(frac, with_replacement, 0)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn rechunk(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let df: DataFrame = df.agg_chunks().into();
    df.try_into_js(&cx)
}
#[js_function(1)]
pub(crate) fn as_str(cx: CallContext) -> JsResult<JsString> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = format!("{:?}", df);
    cx.env.create_string(&s)
}

#[js_function(1)]
pub fn fill_null(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let strategy = params.get_as::<&str>("strategy")?;
    let strat = match strategy {
        "backward" => FillNullStrategy::Backward,
        "forward" => FillNullStrategy::Forward,
        "min" => FillNullStrategy::Min,
        "max" => FillNullStrategy::Max,
        "mean" => FillNullStrategy::Mean,
        "one" => FillNullStrategy::One,
        "zero" => FillNullStrategy::Zero,
        s => return Err(JsPolarsEr::Other(format!("Strategy {} not supported", s)).into()),
    };

    df.fill_null(strat)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn join(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let other = params.get_external::<DataFrame>(&cx, "other")?;
    let left_on = params.get_as::<Vec<&str>>("left_on")?;
    let right_on = params.get_as::<Vec<&str>>("right_on")?;
    let how = params.get_as::<&str>("how")?;
    let suffix = params.get_as::<String>("suffix")?;

    let how = match how {
        "left" => JoinType::Left,
        "inner" => JoinType::Inner,
        "outer" => JoinType::Outer,
        // "asof" => JoinType::AsOf,
        // "cross" => JoinType::Cross,
        _ => panic!("not supported"),
    };

    df.join(other, left_on, right_on, how, Some(suffix))
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn get_columns(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = df.get_columns().clone();
    let mut arr: JsObject = cx.env.create_array_with_length(s.len())?;

    for (idx, series) in s.into_iter().enumerate() {
        let wrapped = cx.env.create_external(series, None)?;
        arr.set_element(idx as u32, wrapped)?;
    }
    Ok(arr)
}

#[js_function(1)]
pub fn columns(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let names = df.get_column_names();
    cx.env.to_js_value(&names)
}

#[js_function(1)]
pub fn set_column_names(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let names = params.get_as::<Vec<&str>>("names")?;
    df.set_column_names(&names).map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}

#[js_function(1)]
pub fn schema(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mut obj = cx.env.create_object()?;

    for field in df.schema().fields() {
        let field_name = format!("{}", field.name()).try_into_js(&cx)?;
        let dtype: JsDataType = field.data_type().clone().into();
        let js_string = dtype.to_string().try_into_js(&cx)?;
        obj.set_property(field_name, js_string).unwrap();
    }
    Ok(obj)
}

#[js_function(1)]
pub fn with_column(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "_series")?;
    let s: Series = s.clone();
    let mut df = df.clone();
    df.with_column(s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub fn dtypes(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let dtypes: Vec<String> = df
        .dtypes()
        .iter()
        .map(|arrow_dtype| {
            let dt: JsDataType = arrow_dtype.into();
            dt.to_string()
        })
        .collect();

    cx.env.to_js_value(&dtypes)
}
#[js_function(1)]
pub fn n_chunks(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let n = df.n_chunks().map_err(JsPolarsEr::from)?;
    cx.env.create_int64(n as i64)
}

#[js_function(1)]
pub fn shape(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let (height, width) = df.shape();
    let height = height.into_js(&cx);
    let width = width.into_js(&cx);
    let mut obj = cx.env.create_object()?;
    obj.set_named_property("height", height)?;
    obj.set_named_property("width", width)?;

    Ok(obj)
}

#[js_function(1)]
pub fn height(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.height().try_into_js(&cx)
}

#[js_function(1)]
pub fn width(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.width().try_into_js(&cx)
}

#[js_function(1)]
pub fn hstack(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let column_obj: JsObject = params.get::<JsObject>("columns")?;
    let in_place = params.get_as::<bool>("in_place")?;

    let len = column_obj.get_array_length()?;
    let mut columns: Vec<Series> = Vec::with_capacity(len as usize);

    for idx in 0..len {
        let item: JsExternal = column_obj.get_element(idx)?;
        let s: &Series = cx.env.get_value_external(&item)?;
        columns.push(s.clone())
    }

    if in_place {
        df.hstack_mut(&columns).map_err(JsPolarsEr::from)?;
        cx.env.get_undefined().map(Either::B)
    } else {
        let df = df.hstack(&columns).map_err(JsPolarsEr::from)?;
        df.try_into_js(&cx).map(Either::A)
    }
}

#[js_function(1)]
pub fn vstack(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let other = params.get_external::<DataFrame>(&cx, "other")?;

    df.vstack(other).map_err(JsPolarsEr::from)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn drop_in_place(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;
    df.drop_in_place(name)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn drop_nulls(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let subset = params.get_as::<Option<Vec<String>>>("subset")?;
    df.drop_nulls(subset.as_ref().map(|s| s.as_ref()))
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn drop(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;
    df.drop(name).map_err(JsPolarsEr::from)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn select_at_idx(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<usize>("index")?;

    let opt = df.select_at_idx(idx).map(|s| s.clone());

    match opt {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn find_idx_by_name(cx: CallContext) -> JsResult<Either<JsNumber, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;
    let opt = df.find_idx_by_name(name);

    match opt {
        Some(idx) => idx.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn column(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;

    df.column(name)
        .map(|s| s.clone())
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn select(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let selection = params.get_as::<Vec<&str>>("selection")?;
    df.select(&selection)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn filter(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mask = params.get_external::<Series>(&cx, "mask")?;
    if let Ok(ca) = mask.bool() {
        df.filter(ca).map_err(JsPolarsEr::from)?.try_into_js(&cx)
    } else {
        Err(napi::Error::from_reason(
            "Expected a boolean mask".to_owned(),
        ))
    }
}
#[js_function(1)]
pub fn take(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let indices = params.get::<JsObject>("indices")?;
    let len = indices.get_array_length()?;
    let indices: AlignedVec<u32> = (0..len)
        .map(|v| {
            let wv: WrappedValue = indices
                .get_element_unchecked::<JsUnknown>(v)
                .unwrap()
                .into();
            wv.extract::<u32>().unwrap()
        })
        .collect();

    let indices = UInt32Chunked::new_from_aligned_vec("", indices);
    df.take(&indices)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn take_with_series(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let indices = params.get_external::<Series>(&cx, "indices")?;
    let idx = indices.u32().map_err(JsPolarsEr::from)?;

    df.take(idx).map_err(JsPolarsEr::from)?.try_into_js(&cx)
}
#[js_function(1)]
pub fn sort(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let by_column = params.get_as::<&str>("by")?;
    let reverse = params.get_as::<bool>("reverse")?;
    df.sort(by_column, reverse)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn sort_in_place(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let by_column = params.get_as::<&str>("by")?;
    let reverse = params.get_as::<bool>("reverse")?;
    df.sort_in_place(by_column, reverse)
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}
#[js_function(1)]
pub fn replace(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;

    let column = params.get_as::<&str>("column")?;
    let new_col = params.get_external::<Series>(&cx, "new_col")?;

    df.replace(column, new_col.clone())
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}

#[js_function(1)]
pub fn rename(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;

    let column = params.get_as::<&str>("column")?;
    let new_col = params.get_as::<&str>("new_col")?;

    df.rename(column, new_col).map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}
#[js_function(1)]
pub fn replace_at_idx(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<usize>("index")?;
    let new_col = params.get_external::<Series>(&cx, "newColumn")?;
    df.replace_at_idx(idx, new_col.clone())
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}
#[js_function(1)]
pub fn insert_at_idx(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<usize>("index")?;
    let new_col = params.get_external::<Series>(&cx, "new_col")?;

    df.insert_at_idx(idx, new_col.clone())
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}
#[js_function(1)]
pub fn slice(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let offset = params.get_as::<usize>("offset")?;
    let length = params.get_as::<usize>("length")?;

    df.slice(offset as i64, length).try_into_js(&cx)
}
#[js_function(1)]
pub fn head(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let length = params.get_as::<Option<usize>>("length")?;
    df.head(length).try_into_js(&cx)
}
#[js_function(1)]
pub fn tail(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let length = params.get_as::<Option<usize>>("length")?;
    df.tail(length).try_into_js(&cx)
}
#[js_function(1)]
pub fn is_unique(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mask = df.is_unique().map_err(JsPolarsEr::from)?;
    mask.into_series().try_into_js(&cx)
}
#[js_function(1)]
pub fn is_duplicated(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mask = df.is_duplicated().map_err(JsPolarsEr::from)?;
    mask.into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn frame_equal(cx: CallContext) -> JsResult<JsBoolean> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let other = params.get_external::<DataFrame>(&cx, "other")?;
    let null_equal = params.get_as::<bool>("nullEqual")?;
    let eq = if null_equal {
        df.frame_equal_missing(other)
    } else {
        df.frame_equal(other)
    };

    cx.env.get_boolean(eq)
}
#[js_function(1)]
pub fn with_row_count(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;
    df.with_row_count(name)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn groupby(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let by = params.get_as::<Vec<&str>>("by")?;
    let agg = params.get_as::<&str>("agg")?;
    let select = params.get_as::<Option<Vec<String>>>("select")?;
    let gb = df.groupby(&by).map_err(JsPolarsEr::from)?;

    let selection = match select.as_ref() {
        Some(s) => gb.select(s),
        None => gb,
    };
    finish_groupby(selection, agg)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn clone(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.clone().try_into_js(&cx)
}
#[js_function(1)]
pub fn melt(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let id_vars = params.get_as::<Vec<&str>>("idVars")?;
    let value_vars = params.get_as::<Vec<&str>>("valueVars")?;

    df.melt(id_vars, value_vars)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn shift(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let periods = params.get_as::<i64>("periods")?;

    df.shift(periods).try_into_js(&cx)
}

#[js_function(1)]
pub fn drop_duplicates(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let maintain_order: bool = params.get_or("maintainOrder", true)?;
    let subset = params.get_as::<Option<Vec<String>>>("subset")?;

    df.drop_duplicates(maintain_order, subset.as_ref().map(|v| v.as_ref()))
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn max(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.max().try_into_js(&cx)
}
#[js_function(1)]
pub fn min(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.min().try_into_js(&cx)
}
#[js_function(1)]
pub fn sum(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.sum().try_into_js(&cx)
}
#[js_function(1)]
pub fn mean(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.mean().try_into_js(&cx)
}
#[js_function(1)]
pub fn std(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.std().try_into_js(&cx)
}
#[js_function(1)]
pub fn var(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.var().try_into_js(&cx)
}
#[js_function(1)]
pub fn median(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.median().try_into_js(&cx)
}
#[js_function(1)]
pub fn null_count(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.null_count().try_into_js(&cx)
}

#[js_function(1)]
pub fn hmax(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = df.hmax().map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn hmean(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let strategy = params.get_as::<&str>("nullStrategy")?;
    let strategy = str_to_null_strategy(strategy)?;

    let s = df.hmean(strategy).map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn hmin(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = df.hmin().map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn hsum(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let strategy = params.get_as::<&str>("nullStrategy")?;
    let strategy = str_to_null_strategy(strategy)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;

    let s = df.hsum(strategy).map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn quantile(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let quantile = params.get_as::<f64>("quantile")?;
    df.quantile(quantile, QuantileInterpolOptions::default())
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn shrink_to_fit(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    df.shrink_to_fit();
    cx.env.get_undefined()
}

#[js_function(1)]
pub fn hash_rows(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let k0 = params.get_as::<u64>("k0")?;
    let k1 = params.get_as::<u64>("k1")?;
    let k2 = params.get_as::<u64>("k2")?;
    let k3 = params.get_as::<u64>("k3")?;

    let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
    let hash = df.hash_rows(Some(hb)).map_err(JsPolarsEr::from)?;
    hash.into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn transpose(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let _df = params.get_external::<DataFrame>(&cx, "_df")?;
    todo!()
}

#[js_function(1)]
pub fn lazy(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.clone().lazy().try_into_js(&cx)
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

fn finish_groupby(gb: GroupBy, agg: &str) -> Result<DataFrame> {
    match agg {
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
    }
}

pub fn resolve_homedir(path: &Path) -> PathBuf {
    // replace "~" with home directory
    if path.starts_with("~") {
        if let Some(homedir) = dirs::home_dir() {
            return homedir.join(path.strip_prefix("~").unwrap());
        }
    }

    path.into()
}
