use crate::conversion::*;
use crate::error::JsPolarsEr;
use crate::series::JsSeries;
use neon::prelude::*;
use polars::frame::row::{rows_to_schema, Row};
use polars::prelude::*;
use polars::prelude::{CsvReader, DataFrame};

#[derive(Clone)]
pub struct JsDataFrame {
    pub df: DataFrame,
}
impl From<DataFrame> for JsDataFrame {
    fn from(df: DataFrame) -> Self {
        Self { df }
    }
}

impl Finalize for JsDataFrame {}
type DataFrameResult<'a> = JsResult<'a, JsBox<JsDataFrame>>;

impl JsDataFrame {
    pub fn new(df: DataFrame) -> Self {
        JsDataFrame { df }
    }
    pub fn new_obj(mut cx: FunctionContext) -> DataFrameResult {
        let params = get_params(&mut cx)?;
        let (js_arr, capacity) = params.get_arr(&mut cx, "columns")?;
        let mut columns: Vec<Series> = Vec::with_capacity(capacity);
        for item in js_arr.to_vec(&mut cx)?.iter() {
            let s = item.downcast_or_throw::<JsBox<JsSeries>, _>(&mut cx)?;
            columns.push((&s.series).clone())
        }
        let df = DataFrame::new(columns).map_err(JsPolarsEr::from)?;
        let jsdf = JsDataFrame::from(df);
        Ok(jsdf.into_js_box(&mut cx))
    }
    
    fn finish_from_rows(rows: Vec<Row>) -> NeonResult<Self> {
        // replace inferred nulls with boolean
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

        let df = DataFrame::from_rows_and_schema(&rows, &schema).map_err(JsPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn read_csv(mut cx: FunctionContext) -> JsResult<JsBox<JsDataFrame>> {
        let params = get_params(&mut cx)?;
        let path = params.get_as::<String, _>(&mut cx, "path")?;
        let reader = match CsvReader::from_path(&path) {
            Ok(r) => Ok(r),
            Err(_) => Err(JsPolarsEr::Other(format!(
                "error reading from path: {}",
                path
            ))),
        };
        let jsdf: JsDataFrame = match reader?.finish() {
            Ok(r) => Ok(r),
            Err(_) => Err(JsPolarsEr::Other(format!(
                "error reading from path: {}",
                path
            ))),
        }?
        .into();
        Ok(jsdf.into_js_box(&mut cx))
    }

    pub fn head(mut cx: FunctionContext) -> DataFrameResult {
        let params = get_params(&mut cx)?;
        let jsdf = params.extract_boxed::<JsDataFrame>(&mut cx, "_df")?;
        let df = &jsdf.df;
        let length = params.get_as::<f64, _>(&mut cx, "length")?;
        let jsdf: JsDataFrame = df.head(Some(length as usize)).into();
        Ok(jsdf.into_js_box(&mut cx))
    }

    pub fn get_fmt(mut cx: FunctionContext) -> JsResult<JsString> {
        let params = get_params(&mut cx)?;
        let jsdf = params.extract_boxed::<JsDataFrame>(&mut cx, "_df")?;
        let df = &jsdf.df;
        Ok(cx.string(format!("{}", df)))
    }
    pub fn shape(mut cx: FunctionContext) -> JsResult<JsObject> {
        let params = get_params(&mut cx)?;
        let jsdf = params.extract_boxed::<JsDataFrame>(&mut cx, "_df")?;
        let df = &jsdf.df;
        let (height, width) = df.shape();
        let js_height = cx.number(height as f64);
        let js_width = cx.number(width as f64);
        let obj = cx.empty_object();
        obj.set(&mut cx, "height", js_height)?;
        obj.set(&mut cx, "width", js_width)?;
        Ok(obj)
    }

    pub fn height(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let params = get_params(&mut cx)?;
        let jsdf = params.extract_boxed::<JsDataFrame>(&mut cx, "_df")?;
        let df = &jsdf.df;
        Ok(df.height().into_js(&mut cx))
    }

    pub fn width(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let params = get_params(&mut cx)?;
        let jsdf = params.extract_boxed::<JsDataFrame>(&mut cx, "_df")?;
        let df = &jsdf.df;
        Ok(df.width().into_js(&mut cx))
    }

    pub fn is_empty(mut cx: FunctionContext) -> JsResult<JsBoolean> {
        let params = get_params(&mut cx)?;
        let jsdf = params.extract_boxed::<JsDataFrame>(&mut cx, "_df")?;
        let df = &jsdf.df;
        Ok(df.is_empty().into_js(&mut cx))
    }

    pub fn from_rows(mut cx: FunctionContext) -> DataFrameResult {
        let params = get_params(&mut cx)?;
        let (js_arr, _) = params.get_arr(&mut cx, "js_objects")?;
        let (rows, names) = objs_to_rows(&mut cx, &js_arr)?;
        let mut jsdf = Self::finish_from_rows(rows)?;
        jsdf.df.set_column_names(&names).map_err(JsPolarsEr::from)?;
        Ok(jsdf.into_js_box(&mut cx))
    }

    pub fn from_js_array(mut _cx: FunctionContext) -> DataFrameResult {
        unimplemented!()
    }
    pub fn into_js(mut cx: FunctionContext) -> JsResult<JsObject> {
        let obj: Handle<JsObject> = cx
            .argument::<JsObject>(0)
            .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;

        Ok(obj)
    }
}
