use crate::conversion::*;
use crate::errors::JsPolarsEr;
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
        let path: String = cx.from_named_parameter("path")?;
        let reader = CsvReader::from_path(path).expect("error reading csv");
        let jsdf: JsDataFrame = reader.finish().expect("error reading csv").into();
        Ok(jsdf.into_js_box(&mut cx))
    }

    pub fn head(mut cx: FunctionContext) -> DataFrameResult {
        let jsdf = JsDataFrame::extract_boxed(&mut cx, "_df")?;
        let length = f64::extract_val(&mut cx, "length")?;
        let df = &jsdf.df;
        let df: JsDataFrame = df.head(Some(length as usize)).into();
        Ok(df.into_js_box(&mut cx))
    }

    pub fn get_fmt(mut cx: FunctionContext) -> JsResult<JsString> {
        let jsdf = JsDataFrame::extract_boxed(&mut cx, "_df")?;
        Ok(cx.string(format!("{}", jsdf.df)))
    }
    pub fn shape(mut cx: FunctionContext) -> JsResult<JsObject> {
        let jsdf = JsDataFrame::extract_boxed(&mut cx, "_df")?;
        let (height, width) = jsdf.df.shape();
        let js_height = cx.number(height as f64);
        let js_width = cx.number(width as f64);
        let obj = cx.empty_object();
        obj.set(&mut cx, "height", js_height)?;
        obj.set(&mut cx, "width", js_width)?;
        Ok(obj)
    }

    pub fn height(mut cx: FunctionContext) -> JsResult<JsNumber> {
        Ok(JsDataFrame::extract_boxed(&mut cx, "_df")?
            .df
            .height()
            .into_js_value(&mut cx))
    }

    pub fn width(mut cx: FunctionContext) -> JsResult<JsNumber> {
        Ok(JsDataFrame::extract_boxed(&mut cx, "_df")?
            .df
            .width()
            .into_js_value(&mut cx))
    }

    pub fn is_empty(mut cx: FunctionContext) -> JsResult<JsBoolean> {
        Ok(JsDataFrame::extract_boxed(&mut cx, "_df")?
            .df
            .is_empty()
            .into_js_value(&mut cx))
    }

    pub fn read_objects(mut cx: FunctionContext) -> DataFrameResult {
        let js_arr = get_array_param(&mut cx, "js_objects")?;
        let (rows, names) = objs_to_rows(&mut cx, &js_arr)?;
        let mut jsdf = Self::finish_from_rows(rows)?;
        jsdf.df.set_column_names(&names).map_err(JsPolarsEr::from)?;
        Ok(jsdf.into_js_box(&mut cx))
    }

    pub fn from_js_array(mut cx: FunctionContext) -> DataFrameResult {
        unimplemented!()
    }
    pub fn to_js(mut cx: FunctionContext) -> JsResult<JsObject> {
        let obj: Handle<JsObject> = cx
            .argument::<JsObject>(0)
            .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;

        Ok(obj)
    }
}
