use crate::conversion::*;
use neon::prelude::*;
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

impl JsDataFrame {
    pub fn new(df: DataFrame) -> Self {
        JsDataFrame { df }
    }

    pub fn read_csv(mut cx: FunctionContext) -> DataFrameResult {
        let path = get_string_param(&mut cx, "path")?;
        println!("JsDataFrame::read_csv::{:#?}", path);

        let f = CsvReader::from_path(path).expect("error reading csv");
        let df = f.finish().expect("error reading csv");
        Ok(cx.boxed(df.into()))
    }

    pub fn head(mut cx: FunctionContext) -> DataFrameResult {
        let boxed_df: _ = get_df(&mut cx)?;
        let df: &DataFrame = &boxed_df.df;
        let length = get_num_param(&mut cx, "length")?;

        Ok(cx.boxed(df.head(Some(length as usize)).into()))
    }

    pub fn show(mut cx: FunctionContext) -> JsResult<JsUndefined> {
        let boxed_df: _ = get_df(&mut cx)?;
        let df: &DataFrame = &boxed_df.df;
        let length = get_num_param(&mut cx, "length")?;

        let head = df.head(Some(length as usize));

        println!("JsDataFrame::show::{:#?}", head);
        Ok(cx.undefined())
    }

    pub fn schema(mut _cx: FunctionContext) -> JsResult<JsBox<Schema>> {
        unimplemented!()
    }
}
