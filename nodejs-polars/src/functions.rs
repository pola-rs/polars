use crate::dataframe::*;
use polars_core::functions as pl_functions;
use polars_core::prelude::DataFrame;

#[napi]
pub fn horizontal_concat(dfs: Vec<&JsDataFrame>) -> napi::Result<JsDataFrame> {
    let dfs: Vec<DataFrame> = dfs.iter().map(|df| df.df.clone()).collect();
    let df = pl_functions::hor_concat_df(&dfs).map_err(crate::error::JsPolarsErr::from)?;
    Ok(df.into())
}
