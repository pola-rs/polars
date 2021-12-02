pub mod conversion;
pub mod dataframe;
pub mod dataframe_object;
pub mod datatypes;
pub mod error;
pub mod file;
pub mod lazy;
pub mod list_construction;
pub mod prelude;
pub mod series;
pub mod series_object;
use crate::dataframe::JsDataFrame;
use crate::lazy::dsl;
use crate::lazy::lazyframe_object::JsLazyFrame;
use crate::series::{repeat, JsSeries};

use napi::{JsObject, Result};

#[macro_use]
extern crate napi_derive;

#[module_exports]
pub fn init(mut exports: JsObject, env: napi::Env) -> Result<()> {
  let lazy_df_obj = JsLazyFrame::to_object(&env)?;

  let series_object = JsSeries::to_object(&env)?;
  let dataframe_object = JsDataFrame::to_object(&env)?;
  let expr = dsl::JsExpr::to_object(&env)?;

  exports.set_named_property("series", series_object)?;
  exports.set_named_property("df", dataframe_object)?;
  exports.set_named_property("lazy", lazy_df_obj)?;
  exports.set_named_property("expr", expr)?;
  exports.create_named_method("repeat", repeat)?;
  exports.create_named_method("col", dsl::col)?;
  exports.create_named_method("cols", dsl::cols)?;
  exports.create_named_method("lit", dsl::lit)?;

  Ok(())
}
