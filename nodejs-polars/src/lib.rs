
pub mod conversion;
pub mod dataframe;
pub mod datatypes;
pub mod file;

pub mod error;
pub mod list_construction;
pub mod prelude;
pub mod series;
pub mod series_object;
pub mod dataframe_object;
use crate::series::*;
use crate::dataframe::*;
use napi::{JsObject, Result};


#[macro_use]
extern crate napi_derive;


#[module_exports]
pub fn init(mut exports: JsObject, env: napi::Env) -> Result<()> {


  let series_object = JsSeries::to_object(&env)?;
  let dataframe_object = JsDataFrame::to_object(&env)?;

  exports.set_named_property("series", series_object)?;
  exports.set_named_property("df", dataframe_object)?;
  exports.create_named_method("repeat", repeat)?;

  Ok(())
}
