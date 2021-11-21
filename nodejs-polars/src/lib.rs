pub mod series;
pub mod conversion;
pub mod error;
pub mod datatypes;
pub mod prelude;
pub mod list_construction;
use crate::series::*;
use napi::{JsObject, Result};
pub mod series_object;

#[macro_use]
extern crate napi_derive;


#[module_exports]
pub fn init(mut exports: JsObject, env: napi::Env) -> Result<()> {


  let series_object = JsSeries::to_object(&env)?;

  exports.set_named_property("series", series_object)?;
  exports.create_named_method("repeat", repeat)?;

  Ok(())
}
