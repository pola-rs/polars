use neon::prelude::*;

pub mod conversion;
pub mod dataframe;
pub mod errors;

use crate::dataframe::*;

pub type BoxedDataFrame = JsBox<JsDataFrame>;
impl Finalize for JsDataFrame {}

register_module!(mut cx, {
    cx.export_function("read_csv", JsDataFrame::read_csv)?;
    cx.export_function("head", JsDataFrame::head)?;
    cx.export_function("show", JsDataFrame::show)?;
    Ok(())
});
