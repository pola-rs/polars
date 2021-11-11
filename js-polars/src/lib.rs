use neon::prelude::*;

pub mod conversion;
pub mod dataframe;
pub mod errors;
pub mod datatypes;
pub mod series;

use crate::dataframe::*;
use crate::series::*;


#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {

    // Neon does not give us a good api for exporting as modules like 
    // {
    //   Dataframe: {
    //     read_csv,
    //     head,
    //     ...
    //     },
    //   Series: {
    //     read_objects
    //   }
    // }
    // so we have to export them all as a flattened type and create a wrapper in JS land.
    cx.export_function("read_csv", JsDataFrame::read_csv)?;
    cx.export_function("read_objects", JsDataFrame::read_objects)?;



    cx.export_function("dataframe_get_fmt", JsDataFrame::get_fmt)?;
    cx.export_function("dataframe_head", JsDataFrame::head)?;
    cx.export_function("dataframe_height", JsDataFrame::height)?;
    cx.export_function("dataframe_is_empty", JsDataFrame::is_empty)?;
    cx.export_function("dataframe_shape", JsDataFrame::shape)?;
    cx.export_function("dataframe_to_js", JsDataFrame::to_js)?;
    cx.export_function("dataframe_width", JsDataFrame::width)?;

    
    cx.export_function("series_add", JsSeries::add)?;
    cx.export_function("series_div", JsSeries::div)?;
    cx.export_function("series_get_fmt", JsSeries::get_fmt)?;
    cx.export_function("series_head", JsSeries::head)?;
    cx.export_function("series_mul", JsSeries::mul)?;
    cx.export_function("series_new", JsSeries::new)?;
    cx.export_function("series_sub", JsSeries::sub)?;
    cx.export_function("series_tail", JsSeries::tail)?;

    Ok(())
}



