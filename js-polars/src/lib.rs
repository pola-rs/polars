use neon::prelude::*;

pub mod conversion;
pub mod dataframe;
pub mod error;
pub mod datatypes;
pub mod series;
pub mod prelude;

use crate::dataframe::*;
use crate::series::*;


/// Neon does not give us a good api for exporting as modules like 
/// 
/// {
///   Dataframe: {
///     read_csv,
///     head,
///     ...
///     },
///   Series: {
///     read_objects
///   }
/// }
/// 
/// 
/// so we have to export them all as a flattened type and create a wrapper in JS land.
/// ideally id like to find a way to split this out into one for each main class
/// So there would be an entrypoint for `DataFrame`, `Series`, etc.
#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {

    cx.export_function("dataframe_read_csv", JsDataFrame::read_csv)?;
    cx.export_function("dataframe_new_obj", JsDataFrame::new_obj)?;
    cx.export_function("dataframe_from_rows", JsDataFrame::from_rows)?;


    cx.export_function("dataframe_get_fmt", JsDataFrame::get_fmt)?;
    cx.export_function("dataframe_head", JsDataFrame::head)?;
    cx.export_function("dataframe_height", JsDataFrame::height)?;
    cx.export_function("dataframe_is_empty", JsDataFrame::is_empty)?;
    cx.export_function("dataframe_shape", JsDataFrame::shape)?;
    cx.export_function("dataframe_width", JsDataFrame::width)?;

    
    cx.export_function("series_add", JsSeries::add)?;
    cx.export_function("series_div", JsSeries::div)?;
    cx.export_function("series_get_fmt", JsSeries::get_fmt)?;
    cx.export_function("series_head", JsSeries::head)?;
    cx.export_function("series_mul", JsSeries::mul)?;
    cx.export_function("series_sub", JsSeries::sub)?;
    cx.export_function("series_tail", JsSeries::tail)?;
    cx.export_function("series_dtype", JsSeries::dtype)?;
    cx.export_function("series_cumsum", JsSeries::cumsum)?;
    cx.export_function("series_cummin", JsSeries::cummin)?;
    cx.export_function("series_cummax", JsSeries::cummax)?;
    cx.export_function("series_cumprod", JsSeries::cumprod)?;

    cx.export_function("series_name", JsSeries::name)?;
    cx.export_function("series_rename", JsSeries::rename)?;
    cx.export_function("series_dtype", JsSeries::dtype)?;

    cx.export_function("series_new_object", JsSeries::new_object)?;
    cx.export_function("series_new_str", JsSeries::new_str)?;
    cx.export_function("series_new_opt_date", JsSeries::new_opt_date)?;
    cx.export_function("series_new_i8", JsSeries::new_i8)?;
    cx.export_function("series_new_i16", JsSeries::new_i16)?;
    cx.export_function("series_new_i32", JsSeries::new_i32)?;
    cx.export_function("series_new_i64", JsSeries::new_i64)?;
    cx.export_function("series_new_u8", JsSeries::new_u8)?;
    cx.export_function("series_new_u16", JsSeries::new_u16)?;
    cx.export_function("series_new_u32", JsSeries::new_u32)?;
    cx.export_function("series_new_u64", JsSeries::new_u64)?;
    cx.export_function("series_new_f32", JsSeries::new_f32)?;
    cx.export_function("series_new_f64", JsSeries::new_f64)?;
    cx.export_function("series_new_bool", JsSeries::new_bool)?;

    cx.export_function("series_new_opt_u16", JsSeries::new_opt_u16)?;
    cx.export_function("series_new_opt_u32", JsSeries::new_opt_u32)?;
    cx.export_function("series_new_opt_u64", JsSeries::new_opt_u64)?;
    cx.export_function("series_new_opt_i8", JsSeries::new_opt_i8)?;
    cx.export_function("series_new_opt_i16", JsSeries::new_opt_i16)?;
    cx.export_function("series_new_opt_i32", JsSeries::new_opt_i32)?;
    cx.export_function("series_new_opt_i64", JsSeries::new_opt_i64)?;
    cx.export_function("series_new_opt_f32", JsSeries::new_opt_f32)?;
    cx.export_function("series_new_opt_f64", JsSeries::new_opt_f64)?;
    cx.export_function("series_new_opt_bool", JsSeries::new_opt_bool)?;

    Ok(())
}



