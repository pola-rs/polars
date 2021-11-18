use neon::prelude::*;

pub mod conversion;
pub mod dataframe;
pub mod error;
pub mod datatypes;
pub mod series;
pub mod series_object;
pub mod prelude;
pub mod list_construction;

use crate::dataframe::*;
use crate::series::*;


#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    let df = JsDataFrame::to_object(&mut cx)?;
    let series = JsSeries::to_object(&mut cx)?;
    cx.export_value("df", df)?;
    cx.export_value("series", series)?;

    Ok(())
}



