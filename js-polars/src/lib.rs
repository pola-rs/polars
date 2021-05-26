mod series;
mod dataframe;

use wasm_bindgen::prelude::*;
use polars_core::error::PolarsError;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;


pub struct JsPolarsError(PolarsError);

impl From<PolarsError> for JsPolarsError {
    fn from(e: PolarsError) -> Self {
        Self(e)
    }
}

impl From<JsPolarsError> for JsValue {
    fn from(e: JsPolarsError) -> Self {
        format!("{:?}", e.0).into()
    }
}
