mod conversion;
mod dataframe;
mod datatypes;
mod error;
mod series;
mod utils;
use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;


pub type JsResult<T> = std::result::Result<T, JsValue>;
extern crate console_error_panic_hook;
use std::panic;

#[wasm_bindgen]
pub fn init_hooks() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn logv(x: &JsValue);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (crate::log(&format_args!($($t)*).to_string()))
}
