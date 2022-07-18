#![allow(clippy::transmute_undefined_repr)]

#[macro_use]
extern crate napi_derive;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[napi]
pub fn version() -> &'static str {
    VERSION
}

#[napi]
pub fn toggle_string_cache(toggle: bool) {
    polars::toggle_string_cache(toggle)
}

pub mod conversion;
pub mod dataframe;
pub mod datatypes;
pub mod error;
pub mod file;
pub mod functions;
pub mod lazy;
pub mod list_construction;
pub mod prelude;
pub mod series;
pub mod utils;
pub use polars_core;
pub mod export {
    pub use crate::dataframe::JsDataFrame;
    pub use crate::lazy::dataframe::JsLazyFrame;
    pub use polars;
    pub use polars::prelude::LazyFrame;
    pub use polars_core;
}
