mod basic;
mod nested;

pub(crate) use basic::{array_to_page, build_statistics, encode_plain};
pub use nested::array_to_page as nested_array_to_page;
