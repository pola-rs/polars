mod basic;
mod nested;

pub(crate) use basic::{array_to_page, encode_plain};
pub use nested::array_to_page as nested_array_to_page;
