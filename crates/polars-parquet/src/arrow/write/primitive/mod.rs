mod basic;
mod nested;

pub(crate) use basic::encode_plain;
pub use basic::{array_to_page_integer, array_to_page_plain};
pub use nested::array_to_page as nested_array_to_page;
