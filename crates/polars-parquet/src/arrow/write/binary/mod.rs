mod basic;
mod nested;

pub use basic::array_to_page;
pub(super) use basic::{ord_binary, encode_non_null_values};
pub(crate) use basic::{build_statistics, encode_plain};
pub use nested::array_to_page as nested_array_to_page;
