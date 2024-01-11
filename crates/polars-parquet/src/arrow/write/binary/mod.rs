mod basic;
mod nested;

pub use basic::array_to_page;
pub(crate) use basic::{build_statistics, encode_plain};
pub(super) use basic::{encode_non_null_values, ord_binary};
pub use nested::array_to_page as nested_array_to_page;
