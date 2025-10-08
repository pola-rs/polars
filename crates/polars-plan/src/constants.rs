use std::sync::OnceLock;

use polars_utils::pl_str::PlSmallStr;

pub static CSE_REPLACED: &str = "__POLARS_CSER_";
pub static POLARS_TMP_PREFIX: &str = "_POLARS_";
pub static POLARS_PLACEHOLDER: &str = "_POLARS_<>";
pub const LEN: &str = "len";
const LITERAL_NAME: &str = "literal";

// Cache the often used LITERAL and LEN constants
static LITERAL_NAME_INIT: OnceLock<PlSmallStr> = OnceLock::new();
static LEN_INIT: OnceLock<PlSmallStr> = OnceLock::new();

pub fn get_literal_name() -> &'static PlSmallStr {
    LITERAL_NAME_INIT.get_or_init(|| PlSmallStr::from_static(LITERAL_NAME))
}
pub(crate) fn get_len_name() -> PlSmallStr {
    LEN_INIT
        .get_or_init(|| PlSmallStr::from_static(LEN))
        .clone()
}
