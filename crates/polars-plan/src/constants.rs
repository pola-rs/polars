use std::sync::{Arc, OnceLock};

use crate::prelude::ColumnName;

pub static MAP_LIST_NAME: &str = "map_list";
pub static CSE_REPLACED: &str = "__POLARS_CSER_";
pub const LEN: &str = "len";
pub const LITERAL_NAME: &str = "literal";
pub const UNLIMITED_CACHE: u32 = u32::MAX;

// Cache the often used LITERAL and LEN constants
static LITERAL_NAME_INIT: OnceLock<Arc<str>> = OnceLock::new();
static LEN_INIT: OnceLock<Arc<str>> = OnceLock::new();

pub(crate) fn get_literal_name() -> Arc<str> {
    LITERAL_NAME_INIT
        .get_or_init(|| ColumnName::from(LITERAL_NAME))
        .clone()
}
pub(crate) fn get_len_name() -> Arc<str> {
    LEN_INIT.get_or_init(|| ColumnName::from(LEN)).clone()
}
