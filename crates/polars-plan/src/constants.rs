use std::sync::{Arc, OnceLock};

pub static MAP_LIST_NAME: &str = "map_list";
pub static CSE_REPLACED: &str = "__POLARS_CSER_";
pub const LEN: &str = "len";
pub const LITERAL_NAME: &str = "literal";

static LITERAL_NAME_INIT: OnceLock<Arc<str>> = OnceLock::new();

pub(crate) fn get_literal_name() -> Arc<str> {
    LITERAL_NAME_INIT.get_or_init(|| {
        Arc::from(crate::constants::LITERAL_NAME)
    }).clone()
}


