use polars_error::{PolarsResult, polars_bail};

use crate::relaxed_cell::RelaxedCell;

pub(crate) fn verbose() -> bool {
    std::env::var("POLARS_VERBOSE").as_deref().unwrap_or("") == "1"
}

pub fn check_allow_importing_interval_as_struct(type_name: &'static str) -> PolarsResult<()> {
    static ALLOW: RelaxedCell<bool> = RelaxedCell::new_bool(false);

    if !ALLOW.load() {
        ALLOW.fetch_or(std::env::var("POLARS_IMPORT_INTERVAL_AS_STRUCT").as_deref() == Ok("1"));
    }

    if ALLOW.load() {
        return Ok(());
    }

    polars_bail!(
        ComputeError:
        "could not import from `{type_name}` type. \
        Hint: This can be imported by setting \
        POLARS_IMPORT_INTERVAL_AS_STRUCT=1 in the environment. \
        Note however that this is unstable functionality \
        that may change at any time."
    )
}
