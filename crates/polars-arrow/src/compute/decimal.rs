use polars_utils::relaxed_cell::RelaxedCell;

static TRIM_DECIMAL_ZEROS: RelaxedCell<bool> = RelaxedCell::new_bool(false);

pub fn get_trim_decimal_zeros() -> bool {
    TRIM_DECIMAL_ZEROS.load()
}

pub fn set_trim_decimal_zeros(trim: Option<bool>) {
    TRIM_DECIMAL_ZEROS.store(trim.unwrap_or(false))
}
