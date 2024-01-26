use std::cell::Cell;

// Will be set/ unset in the fetch operation to communicate overwriting the number of rows to scan.
thread_local! {pub static FETCH_ROWS: Cell<Option<usize>> = const { Cell::new(None) }}

pub fn _set_n_rows_for_scan(n_rows: Option<usize>) -> Option<usize> {
    let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());
    fetch_rows.or(n_rows)
}

pub fn _is_fetch_query() -> bool {
    FETCH_ROWS.with(|fetch_rows| fetch_rows.get().is_some())
}
