use std::cell::Cell;

// Will be set/ unset in the fetch operation to communicate overwriting the number of rows to scan.
thread_local! {pub static FETCH_ROWS: Cell<Option<usize>> = Cell::new(None)}

pub fn _set_n_rows_for_scan(n_rows: Option<usize>) -> Option<usize> {
    let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());
    match fetch_rows {
        None => n_rows,
        Some(n) => Some(n),
    }
}
