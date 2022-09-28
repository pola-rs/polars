use std::cell::Cell;

// Will be set/ unset in the fetch operation to communicate overwriting the number of rows to scan.
thread_local! {pub static FETCH_ROWS: Cell<Option<usize>> = Cell::new(None)}
