use polars_error::{PolarsResult, polars_bail};

use crate::frame::column::Column;

pub(super) fn infer_broadcast_height(columns: &[Column]) -> usize {
    if columns.is_empty() {
        return 0;
    }

    columns
        .iter()
        .map(|c| c.len())
        .find(|len| *len != 1)
        .unwrap_or(1)
}

/// Broadcasts to `height`. Errors if a column has non-unit length that does not match `height`.
/// Does not check name duplicates.
pub(super) fn broadcast_columns(height: usize, columns: &mut [Column]) -> PolarsResult<()> {
    for col in columns.iter_mut() {
        // Length not equal to the broadcast len, needs broadcast or is an error.
        let len = col.len();
        if len != height {
            if len != 1 {
                let name = col.name().clone();

                let extra_info = if let Some(c) = columns.iter().find(|c| c.len() == height) {
                    format!(" (matching column '{}')", c.name())
                } else {
                    String::new()
                };

                polars_bail!(
                    ShapeMismatch:
                    "could not create a new DataFrame: \
                    series {name:?} has length {len} \
                    while trying to broadcast to length {height}{extra_info}",
                );
            }
            *col = col.new_from_index(0, height);
        }
    }

    Ok(())
}
