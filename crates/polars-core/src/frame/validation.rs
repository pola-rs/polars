use polars_error::{PolarsResult, polars_bail};
use polars_utils::aliases::{InitHashMaps, PlHashSet};

use crate::frame::column::Column;

/// Checks for duplicates and mismatching heights.
pub(super) fn validate_columns_slice(
    expected_height: usize,
    columns: &[Column],
) -> PolarsResult<()> {
    if columns.is_empty() {
        return Ok(());
    }

    let expected_height_msg = || {
        if let Some(c) = columns.iter().find(|c| c.len() == expected_height) {
            format!("height of column '{}' ({})", c.name(), c.len())
        } else {
            format!("DataFrame height ({expected_height})")
        }
    };

    if columns.len() <= 4 {
        // Too small to be worth spawning a hashmap for, this is at most 6 comparisons.
        for (i, col) in columns.iter().enumerate() {
            if col.len() != expected_height {
                polars_bail!(
                    ShapeMismatch:
                    "height of column '{}' ({}) does not match {}",
                    col.name(), col.len(), expected_height_msg()
                )
            }

            let name = col.name();

            for other in columns.iter().skip(i + 1) {
                if other.name() == name {
                    polars_bail!(duplicate = name);
                }
            }
        }
    } else {
        let mut names = PlHashSet::with_capacity(columns.len());

        for col in columns {
            let col_name = col.name();
            let col_len = col.len();

            if col_len != expected_height {
                polars_bail!(
                    ShapeMismatch:
                    "height of column '{}' ({}) does not match {}",
                    col_name, col_len, expected_height_msg()
                )
            }

            if names.contains(col_name) {
                polars_bail!(duplicate = col_name)
            }

            names.insert(col_name);
        }
    }

    Ok(())
}

pub(super) fn ensure_names_unique<T>(names: &[T]) -> PolarsResult<()>
where
    T: AsRef<str>,
{
    // Always unique.
    if names.len() <= 1 {
        return Ok(());
    }

    if names.len() <= 4 {
        // Too small to be worth spawning a hashmap for, this is at most 6 comparisons.
        for i in 0..names.len() - 1 {
            let name = names[i].as_ref();

            for other in names.iter().skip(i + 1) {
                if name == other.as_ref() {
                    polars_bail!(duplicate = name);
                }
            }
        }
    } else {
        let mut names_set: PlHashSet<&str> = PlHashSet::with_capacity(names.len());

        for name in names {
            let name = name.as_ref();

            if !names_set.insert(name) {
                polars_bail!(duplicate = name);
            }
        }
    }
    Ok(())
}
