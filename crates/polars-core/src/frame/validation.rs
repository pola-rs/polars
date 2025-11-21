use polars_error::{PolarsResult, polars_bail};
use polars_utils::aliases::{InitHashMaps, PlHashSet};

use super::DataFrame;
use super::column::Column;

impl DataFrame {
    /// Ensure all equal height and names are unique.
    ///
    /// An Ok() result indicates `columns` is a valid state for a DataFrame.
    pub fn validate_columns_slice(columns: &[Column]) -> PolarsResult<()> {
        if columns.len() <= 1 {
            return Ok(());
        }

        if columns.len() <= 4 {
            // Too small to be worth spawning a hashmap for, this is at most 6 comparisons.
            for i in 0..columns.len() - 1 {
                let name = columns[i].name();
                let height = columns[i].len();

                for other in columns.iter().skip(i + 1) {
                    if other.name() == name {
                        polars_bail!(duplicate = name);
                    }

                    if other.len() != height {
                        polars_bail!(
                            ShapeMismatch:
                            "height of column '{}' ({}) does not match height of column '{}' ({})",
                            other.name(), other.len(), name, height
                        )
                    }
                }
            }
        } else {
            let first = &columns[0];

            let first_len = first.len();
            let first_name = first.name();

            let mut names = PlHashSet::with_capacity(columns.len());
            names.insert(first_name);

            for col in &columns[1..] {
                let col_name = col.name();
                let col_len = col.len();

                if col_len != first_len {
                    polars_bail!(
                        ShapeMismatch:
                        "height of column '{}' ({}) does not match height of column '{}' ({})",
                        col_name, col_len, first_name, first_len
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
}
