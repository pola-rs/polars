use polars_error::{polars_bail, PolarsResult};
use polars_utils::aliases::{InitHashMaps, PlHashSet};

use super::column::Column;
use super::DataFrame;

impl DataFrame {
    /// Ensure all equal height and names are unique.
    ///
    /// An Ok() result indicates `columns` is a valid state for a DataFrame.
    pub fn validate_columns_slice(columns: &[Column]) -> PolarsResult<()> {
        Self::validate_columns_iter(columns.iter())
    }

    pub fn validate_columns_iter<'a, I: IntoIterator<Item = &'a Column>>(
        columns_iter: I,
    ) -> PolarsResult<()> {
        fn _validate_columns_iter_impl(
            columns_iter: &mut dyn Iterator<Item = &Column>,
        ) -> PolarsResult<()> {
            let Some(first) = columns_iter.next() else {
                return Ok(());
            };

            let first_len = first.len();
            let first_name = first.name();

            let mut names = PlHashSet::with_capacity({
                let (_, exact) = columns_iter.size_hint();
                exact.unwrap_or(16)
            });

            names.insert(first_name);

            for col in columns_iter {
                let col_name = col.name();
                let col_len = col.len();

                if col_len != first_len {
                    polars_bail!(
                        ShapeMismatch:
                        "height of '{}' ({}) does not match height of '{}' ({})",
                        col_name, col_len, first_name, first_len
                    )
                }

                if names.contains(col_name) {
                    polars_bail!(Duplicate: "column '{}' is duplicate", col_name)
                }

                names.insert(col_name);
            }

            Ok(())
        }

        _validate_columns_iter_impl(&mut columns_iter.into_iter())
    }
}
