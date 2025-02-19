use polars_error::{polars_err, PolarsResult};

use super::Column;
use crate::datatypes::AnyValue;
use crate::frame::DataFrame;

impl DataFrame {
    /// Add columns horizontally.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - the length of all [`Column`] is equal to the height of this [`DataFrame`]
    /// - the columns names are unique
    ///
    /// Note that on a debug build this will panic on duplicates / height mismatch.
    pub unsafe fn hstack_mut_unchecked(&mut self, columns: &[Column]) -> &mut Self {
        // If we don't have any columns yet, copy the height from the given columns.
        if let Some(fst) = columns.first() {
            if self.width() == 0 {
                // SAFETY: The functions invariants asks for all columns to be the same length so
                // that makes that a valid height.
                unsafe { self.set_height(fst.len()) };
            }
        }

        self.clear_schema();
        self.columns.extend_from_slice(columns);

        if cfg!(debug_assertions) {
            Self::validate_columns_slice(&self.columns).unwrap();
        }

        self
    }

    /// Add multiple [`Column`] to a [`DataFrame`].
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn stack(df: &mut DataFrame, columns: &[Column]) {
    ///     df.hstack_mut(columns);
    /// }
    /// ```
    pub fn hstack_mut(&mut self, columns: &[Column]) -> PolarsResult<&mut Self> {
        // Validate first - on a debug build `hstack_mut_unchecked` will panic on invalid columns.
        Self::validate_columns_iter(self.get_columns().iter().chain(columns))?;

        unsafe { self.hstack_mut_unchecked(columns) };

        Ok(self)
    }
}

/// Concat [`DataFrame`]s horizontally.
/// Concat horizontally and extend with null values if lengths don't match
pub fn concat_df_horizontal(dfs: &[DataFrame], check_duplicates: bool) -> PolarsResult<DataFrame> {
    let output_height = dfs
        .iter()
        .map(|df| df.height())
        .max()
        .ok_or_else(|| polars_err!(ComputeError: "cannot concat empty dataframes"))?;

    let owned_df;

    let mut out_width = 0;

    let all_equal_height = dfs.iter().all(|df| {
        out_width += df.width();
        df.height() == output_height
    });

    // if not all equal length, extend the DataFrame with nulls
    let dfs = if !all_equal_height {
        out_width = 0;

        owned_df = dfs
            .iter()
            .cloned()
            .map(|mut df| {
                out_width += df.width();

                if df.height() != output_height {
                    let diff = output_height - df.height();

                    // SAFETY: We extend each column with nulls to the point of being of length
                    // `output_height`. Then, we set the height of the resulting dataframe.
                    unsafe { df.get_columns_mut() }.iter_mut().for_each(|c| {
                        *c = c.extend_constant(AnyValue::Null, diff).unwrap();
                    });
                    df.clear_schema();
                    unsafe {
                        df.set_height(output_height);
                    }
                }
                df
            })
            .collect::<Vec<_>>();
        owned_df.as_slice()
    } else {
        dfs
    };

    let mut acc_cols = Vec::with_capacity(out_width);

    for df in dfs {
        acc_cols.extend(df.get_columns().iter().cloned());
    }

    let df = if check_duplicates {
        DataFrame::new(acc_cols).map_err(|e| e.context("unable to hstack".into()))?
    } else {
        if cfg!(debug_assertions) && acc_cols.len() > 1 {
            assert!(acc_cols.iter().all(|c| c.len() == acc_cols[0].len()));
        }

        unsafe { DataFrame::new_no_checks_height_from_first(acc_cols) }
    };

    Ok(df)
}

#[cfg(test)]
mod tests {
    use polars_error::PolarsError;

    #[test]
    fn test_hstack_mut_empty_frame_height_validation() {
        use crate::frame::DataFrame;
        use crate::prelude::{Column, DataType};
        let mut df = DataFrame::empty();
        let result = df.hstack_mut(&[
            Column::full_null("a".into(), 1, &DataType::Null),
            Column::full_null("b".into(), 3, &DataType::Null),
        ]);

        assert!(
            matches!(result, Err(PolarsError::ShapeMismatch(_))),
            "expected shape mismatch error"
        );
    }
}
