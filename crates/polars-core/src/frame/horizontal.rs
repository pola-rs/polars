use polars_error::{PolarsResult, polars_err};

use super::Column;
use crate::datatypes::AnyValue;
use crate::frame::DataFrame;
use crate::frame::validation::validate_columns_slice;

impl DataFrame {
    /// Add columns horizontally.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - the length of all [`Column`] is equal to the height of this [`DataFrame`]
    /// - the columns names are unique
    ///
    /// Note: If `self` is empty, `self.height` will always be overridden by the height of the first
    /// column in `columns`.
    ///
    /// Note that on a debug build this will panic on duplicates / height mismatch.
    pub unsafe fn hstack_mut_unchecked(&mut self, columns: &[Column]) -> &mut Self {
        if self.shape() == (0, 0)
            && let Some(c) = columns.first()
        {
            unsafe { self.set_height(c.len()) };
        }

        unsafe { self.columns_mut() }.extend_from_slice(columns);

        if cfg!(debug_assertions) {
            if let err @ Err(_) = validate_columns_slice(self.height(), self.columns()) {
                let initial_width = self.width() - columns.len();
                unsafe { self.columns_mut() }.truncate(initial_width);
                err.unwrap();
            }
        }

        self
    }

    /// Add multiple [`Column`] to a [`DataFrame`].
    /// Errors if the resulting DataFrame columns have duplicate names or unequal heights.
    ///
    /// Note: If `self` is empty, `self.height` will always be overridden by the height of the first
    /// column in `columns`.
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
        if self.shape() == (0, 0)
            && let Some(c) = columns.first()
        {
            unsafe { self.set_height(c.len()) };
        }

        unsafe { self.columns_mut() }.extend_from_slice(columns);

        if let err @ Err(_) = validate_columns_slice(self.height(), self.columns()) {
            let initial_width = self.width() - columns.len();
            unsafe { self.columns_mut() }.truncate(initial_width);
            err?;
        }

        Ok(self)
    }
}

/// Concat [`DataFrame`]s horizontally.
///
/// If the lengths don't match and strict is false we pad with nulls, or return a `ShapeError` if strict is true.
pub fn concat_df_horizontal(
    dfs: &[DataFrame],
    check_duplicates: bool,
    strict: bool,
    unit_length_as_scalar: bool,
) -> PolarsResult<DataFrame> {
    let output_height = dfs
        .iter()
        .map(|df| df.height())
        .max()
        .ok_or_else(|| polars_err!(ComputeError: "cannot concat empty dataframes"))?;

    let owned_df;

    let mut out_width = 0;

    let all_equal_height = dfs.iter().filter(|df| df.shape() != (0, 0)).all(|df| {
        out_width += df.width();
        df.height() == output_height
    });

    // if not all equal length, extend the DataFrame with nulls
    let dfs = if !all_equal_height {
        if strict {
            return Err(
                polars_err!(ShapeMismatch: "cannot concat dataframes with different heights in 'strict' mode"),
            );
        }
        out_width = 0;

        owned_df = dfs
            .iter()
            .filter(|df| df.shape() != (0, 0))
            .cloned()
            .map(|mut df| {
                out_width += df.width();
                let h = df.height();

                if h != output_height {
                    if unit_length_as_scalar && h == 1 {
                        // SAFETY: We extend each scalar column length to
                        // `output_height`. Then, we set the height of the resulting dataframe.
                        unsafe { df.columns_mut() }.iter_mut().for_each(|c| {
                            let Column::Scalar(s) = c else {
                                panic!("only supported for scalars");
                            };

                            *c = Column::Scalar(s.resize(output_height));
                        });
                    } else {
                        let diff = output_height - h;

                        // SAFETY: We extend each column with nulls to the point of being of length
                        // `output_height`. Then, we set the height of the resulting dataframe.
                        unsafe { df.columns_mut() }.iter_mut().for_each(|c| {
                            *c = c.extend_constant(AnyValue::Null, diff).unwrap();
                        });
                    }
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
        acc_cols.extend(df.columns().iter().cloned());
    }

    let df = if check_duplicates {
        DataFrame::new(output_height, acc_cols)?
    } else {
        unsafe { DataFrame::new_unchecked(output_height, acc_cols) }
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

        // Ensure the DataFrame is not mutated in the error case.
        assert_eq!(df.width(), 0);
    }
}
