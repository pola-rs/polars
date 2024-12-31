use polars_error::{polars_ensure, polars_err, PolarsResult};
use polars_utils::aliases::PlHashSet;

use super::Column;
use crate::datatypes::AnyValue;
use crate::frame::DataFrame;
use crate::prelude::PlSmallStr;

fn check_hstack(
    col: &Column,
    names: &mut PlHashSet<PlSmallStr>,
    height: usize,
    is_empty: bool,
) -> PolarsResult<()> {
    polars_ensure!(
        col.len() == height || is_empty,
        ShapeMismatch: "unable to hstack Series of length {} and DataFrame of height {}",
        col.len(), height,
    );
    polars_ensure!(
        names.insert(col.name().clone()),
        Duplicate: "unable to hstack, column with name {:?} already exists", col.name().as_str(),
    );
    Ok(())
}

impl DataFrame {
    /// Add columns horizontally.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - the length of all [`Column`] is equal to the height of this [`DataFrame`]
    /// - the columns names are unique
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
        let mut names = self
            .columns
            .iter()
            .map(|c| c.name().clone())
            .collect::<PlHashSet<_>>();

        let height = self.height();
        let is_empty = self.is_empty();
        // first loop check validity. We don't do this in a single pass otherwise
        // this DataFrame is already modified when an error occurs.
        for col in columns {
            check_hstack(col, &mut names, height, is_empty)?;
        }
        drop(names);
        Ok(unsafe { self.hstack_mut_unchecked(columns) })
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

    // if not all equal length, extend the DataFrame with nulls
    let dfs = if !dfs.iter().all(|df| df.height() == output_height) {
        owned_df = dfs
            .iter()
            .cloned()
            .map(|mut df| {
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

    let mut first_df = dfs[0].clone();
    let height = first_df.height();
    let is_empty = first_df.is_empty();

    let mut names = if check_duplicates {
        first_df
            .columns
            .iter()
            .map(|s| s.name().clone())
            .collect::<PlHashSet<_>>()
    } else {
        Default::default()
    };

    for df in &dfs[1..] {
        let cols = df.get_columns();

        if check_duplicates {
            for col in cols {
                check_hstack(col, &mut names, height, is_empty)?;
            }
        }

        unsafe { first_df.hstack_mut_unchecked(cols) };
    }
    Ok(first_df)
}
