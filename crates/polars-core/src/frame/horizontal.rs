use polars_error::{polars_ensure, polars_err, PolarsResult};
use polars_utils::aliases::PlHashSet;

use crate::datatypes::AnyValue;
use crate::frame::DataFrame;
use crate::prelude::{Series, SmartString};

fn check_hstack<'a>(
    col: &'a Series,
    names: &mut PlHashSet<&'a str>,
    height: usize,
    is_empty: bool,
) -> PolarsResult<()> {
    polars_ensure!(
        col.len() == height || is_empty,
        ShapeMismatch: "unable to hstack Series of length {} and DataFrame of height {}",
        col.len(), height,
    );
    polars_ensure!(
        names.insert(col.name()),
        Duplicate: "unable to hstack, column with name {:?} already exists", col.name(),
    );
    Ok(())
}

impl DataFrame {
    /// Add columns horizontally.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - the length of all [`Series`] is equal to the height of this [`DataFrame`]
    /// - the columns names are unique
    pub unsafe fn hstack_mut_unchecked(&mut self, columns: &[Series]) -> &mut Self {
        self.columns.extend_from_slice(columns);
        self
    }

    /// Add multiple [`Series`] to a [`DataFrame`].
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn stack(df: &mut DataFrame, columns: &[Series]) {
    ///     df.hstack_mut(columns);
    /// }
    /// ```
    pub fn hstack_mut(&mut self, columns: &[Series]) -> PolarsResult<&mut Self> {
        let mut names = self
            .columns
            .iter()
            .map(|c| c.name())
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
    let max_len = dfs
        .iter()
        .map(|df| df.height())
        .max()
        .ok_or_else(|| polars_err!(ComputeError: "cannot concat empty dataframes"))?;

    let owned_df;

    // if not all equal length, extend the DataFrame with nulls
    let dfs = if !dfs.iter().all(|df| df.height() == max_len) {
        owned_df = dfs
            .iter()
            .cloned()
            .map(|mut df| {
                if df.height() != max_len {
                    let diff = max_len - df.height();
                    df.columns
                        .iter_mut()
                        .for_each(|s| *s = s.extend_constant(AnyValue::Null, diff).unwrap());
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

    let columns;
    let mut names = if check_duplicates {
        columns = first_df
            .columns
            .iter()
            .map(|s| SmartString::from(s.name()))
            .collect::<Vec<_>>();

        columns.iter().map(|n| n.as_str()).collect::<PlHashSet<_>>()
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
