//! # Functions
//!
//! Functions that might be useful.
//!
#[cfg(any(feature = "diagonal_concat", feature = "horizontal_concat"))]
use crate::prelude::*;
#[cfg(feature = "diagonal_concat")]
use crate::utils::concat_df;

/// Concat [`DataFrame`]s horizontally.
#[cfg(feature = "horizontal_concat")]
/// Concat horizontally and extend with null values if lengths don't match
pub fn concat_df_horizontal(dfs: &[DataFrame]) -> PolarsResult<DataFrame> {
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

    for df in &dfs[1..] {
        first_df.hstack_mut(df.get_columns())?;
    }
    Ok(first_df)
}

/// Concat [`DataFrame`]s diagonally.
#[cfg(feature = "diagonal_concat")]
/// Concat diagonally thereby combining different schemas.
pub fn concat_df_diagonal(dfs: &[DataFrame]) -> PolarsResult<DataFrame> {
    // TODO! replace with lazy only?
    let upper_bound_width = dfs.iter().map(|df| df.width()).sum();
    let mut column_names = PlHashSet::with_capacity(upper_bound_width);
    let mut schema = Vec::with_capacity(upper_bound_width);

    for df in dfs {
        df.get_columns().iter().for_each(|s| {
            let name = s.name();
            if column_names.insert(name) {
                schema.push((name, s.dtype()))
            }
        });
    }

    let dfs = dfs
        .iter()
        .map(|df| {
            let height = df.height();
            let mut columns = Vec::with_capacity(schema.len());

            for (name, dtype) in &schema {
                match df.column(name).ok() {
                    Some(s) => columns.push(s.clone()),
                    None => columns.push(Series::full_null(name, height, dtype)),
                }
            }
            DataFrame::new_no_checks(columns)
        })
        .collect::<Vec<_>>();

    concat_df(&dfs)
}
