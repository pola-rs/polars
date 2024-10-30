//! # Functions
//!
//! Functions that might be useful.
//!
pub use crate::frame::horizontal::concat_df_horizontal;
#[cfg(feature = "diagonal_concat")]
use crate::prelude::*;
#[cfg(feature = "diagonal_concat")]
use crate::utils::concat_df;

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
            let name = s.name().clone();
            if column_names.insert(name.clone()) {
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
                match df.column(name.as_str()).ok() {
                    Some(s) => columns.push(s.clone()),
                    None => columns.push(Column::full_null(name.clone(), height, dtype)),
                }
            }
            unsafe { DataFrame::new_no_checks(height, columns) }
        })
        .collect::<Vec<_>>();

    concat_df(&dfs)
}
