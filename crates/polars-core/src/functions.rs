//! # Functions
//!
//! Functions that might be useful.
//!
use crate::prelude::*;
#[cfg(feature = "diagonal_concat")]
use crate::utils::concat_df;

/**
Concat [`DataFrame`]s horizontally.
Concat horizontally and extend with null values if lengths don't match

# Example
```
# use polars_core::prelude::*;
# use polars_core::functions::concat_df_horizontal;
let df_h1 = df!(
    "l1"=> &[1, 2],
    "l2"=> &[3, 4],
)?;
let df_h2 = df!(
    "r1"=> &[5, 6],
    "r2"=> &[7, 8],
    "r3"=> &[9, 10],
)?;
let df_horizontal_concat = concat_df_horizontal(&[df_h1, df_h2])?;

let df_h3 = df!(
    "l1"=> &[1, 2],
    "l2"=> &[3, 4],
    "r1"=> &[5, 6],
    "r2"=> &[7, 8],
    "r3"=> &[9, 10],
)?;
assert_eq!(df_horizontal_concat, df_h3);
# Ok::<(), PolarsError>(())
```
**/
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

#[cfg(feature = "diagonal_concat")]
/**
Concat [`DataFrame`]s diagonally.
Concat diagonally thereby combining different schemas.
# Example
```
# use polars_core::prelude::*;
# use polars_core::functions::concat_df_diagonal;

let df_d1 = df!(
    "a"=> &[1],
    "b"=> &[3],
)?;
let df_d2 = df!(
    "a"=> &[2],
    "d"=> &[4],
)?;
let df_diagonal_concat = concat_df_diagonal(&[df_d1, df_d2])?;

assert!(df_diagonal_concat.equals_missing(
    &df!(
        "a" => &[Some(1), Some(2)],
        "b" => &[Some(3), None],
        "d" => &[None, Some(4)]
    )?
));

# Ok::<(), PolarsError>(())
```
**/
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
