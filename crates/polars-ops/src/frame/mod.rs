pub mod join;
#[cfg(feature = "pivot")]
pub mod pivot;

pub use join::*;
use polars_core::POOL;
use polars_core::prelude::*;
#[cfg(feature = "to_dummies")]
use polars_core::utils::accumulate_dataframes_horizontal;
use rayon::prelude::*;

use crate::series::SeriesMethods;

// distinguish between "unsorted" and actual errors
#[derive(Debug)]
enum IsSortedResult {
    Unsorted,
    Error(PolarsError),
}

pub trait IntoDf {
    fn to_df(&self) -> &DataFrame;
}

impl IntoDf for DataFrame {
    fn to_df(&self) -> &DataFrame {
        self
    }
}

impl<T: IntoDf> DataFrameOps for T {}

pub trait DataFrameOps: IntoDf {
    /// Create dummy variables.
    ///
    /// # Example
    ///
    /// ```ignore
    ///
    /// # #[macro_use] extern crate polars_core;
    /// # fn main() {
    ///
    ///  use polars_core::prelude::*;
    ///
    ///  let df = df! {
    ///       "id" => &[1, 2, 3, 1, 2, 3, 1, 1],
    ///       "type" => &["A", "B", "B", "B", "C", "C", "C", "B"],
    ///       "code" => &["X1", "X2", "X3", "X3", "X2", "X2", "X1", "X1"]
    ///   }.unwrap();
    ///
    ///   let dummies = df.to_dummies(None, false, false).unwrap();
    ///   println!("{}", dummies);
    /// # }
    /// ```
    /// Outputs:
    /// ```text
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | id_1 | id_3 | id_2 | type_A | type_B | type_C | code_X1 | code_X2 | code_X3 |
    ///  | ---  | ---  | ---  | ---    | ---    | ---    | ---     | ---     | ---     |
    ///  | u8   | u8   | u8   | u8     | u8     | u8     | u8      | u8      | u8      |
    ///  +======+======+======+========+========+========+=========+=========+=========+
    ///  | 1    | 0    | 0    | 1      | 0      | 0      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 0    | 1    | 0      | 1      | 0      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 1    | 0    | 0      | 1      | 0      | 0       | 0       | 1       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 1      | 0      | 0       | 0       | 1       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 0    | 1    | 0      | 0      | 1      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 1    | 0    | 0      | 0      | 1      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 0      | 1      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 1      | 0      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    /// ```
    #[cfg(feature = "to_dummies")]
    fn to_dummies(
        &self,
        separator: Option<&str>,
        drop_first: bool,
        drop_nulls: bool,
    ) -> PolarsResult<DataFrame> {
        self._to_dummies(None, separator, drop_first, drop_nulls)
    }

    #[cfg(feature = "to_dummies")]
    fn columns_to_dummies(
        &self,
        columns: Vec<&str>,
        separator: Option<&str>,
        drop_first: bool,
        drop_nulls: bool,
    ) -> PolarsResult<DataFrame> {
        self._to_dummies(Some(columns), separator, drop_first, drop_nulls)
    }

    #[cfg(feature = "to_dummies")]
    fn _to_dummies(
        &self,
        columns: Option<Vec<&str>>,
        separator: Option<&str>,
        drop_first: bool,
        drop_nulls: bool,
    ) -> PolarsResult<DataFrame> {
        use crate::series::ToDummies;

        let df = self.to_df();

        let set: PlHashSet<&str> = if let Some(columns) = columns {
            PlHashSet::from_iter(columns)
        } else {
            PlHashSet::from_iter(df.iter().map(|s| s.name().as_str()))
        };

        let cols = POOL.install(|| {
            df.get_columns()
                .par_iter()
                .map(|s| match set.contains(s.name().as_str()) {
                    true => s
                        .as_materialized_series()
                        .to_dummies(separator, drop_first, drop_nulls),
                    false => Ok(s.clone().into_frame()),
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        accumulate_dataframes_horizontal(cols)
    }

    /// Checks if a [`DataFrame`] is sorted. Tries to fail fast.
    fn is_sorted(
        &self,
        subset: Option<Vec<PlSmallStr>>,
        mut sort_options: SortMultipleOptions,
    ) -> PolarsResult<bool> {
        let df = self.to_df();

        // early exit opportunities
        if df.height() <= 1 || df.width() == 0 {
            return Ok(true);
        }

        // get col subset if specified, else get all cols
        let sort_cols = if let Some(subset) = subset {
            let subset = PlHashSet::from_iter(subset);
            df.get_columns()
                .iter()
                .filter(|s| subset.contains(s.name()))
                .collect::<Vec<_>>()
        } else {
            df.get_columns().iter().collect::<Vec<_>>()
        };

        // broadcast single desc/nulls_last to match number of sort columns
        let n_cols = sort_cols.len();
        if sort_options.descending.len() == 1 && n_cols > 1 {
            let descending = sort_options.descending[0];
            sort_options.descending = vec![descending; n_cols];
        } else {
            polars_ensure!(
                sort_options.descending.len() == n_cols,
                InvalidOperation: "length mismatch: `descending` has length {} but there are {} sort columns",
                sort_options.descending.len(), n_cols
            );
        }
        if sort_options.nulls_last.len() == 1 && n_cols > 1 {
            let nulls_last = sort_options.nulls_last[0];
            sort_options.nulls_last = vec![nulls_last; n_cols];
        } else {
            polars_ensure!(
                sort_options.nulls_last.len() == n_cols,
                InvalidOperation: "length mismatch: `nulls_last` has length {} but there are {} sort columns",
                sort_options.nulls_last.len(), n_cols
            );
        }

        // check each component series to confirm frame-level sorted order
        if n_cols > 1 && sort_options.multithreaded {
            // parallelized col check with early exit (on first unsorted col or error)
            let result = POOL.install(|| {
                sort_cols.par_iter().enumerate().try_for_each(
                    |(i, col)| -> Result<(), IsSortedResult> {
                        let opts = SortOptions {
                            descending: sort_options.descending[i],
                            nulls_last: sort_options.nulls_last[i],
                            multithreaded: sort_options.multithreaded,
                            maintain_order: sort_options.maintain_order,
                            limit: sort_options.limit,
                        };
                        let s = col.as_materialized_series();
                        match s.is_sorted(opts) {
                            Ok(true) => Ok(()),
                            Ok(false) => Err(IsSortedResult::Unsorted),
                            Err(e) => Err(IsSortedResult::Error(e)),
                        }
                    },
                )
            });
            match result {
                Ok(()) => Ok(true), // all cols sorted
                Err(IsSortedResult::Unsorted) => Ok(false),
                Err(IsSortedResult::Error(e)) => Err(e),
            }
        } else {
            for (i, col) in sort_cols.iter().enumerate() {
                let opts = SortOptions {
                    descending: sort_options.descending[i],
                    nulls_last: sort_options.nulls_last[i],
                    multithreaded: sort_options.multithreaded,
                    maintain_order: sort_options.maintain_order,
                    limit: sort_options.limit,
                };
                let s = col.as_materialized_series();
                if !s.is_sorted(opts)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }
}
