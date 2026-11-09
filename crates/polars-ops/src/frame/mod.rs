pub mod gather;
pub mod is_sorted;
pub mod join;
#[cfg(feature = "pivot")]
pub mod unpivot;

pub use join::*;
use polars_core::prelude::*;
#[cfg(feature = "to_dummies")]
use polars_core::runtime::RAYON;
#[cfg(feature = "to_dummies")]
use polars_core::utils::accumulate_dataframes_horizontal;
#[cfg(feature = "to_dummies")]
use rayon::prelude::*;

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
            PlHashSet::from_iter(df.columns().iter().map(|s| s.name().as_str()))
        };

        let cols = RAYON.install(|| {
            df.columns()
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
}

/// Split a frame into complete leading key groups and the trailing group.
///
/// The frame must be sorted by `key`; every group except the trailing one is
/// guaranteed to contain a single key. Groups are views over the input frame.
///
#[cfg(feature = "rle")]
pub fn split_by_key(frame: &DataFrame, key: &str) -> PolarsResult<(Vec<DataFrame>, DataFrame)> {
    let mut lengths = Vec::new();
    crate::series::ops::rle_lengths(frame.column(key)?, &mut lengths)?;

    let mut complete = Vec::with_capacity(lengths.len().saturating_sub(1));
    let mut pending = DataFrame::empty();
    let last = lengths.len().saturating_sub(1);
    let mut offset = 0usize;

    for (i, len) in lengths.iter().enumerate() {
        let len = *len as usize;
        let group = frame.slice(offset as i64, len);
        if i == last {
            pending = group;
        } else {
            complete.push(group);
        }
        offset += len;
    }

    Ok((complete, pending))
}

/// Return the length of the first run of equal keys in a sorted frame.
#[cfg(feature = "rle")]
pub fn first_run_len(frame: &DataFrame, key: &str) -> PolarsResult<usize> {
    let mut lengths = Vec::new();
    crate::series::ops::rle_lengths(frame.column(key)?, &mut lengths)?;
    Ok(lengths.first().copied().unwrap_or(0) as usize)
}
