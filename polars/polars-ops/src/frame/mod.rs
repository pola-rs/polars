mod join;
#[cfg(feature = "pivot")]
pub mod pivot;

pub use join::*;
#[cfg(any(feature = "to_dummies", feature = "approx_unique"))]
use polars_core::export::rayon::prelude::*;
use polars_core::prelude::*;
#[cfg(feature = "to_dummies")]
use polars_core::utils::accumulate_dataframes_horizontal;
#[cfg(any(feature = "to_dummies", feature = "approx_unique"))]
use polars_core::POOL;

#[allow(unused_imports)]
use crate::prelude::*;

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
    ///   let dummies = df.to_dummies().unwrap();
    ///   dbg!(dummies);
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
    fn to_dummies(&self, separator: Option<&str>) -> PolarsResult<DataFrame> {
        self._to_dummies(None, separator)
    }

    #[cfg(feature = "to_dummies")]
    fn columns_to_dummies(
        &self,
        columns: Vec<&str>,
        separator: Option<&str>,
    ) -> PolarsResult<DataFrame> {
        self._to_dummies(Some(columns), separator)
    }

    #[cfg(feature = "to_dummies")]
    fn _to_dummies(
        &self,
        columns: Option<Vec<&str>>,
        separator: Option<&str>,
    ) -> PolarsResult<DataFrame> {
        let df = self.to_df();

        let set: PlHashSet<&str> =
            PlHashSet::from_iter(columns.unwrap_or_else(|| df.get_column_names()));

        let cols = POOL.install(|| {
            df.get_columns()
                .par_iter()
                .map(|s| match set.contains(s.name()) {
                    true => s.to_dummies(separator),
                    false => Ok(s.clone().into_frame()),
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        accumulate_dataframes_horizontal(cols)
    }

    /// Approx count unique values.
    ///
    /// This is done using the HyperLogLog++ algorithm for cardinality estimation.
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
    ///   let approx_count = df.approx_unique(16).unwrap();
    ///   dbg!(approx_count);
    /// # }
    /// ```
    /// Outputs:
    /// ```text
    /// +-----+------+------+
    /// | id  | type | code |
    /// | --- | ---  | ---  |
    /// | u32 | u32  | u32  |
    /// +===================+
    /// | 3   | 3    | 3    |
    /// +-----+------+------+
    /// ```
    #[cfg(feature = "approx_unique")]
    fn approx_unique(&self) -> PolarsResult<DataFrame> {
        let df = self.to_df();

        let cols = POOL.install(|| {
            df.get_columns()
                .par_iter()
                .map(approx_unique)
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        Ok(DataFrame::new_no_checks(cols))
    }
}
