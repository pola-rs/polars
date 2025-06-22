pub mod join;
#[cfg(feature = "pivot")]
pub mod pivot;

pub use join::*;
#[cfg(feature = "to_dummies")]
use polars_core::POOL;
use polars_core::prelude::*;
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
    ///   let dummies = df.to_dummies(None, false, None).unwrap();
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
        categories: Option<PlHashMap<PlSmallStr, Vec<PlSmallStr>>>,
    ) -> PolarsResult<DataFrame> {
        self._to_dummies(None, separator, drop_first, categories)
    }

    #[cfg(feature = "to_dummies")]
    fn columns_to_dummies(
        &self,
        columns: Vec<&str>,
        separator: Option<&str>,
        drop_first: bool,
        categories: Option<PlHashMap<PlSmallStr, Vec<PlSmallStr>>>,
    ) -> PolarsResult<DataFrame> {
        self._to_dummies(Some(columns), separator, drop_first, categories)
    }

    #[cfg(feature = "to_dummies")]
    fn _to_dummies(
        &self,
        columns: Option<Vec<&str>>,
        separator: Option<&str>,
        drop_first: bool,
        categories: Option<PlHashMap<PlSmallStr, Vec<PlSmallStr>>>,
    ) -> PolarsResult<DataFrame> {
        use crate::series::ToDummies;

        let df = self.to_df();

        let set: PlHashSet<&str> = match (columns, categories.as_ref()) {
            // categories override columns if both are provided
            (Some(_), Some(cats)) => PlHashSet::from_iter(cats.keys().map(|k| k.as_str())),
            (Some(cols), None) => PlHashSet::from_iter(cols),
            (None, Some(cats)) => PlHashSet::from_iter(cats.keys().map(|k| k.as_str())),
            (None, None) => PlHashSet::from_iter(df.iter().map(|s| s.name().as_str())),
        };

        let cols = POOL.install(|| {
            df.get_columns()
                .par_iter()
                .map(
                    |s| match (set.contains(s.name().as_str()), categories.as_ref()) {
                        (true, Some(cats)) => {
                            let cats = cats.get(s.name().as_str());
                            s.as_materialized_series()
                                .to_dummies(separator, drop_first, cats)
                        },
                        (true, None) => s
                            .as_materialized_series()
                            .to_dummies(separator, drop_first, None),
                        (false, _) => Ok(s.clone().into_frame()),
                    },
                )
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        accumulate_dataframes_horizontal(cols)
    }
}
