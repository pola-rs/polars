use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub trait PhysicalIoExpr: Send + Sync {
    /// Take a [`DataFrame`] and produces a boolean [`Series`] that serves
    /// as a predicate mask
    fn evaluate_io(&self, df: &DataFrame) -> PolarsResult<Series>;

    /// Can take &dyn Statistics and determine of a file should be
    /// read -> `true`
    /// or not -> `false`
    fn as_stats_evaluator(&self) -> Option<&dyn StatsEvaluator> {
        None
    }
}

pub trait StatsEvaluator {
    fn should_read(&self, stats: &BatchStats) -> PolarsResult<bool>;
}

#[cfg(feature = "parquet")]
pub(crate) fn apply_predicate(
    df: &mut DataFrame,
    predicate: Option<&dyn PhysicalIoExpr>,
    parallel: bool,
) -> PolarsResult<()> {
    if let (Some(predicate), false) = (&predicate, df.is_empty()) {
        let s = predicate.evaluate_io(df)?;
        let mask = s.bool().expect("filter predicates was not of type boolean");

        if parallel {
            *df = df.filter(mask)?;
        } else {
            *df = df._filter_seq(mask)?;
        }
    }
    Ok(())
}

/// The statistics for a column in a Parquet file
/// or Hive partition.
/// they typically hold
/// - max value
/// - min value
/// - null_count
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColumnStats {
    field: Field,
    // The array may hold the null count for every row group,
    // or for a single row group.
    null_count: Option<Series>,
    min_value: Option<Series>,
    max_value: Option<Series>,
}

impl ColumnStats {
    pub fn new(
        field: Field,
        null_count: Option<Series>,
        min_value: Option<Series>,
        max_value: Option<Series>,
    ) -> Self {
        Self {
            field,
            null_count,
            min_value,
            max_value,
        }
    }

    pub fn from_column_literal(s: Series) -> Self {
        debug_assert_eq!(s.len(), 1);
        Self {
            field: s.field().into_owned(),
            null_count: None,
            min_value: Some(s.clone()),
            max_value: Some(s),
        }
    }

    pub fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    pub fn null_count(&self) -> Option<usize> {
        match self.field.data_type() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => None,
            _ => {
                let s = self.null_count.as_ref()?;
                // if all null, there are no statistics.
                if s.null_count() != s.len() {
                    s.sum().ok()
                } else {
                    None
                }
            },
        }
    }

    pub fn to_min_max(&self) -> Option<Series> {
        let max_val = self.max_value.as_ref()?;
        let min_val = self.min_value.as_ref()?;

        let dtype = min_val.dtype();

        if Self::use_min_max(dtype) {
            let mut min_max_values = min_val.clone();
            min_max_values.append(max_val).unwrap();
            if min_max_values.null_count() > 0 {
                None
            } else {
                Some(min_max_values)
            }
        } else {
            None
        }
    }

    pub fn get_min_state(&self) -> Option<&Series> {
        self.min_value.as_ref()
    }

    pub fn to_min(&self) -> Option<&Series> {
        let min_val = self.min_value.as_ref()?;
        let dtype = min_val.dtype();

        if !Self::use_min_max(dtype) || min_val.len() != 1 {
            return None;
        }

        if min_val.null_count() > 0 {
            None
        } else {
            Some(min_val)
        }
    }

    pub fn to_max(&self) -> Option<&Series> {
        let max_val = self.max_value.as_ref()?;
        let dtype = max_val.dtype();

        if !Self::use_min_max(dtype) || max_val.len() != 1 {
            return None;
        }

        if max_val.null_count() > 0 {
            None
        } else {
            Some(max_val)
        }
    }

    fn use_min_max(dtype: &DataType) -> bool {
        dtype.is_numeric()
            || matches!(
                dtype,
                DataType::String | DataType::Binary | DataType::Boolean
            )
    }
}

/// A collection of column stats with a known schema.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct BatchStats {
    schema: SchemaRef,
    stats: Vec<ColumnStats>,
    // This might not be available,
    // as when prunnign hive partitions.
    num_rows: Option<usize>,
}

impl BatchStats {
    pub fn new(schema: SchemaRef, stats: Vec<ColumnStats>, num_rows: Option<usize>) -> Self {
        Self {
            schema,
            stats,
            num_rows,
        }
    }

    pub fn get_stats(&self, column: &str) -> polars_core::error::PolarsResult<&ColumnStats> {
        self.schema.try_index_of(column).map(|i| &self.stats[i])
    }

    pub fn num_rows(&self) -> Option<usize> {
        self.num_rows
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn column_stats(&self) -> &[ColumnStats] {
        self.stats.as_ref()
    }
}
