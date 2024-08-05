use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub trait PhysicalIoExpr: Send + Sync {
    /// Take a [`DataFrame`] and produces a boolean [`Series`] that serves
    /// as a predicate mask
    fn evaluate_io(&self, df: &DataFrame) -> PolarsResult<Series>;

    /// Get the variables that are used in the expression i.e. live variables.
    fn live_variables(&self) -> Option<Vec<Arc<str>>>;

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

#[cfg(any(feature = "parquet", feature = "ipc"))]
pub fn apply_predicate(
    df: &mut DataFrame,
    predicate: Option<&dyn PhysicalIoExpr>,
    parallel: bool,
) -> PolarsResult<()> {
    if let (Some(predicate), false) = (&predicate, df.get_columns().is_empty()) {
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

/// Statistics of the values in a column.
///
/// The following statistics are tracked for each row group:
/// - Null count
/// - Minimum value
/// - Maximum value
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColumnStats {
    field: Field,
    // Each Series contains the stats for each row group.
    null_count: Option<Series>,
    min_value: Option<Series>,
    max_value: Option<Series>,
}

impl ColumnStats {
    /// Constructs a new [`ColumnStats`].
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

    /// Constructs a new [`ColumnStats`] with only the [`Field`] information and no statistics.
    pub fn from_field(field: Field) -> Self {
        Self {
            field,
            null_count: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Constructs a new [`ColumnStats`] from a single-value Series.
    pub fn from_column_literal(s: Series) -> Self {
        debug_assert_eq!(s.len(), 1);
        Self {
            field: s.field().into_owned(),
            null_count: None,
            min_value: Some(s.clone()),
            max_value: Some(s),
        }
    }

    pub fn field_name(&self) -> &SmartString {
        self.field.name()
    }

    /// Returns the [`DataType`] of the column.
    pub fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    /// Returns the null count of each row group of the column.
    pub fn get_null_count_state(&self) -> Option<&Series> {
        self.null_count.as_ref()
    }

    /// Returns the minimum value of each row group of the column.
    pub fn get_min_state(&self) -> Option<&Series> {
        self.min_value.as_ref()
    }

    /// Returns the maximum value of each row group of the column.
    pub fn get_max_state(&self) -> Option<&Series> {
        self.max_value.as_ref()
    }

    /// Returns the null count of the column.
    pub fn null_count(&self) -> Option<usize> {
        match self.dtype() {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => None,
            _ => {
                let s = self.get_null_count_state()?;
                // if all null, there are no statistics.
                if s.null_count() != s.len() {
                    s.sum().ok()
                } else {
                    None
                }
            },
        }
    }

    /// Returns the minimum and maximum values of the column as a single [`Series`].
    pub fn to_min_max(&self) -> Option<Series> {
        let min_val = self.get_min_state()?;
        let max_val = self.get_max_state()?;
        let dtype = self.dtype();

        if !use_min_max(dtype) {
            return None;
        }

        let mut min_max_values = min_val.clone();
        min_max_values.append(max_val).unwrap();
        if min_max_values.null_count() > 0 {
            None
        } else {
            Some(min_max_values)
        }
    }

    /// Returns the minimum value of the column as a single-value [`Series`].
    ///
    /// Returns `None` if no maximum value is available.
    pub fn to_min(&self) -> Option<&Series> {
        let min_val = self.min_value.as_ref()?;
        let dtype = min_val.dtype();

        if !use_min_max(dtype) || min_val.len() != 1 {
            return None;
        }

        if min_val.null_count() > 0 {
            None
        } else {
            Some(min_val)
        }
    }

    /// Returns the maximum value of the column as a single-value [`Series`].
    ///
    /// Returns `None` if no maximum value is available.
    pub fn to_max(&self) -> Option<&Series> {
        let max_val = self.max_value.as_ref()?;
        let dtype = max_val.dtype();

        if !use_min_max(dtype) || max_val.len() != 1 {
            return None;
        }

        if max_val.null_count() > 0 {
            None
        } else {
            Some(max_val)
        }
    }
}

/// Returns whether the [`DataType`] supports minimum/maximum operations.
fn use_min_max(dtype: &DataType) -> bool {
    dtype.is_numeric()
        || dtype.is_temporal()
        || matches!(
            dtype,
            DataType::String | DataType::Binary | DataType::Boolean
        )
}

/// A collection of column stats with a known schema.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct BatchStats {
    schema: SchemaRef,
    stats: Vec<ColumnStats>,
    // This might not be available, as when pruning hive partitions.
    num_rows: Option<usize>,
}

impl BatchStats {
    /// Constructs a new [`BatchStats`].
    ///
    /// The `stats` should match the order of the `schema`.
    pub fn new(schema: SchemaRef, stats: Vec<ColumnStats>, num_rows: Option<usize>) -> Self {
        Self {
            schema,
            stats,
            num_rows,
        }
    }

    /// Returns the [`Schema`] of the batch.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Returns the [`ColumnStats`] of all columns in the batch, if known.
    pub fn column_stats(&self) -> &[ColumnStats] {
        self.stats.as_ref()
    }

    /// Returns the [`ColumnStats`] of a single column in the batch.
    ///
    /// Returns an `Err` if no statistics are available for the given column.
    pub fn get_stats(&self, column: &str) -> PolarsResult<&ColumnStats> {
        self.schema.try_index_of(column).map(|i| &self.stats[i])
    }

    /// Returns the number of rows in the batch.
    ///
    /// Returns `None` if the number of rows is unknown.
    pub fn num_rows(&self) -> Option<usize> {
        self.num_rows
    }

    pub fn with_schema(&mut self, schema: SchemaRef) {
        self.schema = schema;
    }

    pub fn take_indices(&mut self, indices: &[usize]) {
        self.stats = indices.iter().map(|&i| self.stats[i].clone()).collect();
    }
}
