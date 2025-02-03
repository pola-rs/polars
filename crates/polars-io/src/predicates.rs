use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_parquet::read::expr::{ParquetColumnExpr, ParquetScalar, ParquetScalarRange};
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

    fn isolate_column_expr(
        &self,
        name: &str,
    ) -> Option<(
        Arc<dyn PhysicalIoExpr>,
        Option<SpecializedColumnPredicateExpr>,
    )> {
        _ = name;
        None
    }
}

#[derive(Clone)]
pub enum SpecializedColumnPredicateExpr {
    Eq(Scalar),
    EqMissing(Scalar),
}

#[derive(Clone)]
pub struct ColumnPredicateExpr {
    column_name: PlSmallStr,
    dtype: DataType,
    specialized: Option<SpecializedColumnPredicateExpr>,
    expr: Arc<dyn PhysicalIoExpr>,
}

impl ColumnPredicateExpr {
    pub fn new(
        column_name: PlSmallStr,
        dtype: DataType,
        expr: Arc<dyn PhysicalIoExpr>,
        specialized: Option<SpecializedColumnPredicateExpr>,
    ) -> Self {
        Self {
            column_name,
            dtype,
            specialized,
            expr,
        }
    }

    pub fn is_eq_scalar(&self) -> bool {
        self.to_eq_scalar().is_some()
    }
    pub fn to_eq_scalar(&self) -> Option<&Scalar> {
        match &self.specialized {
            Some(SpecializedColumnPredicateExpr::Eq(sc)) if !sc.is_null() => Some(sc),
            Some(SpecializedColumnPredicateExpr::EqMissing(sc)) => Some(sc),
            _ => None,
        }
    }
}

#[cfg(feature = "parquet")]
impl ParquetColumnExpr for ColumnPredicateExpr {
    fn evaluate_mut(&self, values: &dyn Array, bm: &mut MutableBitmap) {
        // We should never evaluate nulls with this.
        assert!(values.validity().is_none_or(|v| v.set_bits() == 0));
        assert_eq!(
            &self.dtype.to_physical().to_arrow(CompatLevel::newest()),
            values.dtype()
        );

        let series = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                self.column_name.clone(),
                vec![values.to_boxed()],
                &self.dtype,
            )
        };
        let column = series.into_column();
        let df = unsafe { DataFrame::new_no_checks(values.len(), vec![column]) };

        // @TODO: Probably these unwraps should be removed.
        let true_mask = self.expr.evaluate_io(&df).unwrap();
        let true_mask = true_mask.bool().unwrap();

        bm.reserve(true_mask.len());
        for chunk in true_mask.downcast_iter() {
            match chunk.validity() {
                None => bm.extend(chunk.values()),
                Some(v) => bm.extend(chunk.values() & v),
            }
        }
    }
    fn evaluate_null(&self) -> bool {
        let column = Column::full_null(self.column_name.clone(), 1, &self.dtype);
        let df = unsafe { DataFrame::new_no_checks(1, vec![column]) };

        // @TODO: Probably these unwraps should be removed.
        let true_mask = self.expr.evaluate_io(&df).unwrap();
        let true_mask = true_mask.bool().unwrap();

        true_mask.get(0).unwrap_or(false)
    }

    fn to_equals_scalar(&self) -> Option<ParquetScalar> {
        self.to_eq_scalar()
            .and_then(|s| cast_to_parquet_scalar(s.clone()))
    }

    fn to_range_scalar(&self) -> Option<ParquetScalarRange> {
        None
    }
}

#[cfg(feature = "parquet")]
fn cast_to_parquet_scalar(scalar: Scalar) -> Option<ParquetScalar> {
    use {AnyValue as A, ParquetScalar as P};

    Some(match scalar.into_value() {
        A::Null => P::Null,
        A::Boolean(v) => P::Boolean(v),

        A::UInt8(v) => P::UInt8(v),
        A::UInt16(v) => P::UInt16(v),
        A::UInt32(v) => P::UInt32(v),
        A::UInt64(v) => P::UInt64(v),

        A::Int8(v) => P::Int8(v),
        A::Int16(v) => P::Int16(v),
        A::Int32(v) => P::Int32(v),
        A::Int64(v) => P::Int64(v),

        #[cfg(feature = "dtype-time")]
        A::Date(v) => P::Int32(v),
        #[cfg(feature = "dtype-datetime")]
        A::Datetime(v, _, _) | A::DatetimeOwned(v, _, _) => P::Int64(v),
        #[cfg(feature = "dtype-duration")]
        A::Duration(v, _) => P::Int64(v),
        #[cfg(feature = "dtype-time")]
        A::Time(v) => P::Int64(v),

        A::Float32(v) => P::Float32(v),
        A::Float64(v) => P::Float64(v),

        // @TODO: Cast to string
        #[cfg(feature = "dtype-categorical")]
        A::Categorical(_, _, _)
        | A::CategoricalOwned(_, _, _)
        | A::Enum(_, _, _)
        | A::EnumOwned(_, _, _) => return None,

        A::String(v) => P::String(v.into()),
        A::StringOwned(v) => P::String(v.as_str().into()),
        A::Binary(v) => P::Binary(v.into()),
        A::BinaryOwned(v) => P::Binary(v.into()),
        _ => return None,
    })
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

    pub fn field_name(&self) -> &PlSmallStr {
        self.field.name()
    }

    /// Returns the [`DataType`] of the column.
    pub fn dtype(&self) -> &DataType {
        self.field.dtype()
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
        // @scalar-opt
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
        // @scalar-opt
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
    dtype.is_primitive_numeric()
        || dtype.is_temporal()
        || matches!(
            dtype,
            DataType::String | DataType::Binary | DataType::Boolean
        )
}

pub struct ColumnStatistics {
    pub dtype: DataType,
    pub min: AnyValue<'static>,
    pub max: AnyValue<'static>,
    pub null_count: Option<IdxSize>,
}

pub trait SkipBatchPredicate: Send + Sync {
    fn can_skip_batch(
        &self,
        batch_size: IdxSize,
        statistics: PlIndexMap<PlSmallStr, ColumnStatistics>,
    ) -> PolarsResult<bool>;
}

#[derive(Clone)]
pub struct ScanIOPredicate {
    pub predicate: Arc<dyn PhysicalIoExpr>,

    /// Column names that are used in the predicate.
    pub live_columns: Arc<PlIndexSet<PlSmallStr>>,

    /// A predicate that gets given statistics and evaluates whether a batch can be skipped.
    pub skip_batch_predicate: Option<Arc<dyn SkipBatchPredicate>>,
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

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            schema: Arc::new(Schema::default()),
            stats: Vec::new(),
            num_rows: None,
        }
    }
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
