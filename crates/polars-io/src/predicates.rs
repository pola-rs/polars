use std::fmt;

use polars_arrow::array::Array;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::datatypes::ArrowDataType;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_parquet::read::expr::{ParquetColumnExpr, ParquetScalar, SpecializedParquetColumnExpr};
use polars_utils::format_pl_smallstr;

pub trait PhysicalIoExpr: Send + Sync {
    /// Take a [`DataFrame`] and produces a boolean [`Series`] that serves
    /// as a predicate mask
    fn evaluate_io(&self, df: &DataFrame) -> PolarsResult<Series>;
}

#[derive(Debug, Clone)]
pub enum SpecializedColumnPredicate {
    Equal(Scalar),
    /// A closed (inclusive) range.
    Between(Scalar, Scalar),
    EqualOneOf(Box<[Scalar]>),
    StartsWith(Box<[u8]>),
    EndsWith(Box<[u8]>),
    RegexMatch(regex::bytes::Regex),
}

#[derive(Clone)]
pub struct ColumnPredicateExpr {
    column_name: PlSmallStr,
    dtype: DataType,
    source_arrow_dtype: ArrowDataType,
    #[cfg(feature = "parquet")]
    specialized: Option<SpecializedParquetColumnExpr>,
    expr: Arc<dyn PhysicalIoExpr>,
}

impl ColumnPredicateExpr {
    pub fn new(
        column_name: PlSmallStr,
        dtype: DataType,
        source_arrow_dtype: ArrowDataType,
        expr: Arc<dyn PhysicalIoExpr>,
        specialized: Option<SpecializedColumnPredicate>,
    ) -> Self {
        use SpecializedColumnPredicate as S;
        #[cfg(feature = "parquet")]
        use SpecializedParquetColumnExpr as P;
        // A specialized predicate compares its scalars with the values as the file
        // stores them, which polars scales for some Arrow types.
        #[cfg(feature = "parquet")]
        let specialized = specialized.and_then(|s| {
            if DataType::arrow_value_scale(&source_arrow_dtype) != 1 {
                return None;
            }
            Some(match s {
                S::Equal(s) => P::Equal(cast_to_parquet_scalar(s)?),
                S::Between(low, high) => {
                    P::Between(cast_to_parquet_scalar(low)?, cast_to_parquet_scalar(high)?)
                },
                S::EqualOneOf(scalars) => P::EqualOneOf(
                    scalars
                        .into_iter()
                        .map(|s| cast_to_parquet_scalar(s).ok_or(()))
                        .collect::<Result<Box<_>, ()>>()
                        .ok()?,
                ),
                S::StartsWith(s) => P::StartsWith(s),
                S::EndsWith(s) => P::EndsWith(s),
                S::RegexMatch(s) => P::RegexMatch(s),
            })
        });

        Self {
            column_name,
            dtype,
            source_arrow_dtype,
            #[cfg(feature = "parquet")]
            specialized,
            expr,
        }
    }
}

#[cfg(feature = "parquet")]
impl ParquetColumnExpr for ColumnPredicateExpr {
    fn evaluate_mut(&self, values: &dyn Array, bm: &mut BitmapBuilder) {
        // We should never evaluate nulls with this.
        assert!(values.validity().is_none_or(|v| v.set_bits() == 0));

        // @TODO: Probably these unwraps should be removed.
        let series = predicate_values_to_series(
            self.column_name.clone(),
            values,
            &self.dtype,
            &self.source_arrow_dtype,
        )
        .unwrap();
        let column = series.into_column();
        let df = unsafe { DataFrame::new_unchecked(values.len(), vec![column]) };

        // @TODO: Probably these unwraps should be removed.
        let true_mask = self.expr.evaluate_io(&df).unwrap();
        let true_mask = true_mask.bool().unwrap();

        bm.reserve(true_mask.len());
        for chunk in true_mask.downcast_iter() {
            match chunk.validity() {
                None => bm.extend_from_bitmap(chunk.values()),
                Some(v) => bm.extend_from_bitmap(&(chunk.values() & v)),
            }
        }
    }
    fn evaluate_null(&self) -> bool {
        let column = Column::full_null(self.column_name.clone(), 1, &self.dtype);
        let df = unsafe { DataFrame::new_unchecked(1, vec![column]) };

        // @TODO: Probably these unwraps should be removed.
        let true_mask = self.expr.evaluate_io(&df).unwrap();
        let true_mask = true_mask.bool().unwrap();

        true_mask.get(0).unwrap_or(false)
    }

    fn as_specialized(&self) -> Option<&SpecializedParquetColumnExpr> {
        self.specialized.as_ref()
    }
}

#[cfg(feature = "parquet")]
fn predicate_values_to_series(
    name: PlSmallStr,
    values: &dyn Array,
    dtype: &DataType,
    source_arrow_dtype: &ArrowDataType,
) -> PolarsResult<Series> {
    // Polars stores the values of some Arrow types scaled, e.g. Arrow seconds as
    // milliseconds, so the predicate series cannot be constructed zero-copy.
    if DataType::arrow_value_scale(source_arrow_dtype) != 1 {
        let values = polars_compute::cast::cast(
            values,
            source_arrow_dtype,
            polars_compute::cast::CastOptionsImpl::default(),
        )?;
        Series::try_from((name, values))
    } else {
        Series::from_chunk_and_dtype(name, values.to_boxed(), dtype)
    }
}

#[cfg(feature = "parquet")]
fn cast_to_parquet_scalar(scalar: Scalar) -> Option<ParquetScalar> {
    use AnyValue as A;
    use ParquetScalar as P;

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

        #[cfg(feature = "dtype-date")]
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
        A::Categorical(_, _) | A::CategoricalOwned(_, _) | A::Enum(_, _) | A::EnumOwned(_, _) => {
            return None;
        },

        A::String(v) => P::String(v.into()),
        A::StringOwned(v) => P::String(v.as_str().into()),
        A::Binary(v) => P::Binary(v.into()),
        A::BinaryOwned(v) => P::Binary(v.into()),
        _ => return None,
    })
}

#[cfg(any(feature = "parquet", feature = "ipc"))]
pub fn apply_predicate(
    df: &mut DataFrame,
    predicate: Option<&dyn PhysicalIoExpr>,
    parallel: bool,
) -> PolarsResult<()> {
    if let (Some(predicate), false) = (&predicate, df.columns().is_empty()) {
        let s = predicate.evaluate_io(df)?;
        let mask = s.bool().expect("filter predicates was not of type boolean");

        if parallel {
            *df = df.filter(mask)?;
        } else {
            *df = df.filter_seq(mask)?;
        }
    }
    Ok(())
}

pub struct ColumnStatistics {
    pub dtype: DataType,
    pub min: AnyValue<'static>,
    pub max: AnyValue<'static>,
    pub null_count: Option<IdxSize>,
}

pub trait SkipBatchPredicate: Send + Sync {
    fn schema(&self) -> &SchemaRef;

    fn can_skip_batch(
        &self,
        batch_size: IdxSize,
        live_columns: &PlIndexSet<PlSmallStr>,
        mut statistics: PlIndexMap<PlSmallStr, ColumnStatistics>,
    ) -> PolarsResult<bool> {
        let mut columns = Vec::with_capacity(1 + live_columns.len() * 3);

        columns.push(Column::new_scalar(
            PlSmallStr::from_static("len"),
            Scalar::new(IDX_DTYPE, batch_size.into()),
            1,
        ));

        for col in live_columns.iter() {
            let dtype = self.schema().get(col).unwrap();
            let (min, max, nc) = match statistics.swap_remove(col) {
                None => (
                    Scalar::null(dtype.clone()),
                    Scalar::null(dtype.clone()),
                    Scalar::null(IDX_DTYPE),
                ),
                Some(stat) => (
                    Scalar::new(dtype.clone(), stat.min),
                    Scalar::new(dtype.clone(), stat.max),
                    Scalar::new(
                        IDX_DTYPE,
                        stat.null_count.map_or(AnyValue::Null, |nc| nc.into()),
                    ),
                ),
            };
            columns.extend([
                Column::new_scalar(format_pl_smallstr!("{col}_min"), min, 1),
                Column::new_scalar(format_pl_smallstr!("{col}_max"), max, 1),
                Column::new_scalar(format_pl_smallstr!("{col}_nc"), nc, 1),
            ]);
        }

        // SAFETY:
        // * Each column is length = 1
        // * We have an IndexSet, so each column name is unique
        let df = unsafe { DataFrame::new_unchecked(1, columns) };
        Ok(self.evaluate_with_stat_df(&df)?.get_bit(0))
    }
    fn evaluate_with_stat_df(&self, df: &DataFrame) -> PolarsResult<Bitmap>;
}

/// The conjuncts of a row predicate that read one column, conjoined.
#[derive(Clone)]
pub struct ColumnPredicate {
    /// The static conjuncts, conjoined. `None` when the column only has dynamic ones.
    pub predicate: Option<Arc<dyn PhysicalIoExpr>>,
    pub specialized: Option<SpecializedColumnPredicate>,
    /// The conjuncts a producer sets at run time, each on its own.
    pub dynamic: Vec<DynamicColumnPredicate>,
}

impl ColumnPredicate {
    /// Every conjunct, static and dynamic, conjoined.
    pub fn conjoined(&self) -> Arc<dyn PhysicalIoExpr> {
        self.predicate
            .iter()
            .chain(self.dynamic.iter().map(|d| &d.predicate))
            .cloned()
            .reduce(|a, b| Arc::new(AndIoExpr(a, b)))
            .unwrap()
    }
}

/// A conjunct on one column that a producer sets at run time. It keeps every
/// row until `source` says it filters rows.
#[derive(Clone)]
pub struct DynamicColumnPredicate {
    pub predicate: Arc<dyn PhysicalIoExpr>,
    pub source: Arc<dyn DynamicPredicateSource>,
}

/// `a AND b`.
struct AndIoExpr(Arc<dyn PhysicalIoExpr>, Arc<dyn PhysicalIoExpr>);

impl PhysicalIoExpr for AndIoExpr {
    fn evaluate_io(&self, df: &DataFrame) -> PolarsResult<Series> {
        let a = self.0.evaluate_io(df)?;
        let b = self.1.evaluate_io(df)?;
        Ok((a.bool()? & b.bool()?).into_series())
    }
}

pub struct PhysicalExprWithConstCols<T> {
    constants: Vec<(PlSmallStr, Scalar)>,
    child: T,
}

impl SkipBatchPredicate for PhysicalExprWithConstCols<Arc<dyn SkipBatchPredicate>> {
    fn schema(&self) -> &SchemaRef {
        self.child.schema()
    }

    fn evaluate_with_stat_df(&self, df: &DataFrame) -> PolarsResult<Bitmap> {
        let mut df = df.clone();
        for (name, scalar) in self.constants.iter() {
            df.with_column(Column::new_scalar(
                name.clone(),
                scalar.clone(),
                df.height(),
            ))?;
        }
        self.child.evaluate_with_stat_df(&df)
    }
}

impl PhysicalIoExpr for PhysicalExprWithConstCols<Arc<dyn PhysicalIoExpr>> {
    fn evaluate_io(&self, df: &DataFrame) -> PolarsResult<Series> {
        let mut df = df.clone();
        for (name, scalar) in self.constants.iter() {
            df.with_column(Column::new_scalar(
                name.clone(),
                scalar.clone(),
                df.height(),
            ))?;
        }

        self.child.evaluate_io(&df)
    }
}

/// The row predicate split into the conjuncts that read a single column and the rest.
#[derive(Clone)]
pub struct StagedScanIOPredicate {
    pub column_predicates: Arc<PlIndexMap<PlSmallStr, ColumnPredicate>>,
    /// The conjuncts that read no or several columns, conjoined.
    pub rest: Option<Arc<dyn PhysicalIoExpr>>,
}

impl StagedScanIOPredicate {
    /// Every conjunct on a constant column becomes part of `rest`.
    fn with_constant_columns(&self, constants: &[(PlSmallStr, Scalar)]) -> Self {
        let mut column_predicates = self.column_predicates.as_ref().clone();
        let mut rest = self.rest.clone();
        for (c, _) in constants {
            if let Some(p) = column_predicates.shift_remove(c) {
                let p = p.conjoined();
                rest = Some(match rest {
                    None => p,
                    Some(rest) => Arc::new(AndIoExpr(rest, p)),
                });
            }
        }
        Self {
            column_predicates: Arc::new(column_predicates),
            rest: rest.map(|rest| {
                Arc::new(PhysicalExprWithConstCols {
                    constants: constants.to_vec(),
                    child: rest,
                }) as _
            }),
        }
    }
}

/// What a producer has published for a column, read once per file: a reader
/// skips the batches whose statistics fall outside the range.
#[derive(Clone, Debug)]
pub enum RuntimeRange {
    /// Not published yet. Every batch is kept; a later file may see a range.
    Pending,
    /// Never published. Every batch is kept.
    Disabled,
    /// No value can match. Every batch is skipped.
    Empty,
    /// Only values in `lo..=hi` can match.
    Range { lo: Scalar, hi: Scalar },
}

/// A reader's view of a predicate that a producer sets at run time.
pub trait DynamicPredicateSource: Send + Sync {
    fn runtime_range(&self) -> RuntimeRange;

    /// Whether the producer has published a predicate that rejects rows.
    fn filters_rows(&self) -> bool;

    /// Whether a reader may stop evaluating the predicate when it rejects too
    /// little: it stays as it is once set, and the producer checks every row
    /// again.
    fn can_bypass(&self) -> bool;
}

/// A column whose batches a reader may skip by a [`RuntimeRange`]. It is never
/// evaluated per row.
#[derive(Clone)]
pub struct RuntimeRangeHint {
    pub column: PlSmallStr,
    pub source: Arc<dyn DynamicPredicateSource>,
    /// The column's value in this file when it is not stored in the file, such as
    /// a hive column or a missing column with a default.
    pub constant: Option<Scalar>,
}

/// A range bound as `dtype`, or `None` when it does not survive the cast, in
/// which case it bounds nothing.
pub fn cast_bound(bound: &Scalar, dtype: &DataType) -> Option<Scalar> {
    bound
        .clone()
        .cast_with_options(dtype, CastOptions::NonStrict)
        .ok()
        .filter(|b| !b.is_null())
}

impl RuntimeRangeHint {
    /// Whether a file whose column is the constant `value` can hold a match.
    /// `None` when the range does not settle it.
    pub fn constant_matches(range: &RuntimeRange, value: &Scalar) -> Option<bool> {
        match range {
            RuntimeRange::Pending | RuntimeRange::Disabled => None,
            RuntimeRange::Empty => Some(false),
            RuntimeRange::Range { lo, hi } => {
                if value.is_null() {
                    return Some(false);
                }
                let lo = cast_bound(lo, value.dtype())?;
                let hi = cast_bound(hi, value.dtype())?;
                let value = value.value();
                Some(value >= lo.value() && value <= hi.value())
            },
        }
    }

    /// Bind the hints of column `name` to its constant value in a file.
    pub fn set_constant(hints: &mut [Self], name: &str, value: &Scalar) {
        for hint in hints.iter_mut().filter(|h| h.column == name) {
            hint.constant = Some(value.clone());
        }
    }
}

#[derive(Clone)]
pub struct ScanIOPredicate {
    pub predicate: Arc<dyn PhysicalIoExpr>,

    /// `predicate` split for readers that filter while decoding.
    pub staged: Option<StagedScanIOPredicate>,

    /// Whether `predicate` filters rows at all. False when the scan only has
    /// runtime ranges to skip batches by.
    pub filters_rows: bool,

    /// Column names that are used in the predicate.
    pub live_columns: Arc<PlIndexSet<PlSmallStr>>,

    /// A predicate that gets given statistics and evaluates whether a batch can be skipped.
    pub skip_batch_predicate: Option<Arc<dyn SkipBatchPredicate>>,

    /// Columns whose batches are skipped by a range published at run time.
    pub runtime_ranges: Vec<RuntimeRangeHint>,

    /// Predicate parts only referring to hive columns.
    pub hive_predicate: Option<Arc<dyn PhysicalIoExpr>>,

    pub hive_predicate_is_full_predicate: bool,
}

impl ScanIOPredicate {
    /// Whether the predicate or a range hint reads the column.
    pub fn reads_column(&self, name: &str) -> bool {
        self.live_columns.contains(name) || self.runtime_ranges.iter().any(|h| h.column == name)
    }

    pub fn set_external_constant_columns(&mut self, constant_columns: Vec<(PlSmallStr, Scalar)>) {
        if constant_columns.is_empty() {
            return;
        }

        let mut live_columns = self.live_columns.as_ref().clone();
        for (c, _) in constant_columns.iter() {
            live_columns.swap_remove(c);
        }
        self.live_columns = Arc::new(live_columns);

        for (name, value) in constant_columns.iter() {
            RuntimeRangeHint::set_constant(&mut self.runtime_ranges, name, value);
        }

        if let Some(skip_batch_predicate) = self.skip_batch_predicate.take() {
            let mut sbp_constant_columns = Vec::with_capacity(constant_columns.len() * 3);
            for (c, v) in constant_columns.iter() {
                sbp_constant_columns.push((format_pl_smallstr!("{c}_min"), v.clone()));
                sbp_constant_columns.push((format_pl_smallstr!("{c}_max"), v.clone()));
                let nc = if v.is_null() {
                    AnyValue::Null
                } else {
                    (0 as IdxSize).into()
                };
                sbp_constant_columns
                    .push((format_pl_smallstr!("{c}_nc"), Scalar::new(IDX_DTYPE, nc)));
            }
            self.skip_batch_predicate = Some(Arc::new(PhysicalExprWithConstCols {
                constants: sbp_constant_columns,
                child: skip_batch_predicate,
            }));
        }

        if let Some(staged) = self.staged.as_mut() {
            *staged = staged.with_constant_columns(&constant_columns);
        }

        self.predicate = Arc::new(PhysicalExprWithConstCols {
            constants: constant_columns,
            child: self.predicate.clone(),
        });
    }
}

impl fmt::Debug for ScanIOPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("scan_io_predicate")
    }
}
