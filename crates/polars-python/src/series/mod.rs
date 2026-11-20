#[cfg(feature = "pymethods")]
mod aggregation;
#[cfg(feature = "pymethods")]
mod arithmetic;
#[cfg(feature = "pymethods")]
mod c_interface;
#[cfg(feature = "pymethods")]
mod comparison;
#[cfg(feature = "pymethods")]
pub(crate) mod construction;
#[cfg(feature = "pymethods")]
mod export;
#[cfg(feature = "pymethods")]
mod general;
#[cfg(feature = "pymethods")]
mod import;
#[cfg(feature = "pymethods")]
mod map;
#[cfg(feature = "pymethods")]
mod numpy_ufunc;
#[cfg(feature = "pymethods")]
mod scatter;
pub(crate) use import::{call_arrow_c_stream, import_schema_pycapsule, open_stream_capsule};
use parking_lot::RwLock;
use polars::prelude::{Column, Series};
use pyo3::pyclass;

/// The Python `Series`: a [`Column`] behind a lock.
///
/// Everything that reads values goes through [`PySeries::series`], the materialised
/// [`Series`], so its output does not depend on the column's variant.
#[pyclass(frozen, from_py_object)]
#[repr(transparent)]
pub struct PySeries {
    pub column: RwLock<Column>,
}

impl Clone for PySeries {
    fn clone(&self) -> Self {
        Self {
            column: RwLock::new(self.column.read().clone()),
        }
    }
}

impl From<Series> for PySeries {
    fn from(series: Series) -> Self {
        Self::new(series)
    }
}

impl From<Column> for PySeries {
    fn from(column: Column) -> Self {
        Self::from_column(column)
    }
}

impl PySeries {
    /// Wrap a [`Series`] as a [`Column::Series`].
    ///
    /// This does not go through `From<Series> for Column`, which turns a length-1 `Series`
    /// into a `Column::Scalar` (that changes its flags and sortedness).
    pub(crate) fn new(series: Series) -> Self {
        Self::from_column(Column::Series(series.into()))
    }

    pub(crate) fn from_column(column: Column) -> Self {
        PySeries {
            column: RwLock::new(column),
        }
    }

    /// The materialised [`Series`] of the column, through the column's shared cache.
    ///
    /// Cheap for a `Column::Series` (a reference-count bump). The lock is held only while
    /// cloning, so the result can be used across Python calls.
    pub fn series(&self) -> Series {
        self.column.read().as_materialized_series().clone()
    }

    /// A clone of the column when it is a `Column::Scalar` or `Column::Dictionary`: the
    /// variants whose methods run on the column without materialising. `None` for a
    /// `Column::Series`, whose methods keep running on the [`Series`] (a `Column` method would
    /// turn a length-1 result into a `Column::Scalar`).
    ///
    /// The lock is released before the clone is used, so that the operation can take other
    /// locks (`s.append(s)`).
    pub(crate) fn compressed(&self) -> Option<Column> {
        match &*self.column.read() {
            Column::Series(_) => None,
            c => Some(c.clone()),
        }
    }

    /// Take the column out, to go into a `DataFrame`, as `Series::into_column` did: a
    /// length-1 `Column::Series` becomes a `Column::Scalar`. Other variants pass through.
    ///
    /// The conversion is kept: the scalar column also hands out a normalised `Series` (a
    /// sliced nested array, say, comes back without its offset).
    pub(crate) fn into_frame_column(self) -> Column {
        match self.column.into_inner() {
            Column::Series(s) => s.take().into(),
            c => c,
        }
    }

    /// Take the materialised [`Series`] out.
    pub(crate) fn into_series(self) -> Series {
        self.column.into_inner().take_materialized_series()
    }

    /// Mutate the series in place. A column that is not a `Column::Series` becomes one first.
    pub(crate) fn with_series_mut<R>(&self, f: impl FnOnce(&mut Series) -> R) -> R {
        f(self.column.write().into_materialized_series())
    }

    /// Replace the column with a `Column::Series` of `series`.
    pub(crate) fn set_series(&self, series: Series) {
        *self.column.write() = Column::Series(series.into());
    }

    /// The kind of column that backs this `Series`: `"series"`, `"scalar"` or `"dictionary"`.
    pub(crate) fn column_kind(&self) -> &'static str {
        match &*self.column.read() {
            Column::Series(_) => "series",
            Column::Scalar(_) => "scalar",
            Column::Dictionary(_) => "dictionary",
        }
    }
}

pub(crate) trait ToColumns {
    fn to_columns(self) -> Vec<Column>;
}

impl ToColumns for Vec<PySeries> {
    fn to_columns(self) -> Vec<Column> {
        self.into_iter().map(PySeries::into_frame_column).collect()
    }
}

pub(crate) trait ToPySeries {
    fn to_pyseries(self) -> Vec<PySeries>;
}

/// Keeps each column's variant (nothing is materialised).
impl ToPySeries for Vec<Column> {
    fn to_pyseries(self) -> Vec<PySeries> {
        self.into_iter().map(PySeries::from_column).collect()
    }
}
