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

    /// The Series a reduction reads when its answer does not depend on how many times a scalar
    /// column repeats its value (`rows` of the length says how many rows tell it), so that the
    /// column is not materialised: the first `rows(len)` rows of a `Column::Scalar`, flagged
    /// sorted as its materialised Series is. The materialised Series of any other column.
    pub(crate) fn repeat_invariant_series(&self, rows: impl FnOnce(usize) -> usize) -> Series {
        match &*self.column.read() {
            Column::Scalar(sc) => sc.as_n_values_series(rows(sc.len())),
            c => c.as_materialized_series().clone(),
        }
    }

    /// The sorted flag of the materialised Series, read from the column where that is the
    /// same value without materialising: a `Column::Dictionary` sets its materialised Series'
    /// flag from its own.
    pub(crate) fn sorted_flag(&self) -> polars_core::series::IsSorted {
        let flag = match &*self.column.read() {
            // (A Series of at most one row reports itself ascending whatever it was set to.)
            Column::Dictionary(d) if d.len() > 1 => Some(d.is_sorted_flag()),
            _ => None,
        };
        flag.unwrap_or_else(|| self.series().is_sorted_flag())
    }

    /// The Series `first`/`last` (or the `ignore_nulls` variants) reduce: the one row they pick
    /// of a `Column::Dictionary`, or an empty one when there is none, so that the codes are not
    /// gathered. As [`Self::repeat_invariant_series`] for any other column.
    pub(crate) fn edge_series(&self, ignore_nulls: bool, last: bool) -> Series {
        let dictionary = match &*self.column.read() {
            c @ Column::Dictionary(_) => Some(c.clone()),
            _ => None,
        };
        if let Some(c) = dictionary {
            let at = match (ignore_nulls, last) {
                (true, false) => c.first_non_null(),
                (true, true) => c.last_non_null(),
                (false, false) => (!c.is_empty()).then_some(0),
                (false, true) => c.len().checked_sub(1),
            };
            let row = match at {
                Some(i) => c.slice((i as i64, 1usize)),
                None => c.slice((0i64, 0usize)),
            };
            return row.as_materialized_series().clone();
        }
        self.repeat_invariant_series(|n| n.min(1))
    }

    /// Take the column out, to go into a `DataFrame`, as `Series::into_column` did: a
    /// length-1 `Column::Series` becomes a `Column::Scalar`. A `Column::Scalar` is materialised
    /// first (its cache is shared) and converted the same way, so that the engine sees the
    /// column a plain Series would give (it computes other values and flags on a scalar
    /// column). A `Column::Dictionary` passes through.
    ///
    /// The conversion is kept: the scalar column also hands out a normalised `Series` (a
    /// sliced nested array, say, comes back without its offset).
    pub(crate) fn into_frame_column(self) -> Column {
        match self.column.into_inner() {
            Column::Series(s) => s.take().into(),
            Column::Scalar(sc) => sc.as_materialized_series().clone().into(),
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

/// Whether appending this column to, or after, another can leave a sorted flag in the Series
/// path (`update_sorted_flag_before_append`): it has at most one row, is all null, or is
/// flagged sorted. When both columns qualify, a `Column::Dictionary` append (which always
/// drops the flag) must take the Series path; otherwise the flag is dropped there too.
pub(crate) fn keeps_sorted_flag(c: &Column) -> bool {
    c.len() <= 1
        || c.null_count() == c.len()
        || c.is_sorted_flag() != polars_core::series::IsSorted::Not
}
