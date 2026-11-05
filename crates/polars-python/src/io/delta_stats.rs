//! Delta Lake statistics of a Parquet data file, from its footer.
//!
//! The Delta sink commits the files that the native Parquet sink wrote. The add
//! action of each file holds the min, max and null count of its columns as JSON,
//! which this module derives from the file's footer the way delta-rs writes it.
//! Decimals are the exception: delta-rs rounds them to a float, and they are exact
//! here, as Spark writes them.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io::Cursor;

use chrono::{DateTime, Datelike, NaiveDate, TimeDelta, Timelike};
use polars_compute::decimal::{DEC128_MAX_PREC, DecimalFmtBuffer};
use polars_error::{PolarsResult, to_compute_err};
use polars_parquet::parquet::schema::types::{
    IntegerType, PrimitiveConvertedType, PrimitiveLogicalType, PrimitiveType, TimeUnit,
};
use polars_parquet::parquet::statistics::Statistics;
use polars_parquet::read::read_metadata;
use polars_utils::aliases::PlHashSet;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use serde::Serialize;
use serde_json::value::RawValue;

use crate::error::PyPolarsErr;

/// A min or max in the statistics of a file.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum StatValue {
    Bool(bool),
    Int(i128),
    Float(f64),
    /// A decimal as a JSON number, in plain notation with all digits of its scale,
    /// e.g. `-1.50`.
    Decimal(Box<RawValue>),
    /// Strings, and dates and timestamps formatted as text.
    Str(String),
    /// JSON has no infinity, so delta-rs writes null for an infinite min or max.
    Null,
}

/// The statistics of a leaf column of a file, over all its row groups.
#[derive(Debug)]
pub struct LeafStats {
    /// The path of the column in the Parquet schema, e.g. `["a", "b"]` for field `b`
    /// of struct `a`.
    pub path: Vec<PlSmallStr>,
    /// None if a row group has no null count.
    pub null_count: Option<i64>,
    /// None if a row group has no usable min, e.g. for binary data or only NaNs.
    pub min: Option<StatValue>,
    /// None if a row group has no usable max.
    pub max: Option<StatValue>,
}

#[derive(Debug)]
pub struct FileStats {
    pub num_rows: usize,
    /// One per leaf column, in the order of the Parquet schema.
    pub leaves: Vec<LeafStats>,
}

/// The statistics of the Parquet file whose footer is at the end of `buf`.
///
/// `buf` holds the whole file, or only its footer, as the native sink passes it.
pub fn delta_file_stats(buf: &[u8]) -> PolarsResult<FileStats> {
    let metadata = read_metadata(&mut Cursor::new(buf))?;
    let mut leaves: Vec<LeafAcc> = Vec::new();

    for (rg_idx, row_group) in metadata.row_groups.iter().enumerate() {
        for (leaf_idx, chunk) in row_group.parquet_columns().iter().enumerate() {
            if rg_idx == 0 {
                leaves.push(LeafAcc::new(chunk.descriptor().path_in_schema.clone()));
            }
            let leaf = &mut leaves[leaf_idx];

            let null_count = chunk.null_count();
            leaf.null_count = leaf.null_count.zip(null_count).map(|(a, b)| a + b);

            let stats = chunk
                .statistics(&metadata.footer_buf)
                .and_then(|stats| stats.ok());
            match stats.as_ref().filter(|stats| has_min_max(stats)) {
                Some(stats) => {
                    leaf.min.update(bound(stats, true), Ordering::Less);
                    leaf.max.update(bound(stats, false), Ordering::Greater);
                },
                None => {
                    // A row group of only nulls has no min or max, but leaves them as
                    // they are. Otherwise they are unknown, e.g. for only NaNs.
                    let num_non_null = chunk
                        .num_values()
                        .ok()
                        .map(|num_values| num_values - null_count.unwrap_or(0));
                    if stats.is_none() || num_non_null.is_none_or(|n| n > 0) {
                        leaf.min = Extreme::Unusable;
                        leaf.max = Extreme::Unusable;
                    }
                },
            }
        }
    }

    Ok(FileStats {
        num_rows: metadata.num_rows,
        leaves: leaves.into_iter().map(LeafAcc::finish).collect(),
    })
}

struct LeafAcc {
    path: Vec<PlSmallStr>,
    null_count: Option<i64>,
    min: Extreme,
    max: Extreme,
}

impl LeafAcc {
    fn new(path: Vec<PlSmallStr>) -> Self {
        Self {
            path,
            null_count: Some(0),
            min: Extreme::Unset,
            max: Extreme::Unset,
        }
    }

    fn finish(self) -> LeafStats {
        LeafStats {
            path: self.path,
            null_count: self.null_count,
            min: self.min.into_value(),
            max: self.max.into_value(),
        }
    }
}

/// The min or max of a leaf over the row groups seen so far.
enum Extreme {
    Unset,
    /// A row group had no usable value, so the file has none.
    Unusable,
    Value(SortKey, StatValue),
}

impl Extreme {
    /// Combines the bound with the one of a row group. `keep` is the ordering of a
    /// new value that replaces the current one: `Less` for a min.
    fn update(&mut self, row_group: Option<(SortKey, StatValue)>, keep: Ordering) {
        *self = match (std::mem::replace(self, Extreme::Unusable), row_group) {
            (Extreme::Unusable, _) | (_, None) => Extreme::Unusable,
            (Extreme::Unset, Some((key, value))) => Extreme::Value(key, value),
            (Extreme::Value(cur_key, cur_value), Some((key, value))) => {
                if key.partial_cmp(&cur_key) == Some(keep) {
                    Extreme::Value(key, value)
                } else {
                    Extreme::Value(cur_key, cur_value)
                }
            },
        }
    }

    fn into_value(self) -> Option<StatValue> {
        match self {
            Extreme::Value(_, value) => Some(value),
            Extreme::Unset | Extreme::Unusable => None,
        }
    }
}

/// Orders the values of a leaf. Dates, timestamps and decimals order by their
/// physical value, as their Delta values are text or rounded.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum SortKey {
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
}

fn has_min_max(stats: &Statistics) -> bool {
    match stats {
        Statistics::Binary(s) => s.min_value.is_some() && s.max_value.is_some(),
        Statistics::Boolean(s) => s.min_value.is_some() && s.max_value.is_some(),
        Statistics::FixedLen(s) => s.min_value.is_some() && s.max_value.is_some(),
        Statistics::Int32(s) => s.min_value.is_some() && s.max_value.is_some(),
        Statistics::Int64(s) => s.min_value.is_some() && s.max_value.is_some(),
        Statistics::Int96(s) => s.min_value.is_some() && s.max_value.is_some(),
        Statistics::Float(s) => s.min_value.is_some() && s.max_value.is_some(),
        Statistics::Double(s) => s.min_value.is_some() && s.max_value.is_some(),
    }
}

/// The min or max of a row group, or None if Delta has no value for it.
fn bound(stats: &Statistics, is_min: bool) -> Option<(SortKey, StatValue)> {
    fn pick<T: Clone>(min: &Option<T>, max: &Option<T>, is_min: bool) -> Option<T> {
        if is_min { min.clone() } else { max.clone() }
    }

    match stats {
        Statistics::Boolean(s) => {
            let v = pick(&s.min_value, &s.max_value, is_min)?;
            Some((SortKey::Bool(v), StatValue::Bool(v)))
        },
        Statistics::Int32(s) => {
            let v = pick(&s.min_value, &s.max_value, is_min)?;
            match kind(&s.primitive_type) {
                Kind::Plain => int(i128::from(v)),
                Kind::Unsigned => int(i128::from(v as u32)),
                Kind::Date => date(v),
                Kind::Decimal(scale) => decimal(i128::from(v), scale),
                Kind::Timestamp { .. } | Kind::String | Kind::Unsupported => None,
            }
        },
        Statistics::Int64(s) => {
            let v = pick(&s.min_value, &s.max_value, is_min)?;
            match kind(&s.primitive_type) {
                Kind::Plain => int(i128::from(v)),
                Kind::Unsigned => int(i128::from(v as u64)),
                Kind::Decimal(scale) => decimal(i128::from(v), scale),
                Kind::Timestamp { unit, utc } => timestamp(v, unit, utc),
                Kind::Date | Kind::String | Kind::Unsupported => None,
            }
        },
        Statistics::Float(s) => float(f64::from(pick(&s.min_value, &s.max_value, is_min)?)),
        Statistics::Double(s) => float(pick(&s.min_value, &s.max_value, is_min)?),
        Statistics::Binary(s) => {
            bytes(pick(&s.min_value, &s.max_value, is_min)?, &s.primitive_type)
        },
        Statistics::FixedLen(s) => {
            bytes(pick(&s.min_value, &s.max_value, is_min)?, &s.primitive_type)
        },
        // Legacy timestamps, which Delta doesn't use.
        Statistics::Int96(_) => None,
    }
}

/// How the values of a leaf are interpreted, from its logical (or converted) type.
enum Kind {
    Plain,
    Unsigned,
    Date,
    /// The number of digits after the decimal point.
    Decimal(usize),
    Timestamp {
        unit: TimeUnit,
        utc: bool,
    },
    String,
    Unsupported,
}

fn kind(primitive_type: &PrimitiveType) -> Kind {
    use PrimitiveConvertedType as C;
    use PrimitiveLogicalType as L;

    if let Some(logical_type) = &primitive_type.logical_type {
        return match logical_type {
            L::String | L::Enum | L::Json => Kind::String,
            L::Decimal(_, scale) => Kind::Decimal(*scale),
            L::Date => Kind::Date,
            L::Timestamp {
                unit,
                is_adjusted_to_utc,
            } => Kind::Timestamp {
                unit: *unit,
                utc: *is_adjusted_to_utc,
            },
            L::Integer(
                IntegerType::UInt8
                | IntegerType::UInt16
                | IntegerType::UInt32
                | IntegerType::UInt64,
            ) => Kind::Unsigned,
            L::Integer(_) => Kind::Plain,
            L::Time { .. } | L::Unknown | L::Bson | L::Uuid | L::Float16 => Kind::Unsupported,
        };
    }

    match &primitive_type.converted_type {
        None | Some(C::Int8 | C::Int16 | C::Int32 | C::Int64) => Kind::Plain,
        Some(C::Uint8 | C::Uint16 | C::Uint32 | C::Uint64) => Kind::Unsigned,
        Some(C::Utf8 | C::Enum | C::Json) => Kind::String,
        Some(C::Decimal(_, scale)) => Kind::Decimal(*scale),
        Some(C::Date) => Kind::Date,
        Some(C::TimestampMillis) => Kind::Timestamp {
            unit: TimeUnit::Milliseconds,
            utc: true,
        },
        Some(C::TimestampMicros) => Kind::Timestamp {
            unit: TimeUnit::Microseconds,
            utc: true,
        },
        Some(C::TimeMillis | C::TimeMicros | C::Bson | C::Interval) => Kind::Unsupported,
    }
}

fn int(v: i128) -> Option<(SortKey, StatValue)> {
    Some((SortKey::Int(v), StatValue::Int(v)))
}

fn float(v: f64) -> Option<(SortKey, StatValue)> {
    if v.is_nan() {
        return None;
    }
    let value = if v.is_finite() {
        StatValue::Float(v)
    } else {
        StatValue::Null
    };
    Some((SortKey::Float(v), value))
}

/// A date, from the number of days since 1970-01-01, e.g. `2026-01-02`.
fn date(days: i32) -> Option<(SortKey, StatValue)> {
    let date = NaiveDate::from_ymd_opt(1970, 1, 1)?
        .checked_add_signed(TimeDelta::try_days(i64::from(days))?)?;
    if !(1..=9999).contains(&date.year()) {
        return None;
    }
    let text = date.format("%Y-%m-%d").to_string();
    Some((SortKey::Int(i128::from(days)), StatValue::Str(text)))
}

fn timestamp(raw: i64, unit: TimeUnit, utc: bool) -> Option<(SortKey, StatValue)> {
    let micros = match unit {
        TimeUnit::Nanoseconds => raw.div_euclid(1000),
        TimeUnit::Microseconds => raw,
        TimeUnit::Milliseconds => raw.checked_mul(1000)?,
    };
    let text = format_timestamp(micros, utc)?;
    Some((SortKey::Int(i128::from(micros)), StatValue::Str(text)))
}

/// A timestamp in microseconds since the epoch, as delta-rs writes it.
///
/// For example `2026-01-01 00:00:00.123`, or `2026-01-01T00:00:00.123Z` with `utc`.
/// Like Spark, delta-rs truncates the value to milliseconds, so readers widen the
/// max by 1 ms. The fraction is left out when it is zero. None for a year outside
/// 1 to 9999.
fn format_timestamp(micros: i64, utc: bool) -> Option<String> {
    let value = DateTime::from_timestamp_micros(micros)?.naive_utc();
    if !(1..=9999).contains(&value.year()) {
        return None;
    }
    let sub_micros = value.nanosecond() / 1000;
    let fraction = if sub_micros == 0 {
        String::new()
    } else {
        format!(".{:03}", sub_micros / 1000)
    };
    Some(if utc {
        format!("{}{fraction}Z", value.format("%Y-%m-%dT%H:%M:%S"))
    } else {
        format!("{}{fraction}", value.format("%Y-%m-%d %H:%M:%S"))
    })
}

fn bytes(v: Vec<u8>, primitive_type: &PrimitiveType) -> Option<(SortKey, StatValue)> {
    match kind(primitive_type) {
        Kind::String => {
            let text = String::from_utf8(v).ok()?;
            Some((SortKey::Str(text.clone()), StatValue::Str(text)))
        },
        Kind::Decimal(scale) => decimal(be_i128(&v)?, scale),
        // Binary data has no statistics in Delta.
        _ => None,
    }
}

/// The decimal `unscaled * 10^-scale`, in plain notation, e.g. `-0.05` for -5 at
/// scale 2. None for a scale above the maximum precision of Polars.
///
/// A float would be rounded, possibly to the wrong side of the min or max. An
/// exponent (`1E-8`) or a fraction at scale 0 (`5.0`) fails to parse in some readers.
fn decimal(unscaled: i128, scale: usize) -> Option<(SortKey, StatValue)> {
    if scale > DEC128_MAX_PREC {
        return None;
    }
    let mut buf = DecimalFmtBuffer::new();
    let text = buf.format_dec128(unscaled, scale, false, false);
    let number = RawValue::from_string(text.to_owned()).ok()?;
    Some((SortKey::Int(unscaled), StatValue::Decimal(number)))
}

/// A big-endian two's complement integer of up to 16 bytes.
fn be_i128(bytes: &[u8]) -> Option<i128> {
    if bytes.is_empty() || bytes.len() > 16 {
        return None;
    }
    let fill = if bytes[0] & 0x80 != 0 { 0xff } else { 0 };
    let mut buf = [fill; 16];
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(buf))
}

/// The `stats` of an add action, e.g.
/// `{"numRecords":2,"minValues":{"a":1},"maxValues":{"a":5},"nullCount":{"a":0}}`.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DeltaStats<'a> {
    num_records: usize,
    min_values: BTreeMap<&'a str, StatsNode<'a>>,
    max_values: BTreeMap<&'a str, StatsNode<'a>>,
    null_count: BTreeMap<&'a str, StatsNode<'a>>,
}

/// The statistics of a leaf column, or the fields of a struct.
#[derive(Serialize)]
#[serde(untagged)]
enum StatsNode<'a> {
    Value(&'a StatValue),
    NullCount(i64),
    Struct(BTreeMap<&'a str, StatsNode<'a>>),
}

/// Inserts the statistics of the leaf at `path` into `root`, nested in its structs.
fn insert<'a>(
    root: &mut BTreeMap<&'a str, StatsNode<'a>>,
    path: &'a [PlSmallStr],
    node: StatsNode<'a>,
) {
    let Some((name, parents)) = path.split_last() else {
        return;
    };
    let mut fields = root;
    for parent in parents {
        fields = match fields
            .entry(parent.as_str())
            .or_insert_with(|| StatsNode::Struct(BTreeMap::new()))
        {
            StatsNode::Struct(fields) => fields,
            // A leaf has no fields, so its path is never the start of another path.
            StatsNode::Value(_) | StatsNode::NullCount(_) => unreachable!(),
        };
    }
    fields.insert(name.as_str(), node);
}

/// The statistics of a file as the JSON of its add action, with the min, max and null
/// count of the leaf columns in `stats_leaves`.
pub fn delta_stats_json(
    stats: &FileStats,
    stats_leaves: &PlHashSet<Vec<PlSmallStr>>,
) -> PolarsResult<String> {
    let mut out = DeltaStats {
        num_records: stats.num_rows,
        min_values: BTreeMap::new(),
        max_values: BTreeMap::new(),
        null_count: BTreeMap::new(),
    };
    for leaf in stats
        .leaves
        .iter()
        .filter(|l| stats_leaves.contains(&l.path))
    {
        if let Some(min) = &leaf.min {
            insert(&mut out.min_values, &leaf.path, StatsNode::Value(min));
        }
        if let Some(max) = &leaf.max {
            insert(&mut out.max_values, &leaf.path, StatsNode::Value(max));
        }
        if let Some(null_count) = leaf.null_count {
            insert(
                &mut out.null_count,
                &leaf.path,
                StatsNode::NullCount(null_count),
            );
        }
    }
    serde_json::to_string(&out).map_err(to_compute_err)
}

/// The Delta statistics of a Parquet file, from its footer at the end of `buf`.
///
/// Returns `(num_rows, stats_json, null_counts)`:
/// - `stats_json` is the JSON of the add action, with the min, max and null count
///   of the leaf columns in `stats_leaves`, given by their Parquet paths.
/// - `null_counts` has a `(path, null_count)` tuple for every leaf column. The null
///   count is None if a row group has none.
#[pyfunction]
pub fn _delta_file_stats<'py>(
    py: Python<'py>,
    buf: &[u8],
    stats_leaves: Vec<Vec<String>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let stats = delta_file_stats(buf).map_err(PyPolarsErr::from)?;
    let stats_leaves = stats_leaves
        .into_iter()
        .map(|path| path.into_iter().map(PlSmallStr::from).collect())
        .collect();
    let stats_json = delta_stats_json(&stats, &stats_leaves).map_err(PyPolarsErr::from)?;

    let null_counts = PyList::empty(py);
    for leaf in &stats.leaves {
        let path = PyTuple::new(py, leaf.path.iter().map(PlSmallStr::as_str))?;
        null_counts.append((path, leaf.null_count))?;
    }
    (stats.num_rows, stats_json, null_counts).into_pyobject(py)
}
