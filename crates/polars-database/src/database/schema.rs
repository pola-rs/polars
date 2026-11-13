//! Schema normalisation. The planned schema is authoritative: every batch is cast to it, so
//! the output does not depend on which backend or machine produced it.

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail, polars_err};

use super::dialect::DialectName;

/// Normalises a dtype from a backend to the dtype Polars plans with.
pub fn normalize_dtype(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Datetime(tu, Some(tz)) if tz.as_str().is_empty() => DataType::Datetime(*tu, None),
        // The session time zone is UTC; drivers spell it differently.
        DataType::Datetime(tu, Some(tz))
            if matches!(
                tz.as_str(),
                "Etc/UTC" | "Etc/UCT" | "UCT" | "Universal" | "Etc/Universal" | "Zulu" | "Etc/Zulu"
            ) =>
        {
            DataType::Datetime(*tu, Some(TimeZone::UTC))
        },
        // Opaque driver types are read as their storage, except ADBC Postgres `NUMERIC`
        // (text), which is a decimal like with every other driver.
        DataType::Extension(ext, storage) => {
            if ext.name() == "arrow.opaque"
                && ext.serialize_metadata().is_some_and(|m| {
                    m.contains("\"type_name\": \"numeric\"")
                        || m.contains("\"type_name\":\"numeric\"")
                })
            {
                return UNCONSTRAINED_DECIMAL;
            }
            let _ = ext;
            normalize_dtype(storage)
        },
        // `numeric(p, s)[]`: the element too
        DataType::List(inner) => DataType::List(Box::new(normalize_dtype(inner))),
        dt => dt.clone(),
    }
}

/// The dtype of a `NUMERIC` without precision, whose values each have their own scale
/// (read as exact text). A value with more digits than the scale is an error; use
/// `schema_overrides` for another precision.
pub const UNCONSTRAINED_DECIMAL: DataType = DataType::Decimal(38, 9);

/// The dtype of a declared database type, where the declaration fixes it whatever the driver
/// returns. Drivers differ here: ADBC Postgres returns `NUMERIC` as text and ConnectorX as a
/// wider decimal; row-wise drivers return every integer as `Int64`.
///
/// - `NUMERIC(p, s)` / `DECIMAL(p, s)` with `p <= 38` is `Decimal(p, s)`;
/// - integer types map to their width (and signedness, for MySQL `UNSIGNED` and SQL Server
///   `TINYINT`); SQL Server `BIT` is `Boolean`.
pub fn declared_dtype(dialect: DialectName, db_type: &str) -> Option<DataType> {
    let mut t = db_type.trim().to_ascii_lowercase();
    // Arrays of a declared type: drivers pick the element width themselves otherwise.
    if matches!(dialect, DialectName::Postgresql | DialectName::Redshift)
        && let Some(inner) = t.strip_suffix("[]")
    {
        return declared_dtype(dialect, inner).map(|d| DataType::List(Box::new(d)));
    }
    if dialect == DialectName::Clickhouse {
        let written = clickhouse_unwrapped(db_type);
        t = written.to_ascii_lowercase();
        if t.starts_with("array(") {
            let inner = written["array(".len()..].strip_suffix(')')?;
            return declared_dtype(dialect, inner).map(|d| DataType::List(Box::new(d)));
        }
        if let Some(dt) = clickhouse_dtype(written) {
            return Some(dt);
        }
    }
    if let Some(dt) = declared_integer(dialect, &t) {
        return Some(dt);
    }
    if (matches!(dialect, DialectName::Postgresql | DialectName::Redshift) && t == "numeric")
        || (dialect == DialectName::Oracle && t == "number")
    {
        return Some(UNCONSTRAINED_DECIMAL);
    }
    // Oracle `TIMESTAMP(p) [WITH [LOCAL] TIME ZONE]`: the time zone types are read as UTC
    // (see `column_conversion`); 7 to 9 fractional digits are read as text, to keep them.
    if dialect == DialectName::Oracle && t.starts_with("timestamp") {
        let tz = t.contains("time zone").then_some(TimeZone::UTC);
        let unit = match oracle_timestamp_precision(&t) {
            Some(7..) => TimeUnit::Nanoseconds,
            _ => TimeUnit::Microseconds,
        };
        return Some(DataType::Datetime(unit, tz));
    }
    // Postgres `timestamptz` is an instant, read as UTC (a live connection's session zone
    // may be another: see `column_conversion`).
    if matches!(dialect, DialectName::Postgresql | DialectName::Redshift)
        && t.starts_with("timestamp")
        && t.ends_with("with time zone")
    {
        return Some(DataType::Datetime(
            TimeUnit::Microseconds,
            Some(TimeZone::UTC),
        ));
    }
    // Postgres `time`: its `24:00:00` is checked on the way (see `column_conversion`).
    if matches!(dialect, DialectName::Postgresql | DialectName::Redshift)
        && (t == "time without time zone"
            || (t.starts_with("time(") && t.ends_with(") without time zone")))
    {
        return Some(DataType::Time);
    }
    // Drivers disagree on the width of MySQL `YEAR` and `FLOAT` (4 bytes).
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore) {
        if t == "year" || t.starts_with("year(") {
            return Some(DataType::Int16);
        }
        // `FLOAT(M,D)` too: some drivers read it rounded to D places (a Float64), others
        // as stored; cast to Float32, the rounded value is the stored one. (`FLOAT(p)` is
        // shown as `float` or `double`.)
        if t == "float" || t.starts_with("float ") || t.starts_with("float(") {
            return Some(DataType::Float32);
        }
    }
    // MySQL `TIME` is a duration (-838:59:59 to 838:59:59), not a time of day.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && (t == "time" || t.starts_with("time("))
    {
        return Some(DataType::Duration(TimeUnit::Microseconds));
    }
    // MySQL `TIMESTAMP` is stored in UTC (the session time zone is UTC); `DATETIME` is naive.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && (t == "timestamp" || t.starts_with("timestamp("))
    {
        return Some(DataType::Datetime(
            TimeUnit::Microseconds,
            Some(TimeZone::UTC),
        ));
    }
    if dialect == DialectName::Oracle {
        // `DATE` has a time part; `NUMBER(p, 0)` with p <= 18 always fits an Int64.
        if t == "date" {
            return Some(DataType::Datetime(TimeUnit::Microseconds, None));
        }
        if let Some(args) = t.strip_prefix("number(").and_then(|r| r.strip_suffix(')'))
            && let Some((p, s)) = args.split_once(',')
            && let (Ok(p), Ok(s)) = (p.trim().parse::<usize>(), s.trim().parse::<i64>())
            && s <= 0
            // `NUMBER(p, -s)` rounds to 10^s: integers of up to p + s digits (more than 38
            // are read as text, see `text_conversion`)
            && let digits = p + s.unsigned_abs() as usize
            && digits <= 38
        {
            return Some(if digits <= 18 {
                DataType::Int64
            } else {
                DataType::Decimal(digits, 0)
            });
        }
    }
    // MySQL `decimal(30,15) unsigned zerofill`: the attributes don't change the values.
    let mut t = t.as_str();
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore) {
        while let Some(rest) = t
            .strip_suffix("unsigned")
            .or_else(|| t.strip_suffix("zerofill"))
        {
            t = rest.trim_end();
        }
    }
    let args = ["numeric(", "decimal(", "number("]
        .iter()
        .find_map(|p| t.strip_prefix(p))?;
    let args = args.strip_suffix(')')?;
    let (p, s) = match args.split_once(',') {
        Some((p, s)) => (
            p.trim().parse::<usize>().ok()?,
            s.trim().parse::<usize>().ok()?,
        ),
        None => (args.trim().parse::<usize>().ok()?, 0),
    };
    // Oracle (and Postgres 15+) allow a scale above the precision: `NUMBER(5, 7)` holds
    // 0.0012345, a `Decimal(7, 7)`
    if s > p && (1..=38).contains(&s) {
        return Some(DataType::Decimal(s, s));
    }
    if !(1..=38).contains(&p) || s > p {
        return None;
    }
    Some(DataType::Decimal(p, s))
}

/// The fractional second digits of an Oracle `TIMESTAMP(p) …`; `None` without them (6).
pub fn oracle_timestamp_precision(t: &str) -> Option<u32> {
    t.strip_prefix("timestamp(")?
        .split_once(')')?
        .0
        .trim()
        .parse()
        .ok()
}

/// Casts a batch to the planned schema: selects the planned columns in order and casts each.
/// The columns in `overrides` (`schema_overrides`) are cast as the user asked, also when
/// a float becomes an integer or a decimal loses places (as Polars' cast rounds it).
pub fn cast_to_schema(
    df: &DataFrame,
    schema: &Schema,
    overrides: Option<&Schema>,
) -> PolarsResult<DataFrame> {
    let height = df.height();
    let mut columns = Vec::with_capacity(schema.len());
    for (name, dtype) in schema.iter() {
        let col = df.column(name).map_err(|_| {
            polars_err!(
                SchemaMismatch:
                "database result is missing column '{name}'"
            )
        })?;
        let storage;
        let col = if has_extension(col.dtype()) {
            storage = without_extensions(col.as_materialized_series())?.into_column();
            &storage
        } else {
            col
        };
        let converted;
        let col = match duration_to_time(col, dtype)? {
            Some(c) => {
                converted = c;
                &converted
            },
            None => col,
        };
        let parsed;
        let col = match parse_temporal(col, dtype)? {
            Some(p) => {
                parsed = p;
                &parsed
            },
            None => col,
        };
        let col = if col.dtype() == dtype {
            col.clone()
        } else if col.dtype() == &DataType::Null {
            Column::full_null(name.clone(), height, dtype)
        } else {
            let cast = col
                .cast_with_options(dtype, CastOptions::Strict)
                .map_err(|e| {
                    polars_err!(
                        SchemaMismatch:
                        "database column '{name}' is {} but {dtype} was planned: {e}",
                        col.dtype()
                    )
                })?;
            if overrides.is_none_or(|o| !o.contains(name)) {
                check_decimal_exact(col, &cast)?;
                check_numeric_exact(col, &cast)?;
            }
            cast
        };
        columns.push(col);
    }
    DataFrame::new(height, columns)
}

fn has_extension(dtype: &DataType) -> bool {
    match dtype {
        DataType::Extension(_, _) => true,
        DataType::List(inner) => has_extension(inner),
        _ => false,
    }
}

/// A column with its extension types (ADBC Postgres `numeric`, also in arrays) as their
/// storage.
fn without_extensions(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Extension(_, _) => without_extensions(s.to_storage()),
        DataType::List(inner) if has_extension(inner) => Ok(s
            .list()?
            .apply_to_inner(&|inner| without_extensions(&inner))?
            .into_series()),
        _ => Ok(s.clone()),
    }
}

/// A decimal cast must not round an exact value (text or a decimal of a larger scale):
/// unconstrained `numeric` / `NUMBER` are planned as `Decimal(38, 9)`, and a value with
/// more digits is an error, not silently another number. Floats only approximate a
/// decimal, so rounding them to the declared scale is their reading.
fn check_decimal_exact(src: &Column, cast: &Column) -> PolarsResult<()> {
    // `numeric(p, s)[]`: its elements
    if let (DataType::List(_), DataType::List(_)) = (src.dtype(), cast.dtype()) {
        let inner = |c: &Column| -> PolarsResult<Column> {
            Ok(c.list()?
                .get_inner()
                .with_name(c.name().clone())
                .into_column())
        };
        return check_decimal_exact(&inner(src)?, &inner(cast)?);
    }
    let DataType::Decimal(_, scale) = cast.dtype() else {
        return Ok(());
    };
    let bad = match src.dtype() {
        DataType::String => src
            .str()?
            .iter()
            .flatten()
            .find(|v| decimal_text_scale(v) > *scale)
            .map(str::to_string),
        DataType::Decimal(_, src_scale) if src_scale > scale => {
            let back = cast.cast(src.dtype())?;
            let same = src
                .as_materialized_series()
                .equal_missing(back.as_materialized_series())?;
            match same.iter().position(|v| v == Some(false)) {
                Some(i) => Some(src.get(i)?.to_string()),
                None => None,
            }
        },
        _ => None,
    };
    match bad {
        Some(v) => polars_bail!(
            SchemaMismatch:
            "database column '{}' has the value {v}, which {} would round; read the column \
            as text with schema_overrides={{'{}': pl.String}} (or as pl.Float64)",
            src.name(), cast.dtype(), src.name()
        ),
        None => Ok(()),
    }
}

/// The decimal places a decimal's text needs, also in scientific notation (Oracle writes
/// `1E-100`); 0 for text that isn't a number (the cast fails on it).
fn decimal_text_scale(v: &str) -> usize {
    let v = v.trim();
    let (mantissa, exp) = match v.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i64>().unwrap_or(0)),
        None => (v, 0),
    };
    let frac = mantissa
        .split_once('.')
        .map_or(0, |(_, f)| f.trim_end_matches('0').len()) as i64;
    // `1.5E+3` has no decimal places; `15E-1` one
    let digits = if frac == 0 {
        let int = mantissa.split('.').next().unwrap_or("");
        -(int.len() as i64 - int.trim_end_matches('0').len() as i64)
    } else {
        frac
    };
    (digits - exp).max(0) as usize
}

/// A cast between integers and floats (or from a decimal to an integer) must not round
/// either: a strict cast truncates `1000.5` to `1000`, and `2**53 + 1` becomes
/// `9007199254740992.0`. Columns typed from a sample (SQLite, which has no column types)
/// may hold values of both kinds. Nor may a `schema=` dtype coarser than the column cut a
/// value (a `Datetime("ms")` or `Date` over microseconds) or overflow (`Float32`); the
/// nearest float of a decimal or double is its reading.
fn check_numeric_exact(src: &Column, cast: &Column) -> PolarsResult<()> {
    let (from, to) = (src.dtype(), cast.dtype());
    if from.is_float() && to == &DataType::Float32 {
        // the nearest `Float32` is the reading, but not an overflow to infinity
        let finite = src.as_materialized_series().is_finite()?;
        let overflow = cast.as_materialized_series().is_infinite()? & finite;
        if let Some(i) = overflow.iter().position(|v| v == Some(true)) {
            polars_bail!(
                SchemaMismatch:
                "database column '{}' has the value {}, which is out of the range of Float32; \
                read the column as pl.Float64",
                src.name(), src.get(i)?
            )
        }
        return Ok(());
    }
    // a time of day drops the date
    if matches!(from, DataType::Datetime(_, _)) && to == &DataType::Time {
        let days = src.cast(&DataType::Date)?.cast(&DataType::Int32)?;
        if let Some(i) = days.i32()?.iter().position(|d| d.is_some_and(|d| d != 0)) {
            polars_bail!(
                SchemaMismatch:
                "database column '{}' has the value {}, whose date {to} would drop; read the \
                column as {from}",
                src.name(), src.get(i)?
            )
        }
        return Ok(());
    }
    let coarser_time = match (from, to) {
        (DataType::Datetime(f, _), DataType::Datetime(t, _))
        | (DataType::Duration(f), DataType::Duration(t)) => time_unit_ns(*t) > time_unit_ns(*f),
        (DataType::Datetime(_, _), DataType::Date) => true,
        _ => false,
    };
    if coarser_time {
        let back = cast.cast(from)?;
        let same = src
            .as_materialized_series()
            .equal_missing(back.as_materialized_series())?;
        if let Some(i) = same.iter().position(|v| v != Some(true)) {
            polars_bail!(
                SchemaMismatch:
                "database column '{}' has the value {}, which {to} would cut; read the column \
                as {from}",
                src.name(), src.get(i)?
            )
        }
        return Ok(());
    }
    let lossy = (to.is_integer() && (from.is_float() || from.is_decimal()))
        || (to.is_float() && from.is_integer());
    if !lossy {
        return Ok(());
    }
    let back = cast.cast(from)?;
    let same = src
        .as_materialized_series()
        .equal_missing(back.as_materialized_series())?;
    let Some(i) = same.iter().position(|v| v != Some(true)) else {
        return Ok(());
    };
    let alt = if to.is_integer() {
        "pl.Float64"
    } else {
        "pl.Int64"
    };
    polars_bail!(
        SchemaMismatch:
        "database column '{}' has the value {}, which {to} would round; read the column with \
        schema_overrides={{'{}': {alt}}} (or as pl.String)",
        src.name(), src.get(i)?, src.name()
    )
}

fn time_unit_ns(tu: TimeUnit) -> i64 {
    match tu {
        TimeUnit::Nanoseconds => 1,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Milliseconds => 1_000_000,
    }
}

/// A `TIME` read as a duration (PyMySQL returns `timedelta`s) as a time of day. MySQL `TIME`
/// may be negative or exceed 24 hours: such values are an error.
fn duration_to_time(col: &Column, dtype: &DataType) -> PolarsResult<Option<Column>> {
    if let (DataType::Duration(tu), DataType::Time) = (col.dtype(), dtype) {
        let ns_per = match tu {
            TimeUnit::Nanoseconds => 1,
            TimeUnit::Microseconds => 1_000,
            TimeUnit::Milliseconds => 1_000_000,
        };
        let phys = col.to_physical_repr();
        let values = phys.i64()?;
        let day = 86_400_000_000_000i64;
        if let Some(bad) = values
            .iter()
            .flatten()
            .find(|v| !(0..day).contains(&(v * ns_per)))
        {
            polars_bail!(
                SchemaMismatch:
                "database column '{}' has a duration of {bad} {tu}, which is not a time of day",
                col.name()
            );
        }
        let ns = values.apply_values(|v| v * ns_per).into_series();
        return Ok(Some(ns.cast(&DataType::Time)?.into_column()));
    }
    let _ = (col, dtype);
    Ok(None)
}

/// Text of a temporal column (SQLite stores dates as text, some conversions write text)
/// parsed to the planned dtype.
/// Polars 2.0 has no string-to-temporal cast. A value that doesn't parse is an error.
fn parse_temporal(col: &Column, dtype: &DataType) -> PolarsResult<Option<Column>> {
    use polars_core::chunked_array::temporal::string::StringMethods;

    if !col.dtype().is_string() {
        return Ok(None);
    }
    let ca = col.str()?;
    let parsed: Series = match dtype {
        DataType::Date => ca.as_date(Some("%Y-%m-%d"), true)?.into_series(),
        // Text without an offset is UTC, as a naive value cast to a time zone is (and as
        // the conversions write it: Oracle and ClickHouse time zones).
        DataType::Datetime(tu, tz) => {
            let raise = StringChunked::from_slice(PlSmallStr::EMPTY, &["raise"]);
            let naive = if *tu == TimeUnit::Nanoseconds {
                // The parse overflows near the ends of the nanosecond range (1677 to
                // 2262): the microseconds are parsed, and the 7th to 9th fractional
                // digits added. A value outside the range is an error.
                let us = ca.as_datetime(None, TimeUnit::Microseconds, true, false, None, &raise)?;
                let mut ns = Vec::with_capacity(ca.len());
                for (i, (v, text)) in us.physical().iter().zip(ca.iter()).enumerate() {
                    let Some(v) = v else {
                        ns.push(None);
                        continue;
                    };
                    // (in i128: the last microsecond before the minimum starts below it)
                    let ns_value = v as i128 * 1000 + sub_microsecond_ns(text) as i128;
                    let Ok(v) = i64::try_from(ns_value) else {
                        polars_bail!(
                            SchemaMismatch:
                            "database column '{}' has the value {}, outside the range of \
                            {dtype} (1677 to 2262); read the column with \
                            schema_overrides={{'{}': pl.Datetime('us')}}",
                            col.name(), ca.get(i).unwrap_or_default(), col.name()
                        );
                    };
                    ns.push(Some(v));
                }
                Int64Chunked::from_iter_options(col.name().clone(), ns.into_iter())
                    .into_datetime(TimeUnit::Nanoseconds, None)
                    .into_series()
            } else {
                ca.as_datetime(None, *tu, true, false, None, &raise)?
                    .into_series()
            };
            match tz {
                Some(_) => naive.cast(dtype)?,
                None => naive,
            }
        },
        DataType::Time => ca.as_time(None, true)?.into_series(),
        _ => return Ok(None),
    };
    if parsed.null_count() != ca.null_count() {
        let bad = ca
            .iter()
            .zip(parsed.is_null().iter())
            .find_map(|(s, null)| (null == Some(true)).then_some(s).flatten())
            .unwrap_or_default();
        polars_bail!(
            SchemaMismatch:
            "database column '{}' has text '{bad}', which is not a {dtype}",
            col.name()
        );
    }
    Ok(Some(parsed.into_column()))
}

/// The nanoseconds of a timestamp's text beyond its microseconds: its 7th to 9th
/// fractional digits.
fn sub_microsecond_ns(text: Option<&str>) -> i64 {
    let digits = || {
        let digits = text?
            .rsplit_once('.')?
            .1
            .chars()
            .take_while(char::is_ascii_digit)
            .chain(std::iter::repeat('0'))
            .skip(6)
            .take(3)
            .collect::<String>();
        digits.parse::<i64>().ok()
    };
    digits().unwrap_or(0)
}

/// A ClickHouse type without its `Nullable(…)` and `LowCardinality(…)` wrappers, lower-cased.
pub fn clickhouse_value_type(db_type: &str) -> String {
    clickhouse_unwrapped(db_type).to_ascii_lowercase()
}

/// A ClickHouse type without its wrappers, as written: time zone names are case-sensitive.
fn clickhouse_unwrapped(db_type: &str) -> &str {
    let mut t = db_type.trim();
    loop {
        let inner = ["nullable(", "lowcardinality("].iter().find_map(|p| {
            let rest = t
                .get(..p.len())?
                .eq_ignore_ascii_case(p)
                .then(|| &t[p.len()..])?;
            rest.strip_suffix(')')
        });
        match inner {
            Some(inner) => t = inner.trim(),
            None => return t,
        }
    }
}

/// The time zone argument of a ClickHouse `DateTime(…)` / `DateTime64(…)`, as written.
fn clickhouse_time_zone(args: &str) -> Option<TimeZone> {
    let tz = args.trim().trim_matches('\'');
    if tz.eq_ignore_ascii_case("utc") {
        return Some(TimeZone::UTC);
    }
    TimeZone::opt_try_new(Some(tz)).ok()?
}

/// ClickHouse value types whose dtype the declaration fixes. `t` is unwrapped, as written.
fn clickhouse_dtype(written: &str) -> Option<DataType> {
    use DataType as D;
    let lower = written.to_ascii_lowercase();
    let t = lower.as_str();
    // the arguments as written
    let args = |prefix: &str| {
        t.starts_with(prefix)
            .then(|| written[prefix.len()..].strip_suffix(')'))
            .flatten()
    };
    Some(match t {
        "int8" => D::Int8,
        "int16" => D::Int16,
        "int32" => D::Int32,
        "int64" => D::Int64,
        "uint8" => D::UInt8,
        "uint16" => D::UInt16,
        "uint32" => D::UInt32,
        "uint64" => D::UInt64,
        "float32" => D::Float32,
        "float64" => D::Float64,
        "bool" => D::Boolean,
        "date" | "date32" => D::Date,
        "string" => D::String,
        // read as text (`text_conversion`)
        _ if t.starts_with("enum8(") || t.starts_with("enum16(") => D::String,
        // seconds; read through `toDateTime64(…, 0)` (`column_conversion`)
        // An unknown zone name keeps the driver's dtype.
        "datetime" => D::Datetime(TimeUnit::Milliseconds, None),
        _ if t.starts_with("datetime(") => D::Datetime(
            TimeUnit::Milliseconds,
            Some(clickhouse_time_zone(args("datetime(")?)?),
        ),
        _ => {
            let args = args("datetime64(")?;
            let (precision, tz) = match args.split_once(',') {
                Some((p, tz)) => (p, Some(clickhouse_time_zone(tz)?)),
                None => (args, None),
            };
            let unit = match precision.trim().parse::<u32>().ok()? {
                0..=3 => TimeUnit::Milliseconds,
                4..=6 => TimeUnit::Microseconds,
                _ => TimeUnit::Nanoseconds,
            };
            D::Datetime(unit, tz)
        },
    })
}

fn declared_integer(dialect: DialectName, t: &str) -> Option<DataType> {
    use DataType as D;
    let unsigned = t.contains("unsigned");
    // `int(11)`, `tinyint(1) unsigned`: the display width doesn't matter.
    let base = t.split(['(', ' ']).next()?;
    let (signed, unsigned_dt) = match (dialect, base) {
        (DialectName::Mssql, "tinyint") => return Some(D::UInt8),
        (DialectName::Mssql, "bit") => return Some(D::Boolean),
        (DialectName::Mysql | DialectName::Singlestore, "tinyint") => (D::Int8, D::UInt8),
        (_, "smallint" | "int2") => (D::Int16, D::UInt16),
        (DialectName::Mysql | DialectName::Singlestore, "mediumint") => (D::Int32, D::UInt32),
        (_, "integer" | "int" | "int4") => (D::Int32, D::UInt32),
        (_, "bigint" | "int8") => (D::Int64, D::UInt64),
        _ => return None,
    };
    let is_mysql = matches!(dialect, DialectName::Mysql | DialectName::Singlestore);
    if !matches!(
        dialect,
        DialectName::Postgresql | DialectName::Redshift | DialectName::Mssql
    ) && !is_mysql
    {
        return None;
    }
    Some(if unsigned && is_mysql {
        unsigned_dt
    } else {
        signed
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inexact_numeric_casts_fail() {
        let cast = |col: Column, dtype: DataType| {
            let df = DataFrame::new(col.len(), vec![col]).unwrap();
            cast_to_schema(
                &df,
                &Schema::from_iter([Field::new("x".into(), dtype)]),
                None,
            )
        };
        let floats = |v: &[f64]| Column::new("x".into(), v);
        assert!(cast(floats(&[1.0, 2.0]), DataType::Int64).is_ok());
        let e = cast(floats(&[1.0, 1000.5]), DataType::Int64).unwrap_err();
        assert!(e.to_string().contains("1000.5"), "{e}");
        let ints = |v: &[i64]| Column::new("x".into(), v);
        assert!(cast(ints(&[1, 1 << 53]), DataType::Float64).is_ok());
        assert!(cast(ints(&[1, (1 << 53) + 1]), DataType::Float64).is_err());
        let with_null = Column::new("x".into(), &[Some(1.0), None]);
        assert!(cast(with_null, DataType::Int64).is_ok());
    }

    #[test]
    fn decimal_text_in_scientific_notation() {
        assert_eq!(decimal_text_scale("1.25"), 2);
        assert_eq!(decimal_text_scale("100"), 0);
        assert_eq!(decimal_text_scale("1E-100"), 100);
        assert_eq!(decimal_text_scale("1.5E-70"), 71);
        assert_eq!(decimal_text_scale("1.5E+25"), 0);
        assert_eq!(decimal_text_scale("100E-2"), 0);
        assert_eq!(decimal_text_scale("15e-1"), 1);
        let text = Column::new("n".into(), &["1E-100", "1"]);
        let df = DataFrame::new(2, vec![text]).unwrap();
        let schema = Schema::from_iter([Field::new("n".into(), UNCONSTRAINED_DECIMAL)]);
        let e = cast_to_schema(&df, &schema, None).unwrap_err();
        assert!(e.to_string().contains("1E-100"), "{e}");
    }

    #[test]
    fn nanoseconds_out_of_range_fail() {
        let ns = DataType::Datetime(TimeUnit::Nanoseconds, None);
        let cast = |v: &str| {
            let df = DataFrame::new(1, vec![Column::new("t".into(), &[v])]).unwrap();
            cast_to_schema(
                &df,
                &Schema::from_iter([Field::new("t".into(), ns.clone())]),
                None,
            )
        };
        assert!(cast("2024-01-02 03:04:05.123456789").is_ok());
        let e = cast("9999-12-31 23:59:59.999999999").unwrap_err();
        assert!(e.to_string().contains("outside the range"), "{e}");
        assert!(cast("1600-01-01 00:00:00.5").is_err());
        assert!(cast("-2000-01-01 00:00:00").is_err());
        // the ends of the range are exact, to the nanosecond
        let value = |v: &str| {
            cast(v).unwrap()[0]
                .as_materialized_series()
                .to_physical_repr()
                .i64()
                .unwrap()
                .get(0)
        };
        assert_eq!(value("2262-04-11 23:47:16.854775807"), Some(i64::MAX));
        assert_eq!(value("2262-04-11 23:47:16.5"), Some(i64::MAX - 354775807));
        assert!(cast("2262-04-11 23:47:16.854775808").is_err());
        assert_eq!(value("1677-09-21 00:12:43.145224192"), Some(i64::MIN));
        assert_eq!(value("1677-09-21 00:12:43.9"), Some(i64::MIN + 754775808));
        assert!(cast("1677-09-21 00:12:43.145224191").is_err());
        assert_eq!(value("1969-12-31 23:59:59.999999999"), Some(-1));
    }

    #[test]
    fn declared_types() {
        {
            let pg = DialectName::Postgresql;
            assert_eq!(
                declared_dtype(pg, "numeric(12,2)"),
                Some(DataType::Decimal(12, 2))
            );
            assert_eq!(
                declared_dtype(pg, "DECIMAL(10, 0)"),
                Some(DataType::Decimal(10, 0))
            );
            assert_eq!(
                declared_dtype(DialectName::Oracle, "number(19,0)"),
                Some(DataType::Decimal(19, 0))
            );
            assert_eq!(
                declared_dtype(DialectName::Oracle, "number(10,0)"),
                Some(DataType::Int64)
            );
        }
        let pg = DialectName::Postgresql;
        assert_eq!(declared_dtype(pg, "numeric"), Some(UNCONSTRAINED_DECIMAL));
        assert_eq!(declared_dtype(pg, "numeric(50,2)"), None);
        assert_eq!(declared_dtype(pg, "text"), None);
        assert_eq!(declared_dtype(pg, "integer"), Some(DataType::Int32));
        let my = DialectName::Mysql;
        assert_eq!(declared_dtype(my, "tinyint(1)"), Some(DataType::Int8));
        assert_eq!(
            declared_dtype(my, "int(10) unsigned"),
            Some(DataType::UInt32)
        );
        assert_eq!(
            declared_dtype(my, "bigint unsigned"),
            Some(DataType::UInt64)
        );
        assert_eq!(
            declared_dtype(my, "decimal(30,15) unsigned"),
            Some(DataType::Decimal(30, 15))
        );
        assert_eq!(
            declared_dtype(my, "DECIMAL(20,5) UNSIGNED ZEROFILL"),
            Some(DataType::Decimal(20, 5))
        );
        let ms = DialectName::Mssql;
        assert_eq!(declared_dtype(ms, "tinyint"), Some(DataType::UInt8));
        assert_eq!(declared_dtype(ms, "bit"), Some(DataType::Boolean));
        assert_eq!(declared_dtype(DialectName::Sqlite, "integer"), None);
        let ch = DialectName::Clickhouse;
        assert_eq!(declared_dtype(ch, "Nullable(Int16)"), Some(DataType::Int16));
        assert_eq!(
            declared_dtype(ch, "LowCardinality(Nullable(String))"),
            Some(DataType::String)
        );
        assert_eq!(
            declared_dtype(ch, "Nullable(Decimal(18, 4))"),
            Some(DataType::Decimal(18, 4))
        );
        assert_eq!(
            declared_dtype(ch, "Nullable(DateTime64(6))"),
            Some(DataType::Datetime(TimeUnit::Microseconds, None))
        );
        assert_eq!(
            declared_dtype(ch, "DateTime64(3, 'UTC')"),
            Some(DataType::Datetime(
                TimeUnit::Milliseconds,
                Some(TimeZone::UTC)
            ))
        );
        // a zone's name keeps its case
        let tokyo = TimeZone::opt_try_new(Some("Asia/Tokyo")).unwrap();
        assert_eq!(
            declared_dtype(ch, "Nullable(DateTime64(9, 'Asia/Tokyo'))"),
            Some(DataType::Datetime(TimeUnit::Nanoseconds, tokyo.clone()))
        );
        assert_eq!(
            declared_dtype(ch, "Array(DateTime('Asia/Tokyo'))"),
            Some(DataType::List(Box::new(DataType::Datetime(
                TimeUnit::Milliseconds,
                tokyo
            ))))
        );
        let ora = DialectName::Oracle;
        let ns = |tz| Some(DataType::Datetime(TimeUnit::Nanoseconds, tz));
        assert_eq!(declared_dtype(ora, "timestamp(9)"), ns(None));
        assert_eq!(
            declared_dtype(ora, "TIMESTAMP(7) WITH TIME ZONE"),
            ns(Some(TimeZone::UTC))
        );
        assert_eq!(
            declared_dtype(ora, "timestamp(6) with local time zone"),
            Some(DataType::Datetime(
                TimeUnit::Microseconds,
                Some(TimeZone::UTC)
            ))
        );
        assert_eq!(
            declared_dtype(ora, "timestamp"),
            Some(DataType::Datetime(TimeUnit::Microseconds, None))
        );
        assert_eq!(
            declared_dtype(pg, "time(3) without time zone"),
            Some(DataType::Time)
        );
        assert_eq!(declared_dtype(pg, "timestamp without time zone"), None);
    }
}
