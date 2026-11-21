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
        DataType::Array(inner, width) => DataType::Array(Box::new(normalize_dtype(inner)), *width),
        DataType::Map(key, value) => DataType::Map(
            Box::new(normalize_dtype(key)),
            Box::new(normalize_dtype(value)),
        ),
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| Field::new(f.name().clone(), normalize_dtype(f.dtype())))
                .collect(),
        ),
        dt => dt.clone(),
    }
}

/// The dtype of a `NUMERIC` without precision, whose values each have their own scale
/// (read as exact text). A value with more digits than the scale is an error; use
/// `schema_overrides` for another precision.
pub const UNCONSTRAINED_DECIMAL: DataType = DataType::Decimal(38, 9);

/// The base type (`datetime2`, `datetimeoffset` or `time`) of a SQL Server column of 7
/// fractional digits, which a bare name has (the scale defaults to 7): the ones drivers cut
/// to microseconds.
pub fn mssql_nanosecond_type(db_type: &str) -> Option<&'static str> {
    let t = db_type.trim().to_ascii_lowercase();
    let (base, args) = t.split_once('(').unwrap_or((&t, ""));
    let base = match base.trim() {
        "datetime2" => "datetime2",
        "datetimeoffset" => "datetimeoffset",
        "time" => "time",
        _ => return None,
    };
    let args = args.trim_end_matches(')').trim();
    (args.is_empty() || args.parse::<u32>().is_ok_and(|scale| scale > 6)).then_some(base)
}

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
    // Trino `timestamp(p) with time zone` (p <= 6; wider ones are read as text): an instant,
    // read as UTC (see `column_conversion`): a value carries a zone of its own, and the
    // drivers take the first of two instants of a named zone's repeated hour.
    if dialect == DialectName::Trino
        && t.starts_with("timestamp(")
        && t.ends_with(" with time zone")
    {
        return Some(DataType::Datetime(
            TimeUnit::Microseconds,
            Some(TimeZone::UTC),
        ));
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
    // SQL Server's `datetime2`, `datetime` and `smalldatetime` have no time zone: ADBC reads
    // them as UTC timestamps, the others as naive ones. A `real` is read as a `float`, as
    // the row drivers do (the stored value, which a pushed `float` literal compares with
    // exactly). A query the drivers can't type from rows (a table-valued function) reads
    // float and date columns as text otherwise.
    if dialect == DialectName::Mssql {
        // 7 fractional digits (100 ns): read as text, as nanoseconds (see `column_conversion`)
        match mssql_nanosecond_type(&t) {
            Some("time") => return Some(DataType::Time),
            Some("datetimeoffset") => {
                return Some(DataType::Datetime(
                    TimeUnit::Nanoseconds,
                    Some(TimeZone::UTC),
                ));
            },
            Some(_) => return Some(DataType::Datetime(TimeUnit::Nanoseconds, None)),
            None => {},
        }
        match t.split('(').next().unwrap_or("").trim() {
            "datetime" | "smalldatetime" | "datetime2" => {
                return Some(DataType::Datetime(TimeUnit::Microseconds, None));
            },
            "float" | "real" => return Some(DataType::Float64),
            // ConnectorX reads them as doubles, the other drivers as decimals of other
            // precisions
            "money" => return Some(DataType::Decimal(19, 4)),
            "smallmoney" => return Some(DataType::Decimal(10, 4)),
            "date" => return Some(DataType::Date),
            _ => {},
        }
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
    cast_to_user_schema(df, schema, overrides, false, &PlHashMap::new())
}

/// [`cast_to_schema`], where `user_schema` says the planned schema is the user's `schema=`.
/// A column the user typed (`schema=`, or in `overrides`) isn't cast from a number to a
/// temporal type: the number would be read as raw ticks of the unit (`1700000000` seconds as
/// `1970-01-01 00:28:20` microseconds, or days out of the calendar's range), silently. The
/// database's own conversions (a MySQL `TIME` as microseconds, ClickHouse arrays as
/// nanoseconds) plan the type themselves. `own_numbers`: the columns the plugin's own
/// conversion reads as numbers of a temporal database type, with the dtype whose ticks they are
/// (nanoseconds since the epoch, a duration's microseconds): an override in that unit
/// (`Datetime("ns", "UTC")`) is no guess.
pub fn cast_to_user_schema(
    df: &DataFrame,
    schema: &Schema,
    overrides: Option<&Schema>,
    user_schema: bool,
    own_numbers: &PlHashMap<PlSmallStr, DataType>,
) -> PolarsResult<DataFrame> {
    cast_columns(
        df,
        schema,
        overrides,
        user_schema,
        own_numbers,
        &PlHashSet::new(),
        &PlHashMap::new(),
        &PlHashSet::new(),
    )
}

/// [`cast_to_user_schema`], where `plugin_converted` are the columns the plugin reads through a
/// conversion of its own (a number as text, a decimal as text): their dtype is not the database's,
/// so a text read for a number is no change of the column's kind. `truncating`: the temporal
/// columns read as text whose digits below the planned unit are cut (toward the earlier
/// instant), as the user asked (see `truncating_columns`).
#[allow(clippy::too_many_arguments)]
pub fn cast_columns(
    df: &DataFrame,
    schema: &Schema,
    overrides: Option<&Schema>,
    user_schema: bool,
    own_numbers: &PlHashMap<PlSmallStr, DataType>,
    plugin_converted: &PlHashSet<PlSmallStr>,
    time_text_risk: &PlHashMap<PlSmallStr, String>,
    truncating: &PlHashSet<PlSmallStr>,
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
        let col = match parse_temporal(col, dtype, truncating.contains(name))? {
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
            let user_typed = user_schema || overrides.is_some_and(|o| o.contains(name));
            // the conversion's numbers are ticks of one unit: any other unit is a guess
            let own_ticks = own_numbers
                .get(name)
                .is_some_and(|ticks| temporal_leaves_are(dtype, ticks));
            // (a column of NULLs only has no values to misread: a driver types it as it likes)
            let has_values = col.null_count() < col.len();
            if has_values && user_typed && !own_ticks && numeric_to_temporal(col.dtype(), dtype) {
                polars_bail!(
                    SchemaMismatch:
                    "database column '{name}' is {} and {dtype} was requested: a number isn't \
                    converted to a date or time as raw ticks (the unit and epoch of the number \
                    aren't known); read the column as {} and convert it in Polars, for example \
                    with `pl.from_epoch`",
                    col.dtype(), col.dtype()
                )
            }
            // A column the database retyped since planning (an `INT` that is now a `DATE`) would be
            // read as another kind of value: day numbers, raw ticks, midnight. A time of day for a
            // column declared as text (`schema=`, where nothing looked the type up) on a driver
            // that can't tell a `TIMETZ` (the planner lists such columns): it comes back as the
            // local time of day, without the offset the text would have had.
            if has_values
                && matches!((col.dtype(), dtype), (DataType::Time, DataType::String))
                && let Some(cast) = time_text_risk.get(name)
            {
                polars_bail!(
                    SchemaMismatch:
                    "database column '{name}' is read as a time of day but String was \
                    declared: this driver can't tell a time with a time zone (`TIMETZ`), \
                    which it returns without its offset, and the text would lose it. Read \
                    the column as text in a query ({cast}), or with a Python driver"
                )
            }
            if has_values
                && !user_typed
                && (kind_changed(col.dtype(), dtype)
                    || value_kind_changed(col.dtype(), dtype, plugin_converted.contains(name)))
            {
                polars_bail!(
                    SchemaMismatch:
                    "database column '{name}' is {} but {dtype} was planned: the column's type \
                    changed since the scan was planned; plan the scan again",
                    col.dtype()
                )
            }
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
                check_numeric_exact(col, &cast)?;
            }
            cast
        };
        columns.push(col);
    }
    DataFrame::new(height, columns)
}

/// Whether `f` holds for a pair of dtypes of `from` and `to` that aren't nested, matched through
/// lists, arrays, maps and structs (fields by name, as the cast matches them).
fn any_leaf_pair(
    from: &DataType,
    to: &DataType,
    f: &impl Fn(&DataType, &DataType) -> bool,
) -> bool {
    match (from, to) {
        (DataType::List(a) | DataType::Array(a, _), DataType::List(b) | DataType::Array(b, _)) => {
            any_leaf_pair(a, b, f)
        },
        (DataType::Map(ka, va), DataType::Map(kb, vb)) => {
            any_leaf_pair(ka, kb, f) || any_leaf_pair(va, vb, f)
        },
        // a map cast to its storage, a list of `key` / `value` structs, and back
        (DataType::Map(k, v), DataType::List(entries)) => match entries.as_ref() {
            DataType::Struct(fields) => fields.iter().any(|fb| match fb.name().as_str() {
                "key" => any_leaf_pair(k, fb.dtype(), f),
                "value" => any_leaf_pair(v, fb.dtype(), f),
                _ => false,
            }),
            _ => false,
        },
        (DataType::List(entries), DataType::Map(k, v)) => match entries.as_ref() {
            DataType::Struct(fields) => fields.iter().any(|fa| match fa.name().as_str() {
                "key" => any_leaf_pair(fa.dtype(), k, f),
                "value" => any_leaf_pair(fa.dtype(), v, f),
                _ => false,
            }),
            _ => false,
        },
        (DataType::Struct(a), DataType::Struct(b)) => b.iter().any(|fb| {
            a.iter()
                .find(|fa| fa.name() == fb.name())
                .is_some_and(|fa| any_leaf_pair(fa.dtype(), fb.dtype(), f))
        }),
        // a value cast to a struct of one field is cast to the field
        (_, DataType::Struct(b)) if b.len() == 1 => any_leaf_pair(from, b[0].dtype(), f),
        _ => f(from, to),
    }
}

/// Whether every date or time dtype in `dtype` (also nested) is `ticks`, time zones aside.
fn temporal_leaves_are(dtype: &DataType, ticks: &DataType) -> bool {
    match dtype {
        DataType::List(inner) | DataType::Array(inner, _) => temporal_leaves_are(inner, ticks),
        DataType::Map(k, v) => temporal_leaves_are(k, ticks) && temporal_leaves_are(v, ticks),
        DataType::Struct(fields) => fields.iter().all(|f| temporal_leaves_are(f.dtype(), ticks)),
        DataType::Datetime(unit, _) => matches!(ticks, DataType::Datetime(u, _) if u == unit),
        DataType::Date | DataType::Time | DataType::Duration(_) => dtype == ticks,
        _ => true,
    }
}

fn is_number(dtype: &DataType) -> bool {
    dtype.is_integer() || dtype.is_float() || dtype.is_decimal()
}

fn is_temporal(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Date | DataType::Datetime(_, _) | DataType::Time | DataType::Duration(_)
    )
}

/// A cast of a number to a date, time or duration, also nested. A duration of a database's `TIME`
/// is planned from microseconds by [`duration_to_time`], before this.
fn numeric_to_temporal(from: &DataType, to: &DataType) -> bool {
    any_leaf_pair(from, to, &|from, to| is_number(from) && is_temporal(to))
}

/// A change of kind between the column read and the one planned: a date or time read where a number
/// was planned, or a date and a datetime (the cast reads day numbers, raw ticks or midnight).
/// Nested too.
fn kind_changed(from: &DataType, to: &DataType) -> bool {
    any_leaf_pair(from, to, &|from, to| {
        // (a number planned as a date or time is a database's own conversion)
        (is_temporal(from) && is_number(to))
            || matches!(
                (from, to),
                (DataType::Date, DataType::Datetime(_, _))
                    | (DataType::Datetime(_, _), DataType::Date)
            )
    })
}

/// A column read as another kind of value than planned, unless the plugin reads it through a
/// conversion of its own (`converted`: it may make text of a number, or a naive UTC time of an
/// instant): text where an integer or float was planned (`'42'` is no 42), a boolean where a number
/// was (read as 1), and a naive timestamp where one with a time zone was (its wall clock would be
/// read as UTC). Nested too.
fn value_kind_changed(from: &DataType, to: &DataType, converted: bool) -> bool {
    !converted
        && any_leaf_pair(from, to, &|from, to| {
            let plain_number = |d: &DataType| d.is_integer() || d.is_float();
            (matches!(from, DataType::String) && plain_number(to))
                || (matches!(from, DataType::Boolean) && is_number(to))
                || matches!(
                    (from, to),
                    (DataType::Datetime(_, None), DataType::Datetime(_, Some(_)))
                )
        })
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
fn check_decimal_exact(src: &Column, cast: &Column, root: &PlSmallStr) -> PolarsResult<()> {
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
            as text with schema_overrides={{'{root}': pl.String}} (or as pl.Float64)",
            src.name(), cast.dtype()
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
    check_exact_at(src, cast, src.name())
}

/// `check_numeric_exact` at any depth: the values of lists and arrays, and the fields of
/// structs (matched by name, as the cast does), are checked as a column of their own; the
/// decimal check too, at the leaves. `root` is the database column, which a message names
/// with the path to the value.
fn check_exact_at(src: &Column, cast: &Column, root: &PlSmallStr) -> PolarsResult<()> {
    let path = |suffix: &str| -> PlSmallStr { format!("{}{suffix}", src.name()).into() };
    match (src.dtype(), cast.dtype()) {
        (DataType::Map(_, _), DataType::Map(_, _) | DataType::List(_)) => {
            // a map is a list of `key` / `value` structs: its values are checked as such
            let storage = |c: &Column| -> PolarsResult<Column> {
                if !matches!(c.dtype(), DataType::Map(_, _)) {
                    return Ok(c.clone());
                }
                Ok(c.as_materialized_series()
                    .map()?
                    .storage()
                    .clone()
                    .with_name(c.name().clone())
                    .into_column())
            };
            return check_exact_at(&storage(src)?, &storage(cast)?, root);
        },
        (DataType::List(_) | DataType::Array(_, _), DataType::List(_) | DataType::Array(_, _)) => {
            // The values of the rows that have a list (exploded: a null row holds none in a
            // list, a width's worth in an array, and a slice keeps the values outside it).
            let values = |c: &Column| -> PolarsResult<Column> {
                let options = ExplodeOptions {
                    empty_as_null: false,
                    keep_nulls: false,
                };
                let inner = c.as_materialized_series().explode(options)?;
                Ok(inner.with_name(path("[]")).into_column())
            };
            return check_exact_at(&values(src)?, &values(cast)?, root);
        },
        (DataType::List(_), DataType::Map(key, value)) => {
            // checked as the list of entries, before a repeated key collapses them
            let entries = DataType::List(Box::new(DataType::Struct(vec![
                Field::new("key".into(), key.as_ref().clone()),
                Field::new("value".into(), value.as_ref().clone()),
            ])));
            return match src.cast(&entries) {
                Ok(entries) => check_exact_at(src, &entries, root),
                Err(_) => Ok(()),
            };
        },
        (from, DataType::Struct(fields)) if fields.len() == 1 && !from.is_struct() => {
            // a value cast to a struct of one field is cast to the field
            let field = cast.struct_()?.fields_as_series().swap_remove(0);
            let name = path(&format!(".{}", field.name()));
            let src = src.as_materialized_series().clone().with_name(name.clone());
            return check_exact_at(
                &src.into_column(),
                &field.with_name(name).into_column(),
                root,
            );
        },
        (DataType::Struct(_), DataType::Struct(_)) => {
            let (src_s, cast_s) = (src.struct_()?, cast.struct_()?);
            let src_fields = src_s.fields_as_series();
            for b in cast_s.fields_as_series() {
                let name = path(&format!(".{}", b.name()));
                let Some(a) = src_fields.iter().find(|a| a.name() == b.name()) else {
                    if src.null_count() < src.len() {
                        polars_bail!(
                            SchemaMismatch:
                            "database column '{root}' has no field '{}', which {} plans; the \
                            cast would fill it with nulls",
                            b.name(), cast.dtype()
                        )
                    }
                    continue;
                };
                check_exact_at(
                    &a.clone().with_name(name.clone()).into_column(),
                    &b.with_name(name).into_column(),
                    root,
                )?;
            }
            return Ok(());
        },
        _ => {},
    }
    check_decimal_exact(src, cast, root)?;
    let (from, to) = (src.dtype(), cast.dtype());
    if from.is_float() && matches!(to, DataType::Float32 | DataType::Float16) {
        // the nearest `Float32` is the reading, but not an overflow to infinity
        let finite = src.as_materialized_series().is_finite()?;
        let overflow = cast.as_materialized_series().is_infinite()? & finite;
        if let Some(i) = overflow.iter().position(|v| v == Some(true)) {
            polars_bail!(
                SchemaMismatch:
                "database column '{}' has the value {}, which is out of the range of {}; \
                read the column as pl.Float64",
                src.name(), src.get(i)?,
                if to == &DataType::Float16 { "Float16" } else { "Float32" }
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
        schema_overrides={{'{root}': {alt}}} (or as pl.String)",
        src.name(), src.get(i)?
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
/// `truncate`: fractional digits below the planned unit are cut toward the earlier instant
/// (never rounded: `.9999999` of `9999-12-31` would overflow), where the user asked for the
/// unit; otherwise a value the unit can't hold is an error.
fn parse_temporal(col: &Column, dtype: &DataType, truncate: bool) -> PolarsResult<Option<Column>> {
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
            // The parse overflows near the ends of the nanosecond range (1677 to 2262)
            // and drops fractional digits below its unit: the microseconds are parsed, and
            // the 7th to 9th fractional digits added. The value is then converted to the
            // planned unit, and a value it can't hold exactly is an error.
            let us = ca.as_datetime(None, TimeUnit::Microseconds, true, false, None, &raise)?;
            let unit = time_unit_ns(*tu) as i128;
            let naive = if *tu == TimeUnit::Microseconds
                && ca
                    .iter()
                    .all(|t| t.is_none_or(|t| fraction_beyond_micro(t) == (0, false)))
            {
                // the common case: the parse is the value
                us.into_series().with_name(col.name().clone())
            } else {
                let mut values = Vec::with_capacity(ca.len());
                for (i, (v, text)) in us.physical().iter().zip(ca.iter()).enumerate() {
                    let Some(v) = v else {
                        values.push(None);
                        continue;
                    };
                    let (sub_ns, beyond) = text.map_or((0, false), fraction_beyond_micro);
                    if beyond {
                        polars_bail!(
                            SchemaMismatch:
                            "database column '{}' has the value {}, with fractional digits \
                            beyond the nanosecond, which {dtype} would cut; read the column \
                            as pl.String",
                            col.name(), ca.get(i).unwrap_or_default()
                        );
                    }
                    // (in i128: the last microsecond before the minimum starts below it)
                    let ns_value = v as i128 * 1000 + sub_ns as i128;
                    if !truncate && ns_value % unit != 0 {
                        let finer = if ns_value % 1000 == 0 { "us" } else { "ns" };
                        polars_bail!(
                            SchemaMismatch:
                            "database column '{}' has the value {}, which {dtype} would cut; \
                            read the column as pl.Datetime('{finer}')",
                            col.name(), ca.get(i).unwrap_or_default()
                        );
                    }
                    let Ok(v) = i64::try_from(ns_value.div_euclid(unit)) else {
                        polars_bail!(
                            SchemaMismatch:
                            "database column '{}' has the value {}, outside the range of \
                            {dtype} (1677 to 2262); read the column with \
                            schema_overrides={{'{}': pl.Datetime('us')}}",
                            col.name(), ca.get(i).unwrap_or_default(), col.name()
                        );
                    };
                    values.push(Some(v));
                }
                Int64Chunked::from_iter_options(col.name().clone(), values.into_iter())
                    .into_datetime(*tu, None)
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

/// The 7th to 9th fractional digits of a timestamp's text, as nanoseconds beyond its
/// microseconds, and whether any digit after the 9th is not zero (the parse drops them).
fn fraction_beyond_micro(text: &str) -> (i64, bool) {
    let bytes = text.as_bytes();
    let Some(dot) = bytes.iter().rposition(|&b| b == b'.') else {
        return (0, false);
    };
    let digits = &bytes[dot + 1..];
    let digits = &digits[..digits.iter().take_while(|b| b.is_ascii_digit()).count()];
    if digits.len() <= 6 {
        return (0, false);
    }
    let ns = (0..3).fold(0i64, |acc, i| {
        acc * 10 + digits.get(6 + i).map_or(0, |d| (d - b'0') as i64)
    });
    (ns, digits.iter().skip(9).any(|&d| d != b'0'))
}

/// A ClickHouse type without its `Nullable(…)` and `LowCardinality(…)` wrappers, lower-cased.
pub fn clickhouse_value_type(db_type: &str) -> String {
    clickhouse_unwrapped(db_type).to_ascii_lowercase()
}

/// A ClickHouse type without its wrappers, as written: time zone names are case-sensitive.
pub(super) fn clickhouse_unwrapped(db_type: &str) -> &str {
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
    fn numbers_are_not_cast_to_dates_of_user_columns() {
        let ns_ticks = DataType::Datetime(TimeUnit::Nanoseconds, None);
        let ints = Column::new("x".into(), &[1_700_000_000i64]);
        let df = DataFrame::new(1, vec![ints]).unwrap();
        let none = PlHashMap::new();
        for dtype in [
            DataType::Date,
            DataType::Time,
            DataType::Datetime(TimeUnit::Microseconds, None),
        ] {
            let schema = Schema::from_iter([Field::new("x".into(), dtype.clone())]);
            // `schema=`, and a column of `schema_overrides`
            let e = cast_to_user_schema(&df, &schema, None, true, &none).unwrap_err();
            assert!(e.to_string().contains("read the column as i64"), "{e}");
            assert!(cast_to_user_schema(&df, &schema, Some(&schema), false, &none).is_err());
            // the database's own conversion plans it
            assert!(cast_to_user_schema(&df, &schema, None, false, &none).is_ok());
            // an override of a column the plugin itself read as a number (ClickHouse
            // `DateTime64(9)` as nanoseconds) is no raw-ticks guess
            let converted = PlHashMap::from_iter([(PlSmallStr::from("x"), ns_ticks.clone())]);
            assert!(cast_to_user_schema(&df, &schema, Some(&schema), false, &converted).is_err());
            // `schema=` names no database type: nothing converted
            assert!(cast_to_user_schema(&df, &schema, None, true, &none).is_err());
        }
        // (the conversion's numbers are nanoseconds: only that unit is no guess)
        let ns = Schema::from_iter([Field::new(
            "x".into(),
            DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC)),
        )]);
        let converted = PlHashMap::from_iter([(PlSmallStr::from("x"), ns_ticks.clone())]);
        assert!(cast_to_user_schema(&df, &ns, Some(&ns), false, &converted).is_ok());
        assert!(cast_to_user_schema(&df, &ns, Some(&ns), false, &none).is_err());
        let dur = Schema::from_iter([Field::new(
            "x".into(),
            DataType::Duration(TimeUnit::Microseconds),
        )]);
        // a duration is no more a number of ticks than a date is, and nesting doesn't hide it
        assert!(cast_to_user_schema(&df, &dur, Some(&dur), true, &none).is_err());
        // unless the plugin's conversion reads it in that unit (a MySQL `TIME`)
        let us_ticks = DataType::Duration(TimeUnit::Microseconds);
        let own_us = PlHashMap::from_iter([(PlSmallStr::from("x"), us_ticks)]);
        assert!(cast_to_user_schema(&df, &dur, Some(&dur), false, &own_us).is_ok());
        let ms = Schema::from_iter([Field::new(
            "x".into(),
            DataType::Duration(TimeUnit::Milliseconds),
        )]);
        assert!(cast_to_user_schema(&df, &ms, Some(&ms), false, &own_us).is_err());
        let nested = |inner: DataType| {
            Schema::from_iter([Field::new("x".into(), DataType::List(Box::new(inner)))])
        };
        let list = DataFrame::new(
            1,
            vec![
                Column::new("x".into(), &[1_700_000_000i64])
                    .as_materialized_series()
                    .implode()
                    .unwrap()
                    .into_column(),
            ],
        )
        .unwrap();
        for inner in [
            DataType::Date,
            DataType::Datetime(TimeUnit::Microseconds, None),
        ] {
            let schema = nested(inner);
            assert!(cast_to_user_schema(&list, &schema, None, true, &none).is_err());
            assert!(cast_to_user_schema(&list, &schema, Some(&schema), false, &none).is_err());
        }
        // a ClickHouse array of nanoseconds, retyped in its own unit
        let schema = nested(DataType::Datetime(
            TimeUnit::Nanoseconds,
            Some(TimeZone::UTC),
        ));
        assert!(cast_to_user_schema(&list, &schema, Some(&schema), false, &converted).is_ok());
        let schema = nested(DataType::Datetime(
            TimeUnit::Microseconds,
            Some(TimeZone::UTC),
        ));
        assert!(cast_to_user_schema(&list, &schema, Some(&schema), false, &converted).is_err());
        // a map's values
        let map = |value: DataType| DataType::Map(Box::new(DataType::String), Box::new(value));
        assert!(numeric_to_temporal(
            &map(DataType::Int64),
            &map(DataType::Datetime(TimeUnit::Microseconds, None))
        ));
        assert!(!numeric_to_temporal(
            &map(DataType::Int64),
            &map(DataType::Float64)
        ));
        // and of its storage
        let entries = DataType::List(Box::new(DataType::Struct(vec![
            Field::new("key".into(), DataType::String),
            Field::new(
                "value".into(),
                DataType::Datetime(TimeUnit::Microseconds, None),
            ),
        ])));
        assert!(numeric_to_temporal(&map(DataType::Int64), &entries));
        let ints = DataType::List(Box::new(DataType::Struct(vec![
            Field::new("key".into(), DataType::String),
            Field::new("value".into(), DataType::Int64),
        ])));
        let ticks = map(DataType::Datetime(TimeUnit::Microseconds, None));
        assert!(numeric_to_temporal(&ints, &ticks));
        // a struct of one field
        let one = DataType::Struct(vec![Field::new("t".into(), DataType::Date)]);
        assert!(numeric_to_temporal(&DataType::Int64, &one));
    }

    #[test]
    fn a_column_retyped_since_planning_is_not_reread_as_another_kind() {
        let none = PlHashMap::new();
        let day = Column::new("x".into(), &[1i32])
            .cast(&DataType::Date)
            .unwrap();
        let date = DataFrame::new(1, vec![day]).unwrap();
        // a date where an integer was planned: day numbers
        let ints = Schema::from_iter([Field::new("x".into(), DataType::Int32)]);
        let e = cast_to_user_schema(&date, &ints, None, false, &none).unwrap_err();
        assert!(
            e.to_string().contains("changed since the scan was planned"),
            "{e}"
        );
        // a date where a datetime was planned: midnight
        let dt = Schema::from_iter([Field::new(
            "x".into(),
            DataType::Datetime(TimeUnit::Microseconds, None),
        )]);
        assert!(cast_to_user_schema(&date, &dt, None, false, &none).is_err());
        // what the user typed is cast as asked
        assert!(cast_to_user_schema(&date, &dt, Some(&dt), false, &none).is_ok());
    }

    #[test]
    fn utc_spellings_normalise_inside_nested_types() {
        let est = DataType::Datetime(
            TimeUnit::Microseconds,
            TimeZone::opt_try_new(Some("Etc/UTC")).unwrap(),
        );
        let utc = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
        let nested = DataType::Struct(vec![Field::new("b".into(), est.clone())]);
        assert_eq!(
            normalize_dtype(&nested),
            DataType::Struct(vec![Field::new("b".into(), utc.clone())])
        );
        assert_eq!(
            normalize_dtype(&DataType::Array(Box::new(est), 2)),
            DataType::Array(Box::new(utc), 2)
        );
    }

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
    fn nested_inexact_casts_fail() {
        let cast = |col: Column, dtype: DataType| {
            let df = DataFrame::new(col.len(), vec![col]).unwrap();
            cast_to_schema(
                &df,
                &Schema::from_iter([Field::new("x".into(), dtype)]),
                None,
            )
        };
        let list_of = |v: Series| Series::new("x".into(), &[v]).into_column();
        let floats = Series::new("".into(), &[1.0, 1.5]);
        let e = cast(
            list_of(floats.clone()),
            DataType::List(Box::new(DataType::Int64)),
        )
        .unwrap_err();
        assert!(
            e.to_string().contains("1.5") && e.to_string().contains("x[]"),
            "{e}"
        );
        let e = cast(
            list_of(Series::new("".into(), &[1e300])),
            DataType::List(Box::new(DataType::Float32)),
        )
        .unwrap_err();
        assert!(e.to_string().contains("Float32"), "{e}");
        let e = cast(
            list_of(Series::new("".into(), &[1e5])),
            DataType::List(Box::new(DataType::Float16)),
        )
        .unwrap_err();
        assert!(e.to_string().contains("Float16"), "{e}");
        let ints = Series::new("".into(), &[(1i64 << 53) + 1]);
        assert!(cast(list_of(ints), DataType::List(Box::new(DataType::Float64))).is_err());
        let ok = Series::new("".into(), &[1.0, 2.0]);
        assert!(cast(list_of(ok), DataType::List(Box::new(DataType::Int64))).is_ok());
        let us = Series::new("".into(), &[789_123i64])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let ms = DataType::List(Box::new(DataType::Datetime(TimeUnit::Milliseconds, None)));
        assert!(cast(list_of(us), ms).is_err());
        // structs, by field, and nested in lists
        let field = Series::new("a".into(), &[1.5]);
        let st = StructChunked::from_series("x".into(), 1, [field].iter())
            .unwrap()
            .into_column();
        let target = DataType::Struct(vec![Field::new("a".into(), DataType::Int64)]);
        let e = cast(st.clone(), target.clone()).unwrap_err();
        assert!(e.to_string().contains("x.a"), "{e}");
        let in_list = Series::new("x".into(), &[st.as_materialized_series().clone()]);
        assert!(cast(in_list.into_column(), DataType::List(Box::new(target))).is_err());
    }

    #[test]
    fn nested_casts_match_struct_fields_by_name_and_check_decimals_and_arrays() {
        let cast = |col: Column, dtype: DataType| {
            let df = DataFrame::new(col.len(), vec![col]).unwrap();
            cast_to_schema(
                &df,
                &Schema::from_iter([Field::new("x".into(), dtype)]),
                None,
            )
        };
        let st = |fields: &[Series]| {
            StructChunked::from_series("x".into(), fields[0].len(), fields.iter())
                .unwrap()
                .into_column()
        };
        let (a, b) = (
            Series::new("a".into(), &[1.5]),
            Series::new("b".into(), &[1.0]),
        );
        // by name, not position: `a` (1.5) is the one that would be cut
        let reordered = DataType::Struct(vec![
            Field::new("b".into(), DataType::Float64),
            Field::new("a".into(), DataType::Int64),
        ]);
        // (the strict cast refuses these itself; the check pairs by name regardless)
        assert!(cast(st(&[a.clone(), b.clone()]), reordered).is_err());
        // a planned field the column lacks
        let missing = DataType::Struct(vec![Field::new("c".into(), DataType::Int64)]);
        assert!(cast(st(&[a.clone(), b.clone()]), missing).is_err());
        // decimals in a struct and in a list of structs
        let dec = |v: &str, scale: usize| {
            Series::new("d".into(), &[v])
                .cast(&DataType::Decimal(20, scale))
                .unwrap()
        };
        let target = DataType::Struct(vec![Field::new("d".into(), DataType::Decimal(20, 1))]);
        let e = cast(st(&[dec("1.25", 2)]), target.clone()).unwrap_err();
        assert!(e.to_string().contains("round"), "{e}");
        assert!(cast(st(&[dec("1.20", 2)]), target.clone()).is_ok());
        let in_list = Series::new(
            "x".into(),
            &[st(&[dec("1.25", 2)]).take_materialized_series()],
        );
        assert!(cast(in_list.into_column(), DataType::List(Box::new(target))).is_err());
        // an array and a list, either way, also with a null row
        let list = |rows: &[Option<[f64; 2]>]| {
            let rows: Vec<Option<Series>> = rows
                .iter()
                .map(|r| r.map(|r| Series::new("".into(), &r)))
                .collect();
            Series::new("x".into(), rows).into_column()
        };
        let list_of_ints = DataType::List(Box::new(DataType::Int64));
        let array_of_ints = DataType::Array(Box::new(DataType::Int64), 2);
        let bad = list(&[None, Some([1.0, 1.5])]);
        assert!(cast(bad.clone(), list_of_ints.clone()).is_err());
        assert!(cast(bad.clone(), array_of_ints.clone()).is_err());
        let fine = list(&[None, Some([1.0, 2.0])]);
        assert!(cast(fine.clone(), array_of_ints.clone()).is_ok());
        let array = fine
            .cast(&DataType::Array(Box::new(DataType::Float64), 2))
            .unwrap();
        cast(array.clone(), list_of_ints).unwrap();
        let bad_array = bad
            .cast(&DataType::Array(Box::new(DataType::Float64), 2))
            .unwrap();
        assert!(cast(bad_array.clone(), DataType::List(Box::new(DataType::Int64))).is_err());
        assert!(cast(bad_array, array_of_ints).is_err());
    }

    #[test]
    fn text_timestamps_finer_than_the_unit_fail() {
        let cast = |v: &str, tu: TimeUnit| {
            let df = DataFrame::new(1, vec![Column::new("t".into(), &[v])]).unwrap();
            cast_to_schema(
                &df,
                &Schema::from_iter([Field::new("t".into(), DataType::Datetime(tu, None))]),
                None,
            )
        };
        let ms = TimeUnit::Milliseconds;
        let us = TimeUnit::Microseconds;
        assert!(cast("2026-01-01 10:00:00.789", ms).is_ok());
        assert!(cast("2026-01-01 10:00:00.789000", ms).is_ok());
        assert!(cast("2026-01-01 10:00:00.789000000", ms).is_ok());
        assert!(cast("2026-01-01 10:00:00", ms).is_ok());
        let e = cast("2026-01-01 10:00:00.789123", ms).unwrap_err();
        assert!(e.to_string().contains("would cut"), "{e}");
        assert!(cast("2026-01-01 10:00:00.123456", us).is_ok());
        assert!(cast("2026-01-01 10:00:00.123456789", us).is_err());
        assert!(cast("1969-12-31 23:59:59.9995", ms).is_err());
        // (the parser itself refuses digits beyond the ninth)
        let ns = TimeUnit::Nanoseconds;
        assert!(cast("2026-01-01 10:00:00.1234567891", ns).is_err());
        assert!(cast("2026-01-01 10:00:00.0000000005", ns).is_err());
        let df = cast("1969-12-31 23:59:59.999", ms).unwrap();
        let v = df[0]
            .as_materialized_series()
            .to_physical_repr()
            .i64()
            .unwrap()
            .get(0);
        assert_eq!(v, Some(-1));
    }

    #[test]
    fn mssql_and_trino_declared_types() {
        let ms = DialectName::Mssql;
        let ns = DataType::Datetime(TimeUnit::Nanoseconds, None);
        let us = DataType::Datetime(TimeUnit::Microseconds, None);
        assert_eq!(declared_dtype(ms, "datetime2"), Some(ns.clone()));
        assert_eq!(declared_dtype(ms, "datetime2(7)"), Some(ns));
        assert_eq!(declared_dtype(ms, "datetime2(6)"), Some(us.clone()));
        assert_eq!(declared_dtype(ms, "datetime"), Some(us));
        assert_eq!(
            declared_dtype(ms, "datetimeoffset"),
            Some(DataType::Datetime(
                TimeUnit::Nanoseconds,
                Some(TimeZone::UTC)
            ))
        );
        assert_eq!(declared_dtype(ms, "time(7)"), Some(DataType::Time));
        assert_eq!(declared_dtype(ms, "money"), Some(DataType::Decimal(19, 4)));
        assert_eq!(declared_dtype(ms, "real"), Some(DataType::Float64));
        assert_eq!(
            declared_dtype(DialectName::Trino, "timestamp(3) with time zone"),
            Some(DataType::Datetime(
                TimeUnit::Microseconds,
                Some(TimeZone::UTC)
            ))
        );
    }

    #[test]
    fn truncated_text_timestamps_are_cut_toward_the_earlier_instant() {
        let us = DataType::Datetime(TimeUnit::Microseconds, None);
        let col = Column::new("t".into(), &["9999-12-31 23:59:59.9999999"]);
        assert!(parse_temporal(&col, &us, false).is_err());
        let cut = parse_temporal(&col, &us, true).unwrap().unwrap();
        let expected = Column::new("t".into(), &["9999-12-31 23:59:59.999999"]);
        let expected = parse_temporal(&expected, &us, false).unwrap().unwrap();
        assert_eq!(cut.get(0).unwrap(), expected.get(0).unwrap());
        // before 1970: toward the earlier instant, not toward zero
        let col = Column::new("t".into(), &["1969-12-31 23:59:59.9999995"]);
        let cut = parse_temporal(&col, &us, true).unwrap().unwrap();
        let v = cut
            .as_materialized_series()
            .to_physical_repr()
            .i64()
            .unwrap()
            .get(0);
        assert_eq!(v, Some(-1));
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
