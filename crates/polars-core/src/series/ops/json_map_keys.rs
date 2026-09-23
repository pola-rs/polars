//! JSON support for `Map`: JSON objects have string keys, so every key dtype has an explicit
//! string codec. Readers decode a Map as `List(Struct {key: String, value})` in source order
//! and rebuild it with [`Series::from_json_decoded`].
#[cfg(feature = "dtype-map")]
use std::fmt::Write;

#[cfg(feature = "dtype-map")]
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};

use crate::chunked_array::cast::CastOptions;
#[cfg(feature = "dtype-map")]
use crate::chunked_array::logical::try_apply_map_entries;
use crate::prelude::*;

#[cfg(all(feature = "dtype-map", feature = "dtype-date"))]
const EPOCH_DAYS_FROM_CE: i32 = 719_163;
#[cfg(all(feature = "dtype-map", feature = "dtype-datetime"))]
const NAIVE_DATETIME_FMT: &str = "%Y-%m-%dT%H:%M:%S%.f";
#[cfg(all(feature = "dtype-map", feature = "dtype-time"))]
const TIME_FMT: &str = "%H:%M:%S%.f";

impl DataType {
    /// Reject nested `Map`s whose key dtype has no JSON object-key codec.
    pub fn ensure_json_map_keys(&self) -> PolarsResult<()> {
        use DataType as D;
        match self {
            #[cfg(feature = "dtype-map")]
            D::Map(key, value) => {
                polars_ensure!(
                    is_json_map_key(key),
                    ComputeError:
                    "JSON does not support Map keys of type `{key}`\n\nConsider `Expr.map.entries` to use the entries as a list of structs instead."
                );
                value.ensure_json_map_keys()
            },
            D::List(inner) => inner.ensure_json_map_keys(),
            #[cfg(feature = "dtype-array")]
            D::Array(inner, _) => inner.ensure_json_map_keys(),
            #[cfg(feature = "dtype-struct")]
            D::Struct(fields) => fields
                .iter()
                .try_for_each(|field| field.dtype.ensure_json_map_keys()),
            #[cfg(feature = "dtype-extension")]
            D::Extension(_, storage) => storage.ensure_json_map_keys(),
            _ => Ok(()),
        }
    }

    /// The dtype a JSON reader decodes into before [`Series::from_json_decoded`].
    ///
    /// Maps become `List(Struct {key: String, value})`, Arrays containing a Map become Lists
    /// and Enum/Categorical leaves become String.
    pub fn json_map_decode_dtype(&self) -> DataType {
        use DataType as D;
        match self {
            #[cfg(feature = "dtype-map")]
            D::Map(_, value) => {
                D::Map(Box::new(D::String), Box::new(value.json_map_decode_dtype()))
                    .map_storage_dtype()
                    .unwrap()
            },
            D::List(inner) => D::List(Box::new(inner.json_map_decode_dtype())),
            #[cfg(feature = "dtype-array")]
            D::Array(inner, _) if inner.contains_map() => {
                D::List(Box::new(inner.json_map_decode_dtype()))
            },
            #[cfg(feature = "dtype-array")]
            D::Array(inner, width) => D::Array(Box::new(inner.json_map_decode_dtype()), *width),
            #[cfg(feature = "dtype-struct")]
            D::Struct(fields) => D::Struct(
                fields
                    .iter()
                    .map(|f| Field::new(f.name.clone(), f.dtype.json_map_decode_dtype()))
                    .collect(),
            ),
            #[cfg(feature = "dtype-categorical")]
            D::Enum(..) | D::Categorical(..) => D::String,
            dt => dt.clone(),
        }
    }
}

#[cfg(feature = "dtype-map")]
fn is_json_map_key(dtype: &DataType) -> bool {
    use DataType as D;
    match dtype {
        D::String | D::Boolean => true,
        #[cfg(feature = "dtype-categorical")]
        D::Categorical(..) | D::Enum(..) => true,
        #[cfg(feature = "dtype-decimal")]
        D::Decimal(..) => true,
        #[cfg(feature = "dtype-date")]
        D::Date => true,
        #[cfg(feature = "dtype-datetime")]
        D::Datetime(..) => true,
        #[cfg(feature = "dtype-time")]
        D::Time => true,
        #[cfg(feature = "dtype-duration")]
        D::Duration(..) => true,
        dt => dt.is_integer() || dt.is_float(),
    }
}

impl Series {
    /// Encode the keys of all nested `Map`s as JSON object keys. Returns `None` if unchanged.
    pub fn map_keys_to_json(&self) -> PolarsResult<Option<Series>> {
        self.dtype().ensure_json_map_keys()?;
        #[cfg(feature = "dtype-map")]
        {
            map_keys_to_json_rec(self)
        }
        #[cfg(not(feature = "dtype-map"))]
        {
            Ok(None)
        }
    }

    /// Build `target` from a series decoded as [`DataType::json_map_decode_dtype`].
    ///
    /// Map keys are decoded and deduplicated with first-position/last-value semantics. With
    /// `ignore_errors`, rows with an undecodable key or a wrong Array width become null.
    pub fn from_json_decoded(self, target: &DataType, ignore_errors: bool) -> PolarsResult<Series> {
        #[cfg(feature = "dtype-map")]
        if target.contains_map() {
            target.ensure_json_map_keys()?;
            let decoded = cast_leaf(self, &target.json_map_decode_dtype(), ignore_errors)?;
            return from_json_decoded_rec(&decoded, target, ignore_errors);
        }
        cast_leaf(self, target, ignore_errors)
    }
}

fn cast_leaf(s: Series, target: &DataType, ignore_errors: bool) -> PolarsResult<Series> {
    if s.dtype() == target {
        return Ok(s);
    }
    let options = if ignore_errors {
        CastOptions::NonStrict
    } else {
        CastOptions::Strict
    };
    s.cast_with_options(target, options)
}

#[cfg(feature = "dtype-map")]
fn map_keys_to_json_rec(series: &Series) -> PolarsResult<Option<Series>> {
    if !series.dtype().contains_map() {
        return Ok(None);
    }

    match series.dtype() {
        DataType::Map(key, value) => {
            if key.is_string() && !value.contains_map() {
                return Ok(None);
            }

            let map = series.map().unwrap();
            let mut value_dtype = None;
            let storage = try_apply_map_entries(&map.live_storage(), |key, value| {
                let value = map_keys_to_json_rec(value)?.unwrap_or_else(|| value.clone());
                value_dtype = Some(value.dtype().clone());
                Ok((encode_keys(key)?, value))
            })?
            .into_series();

            let dtype = DataType::Map(Box::new(DataType::String), Box::new(value_dtype.unwrap()));
            // SAFETY: the key codec is injective and preserves nulls, so live keys stay
            // non-null and unique; values keep their validity.
            Ok(Some(
                unsafe { MapChunked::from_storage_unchecked(dtype, storage) }.into_series(),
            ))
        },
        DataType::List(_) => {
            let ca = series.list().unwrap();
            Ok(map_keys_to_json_rec(&ca.get_inner())?
                .map(|values| ca.with_inner_values(&values).into_series()))
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => {
            let ca = series.array().unwrap();
            Ok(map_keys_to_json_rec(&ca.get_inner())?
                .map(|values| ca.with_inner_values(&values).into_series()))
        },
        DataType::Struct(_) => {
            let ca = series.struct_().unwrap();
            let out = ca.try_apply_fields(|field| {
                Ok(map_keys_to_json_rec(field)?.unwrap_or_else(|| field.clone()))
            })?;
            Ok(Some(out.into_series()))
        },
        #[cfg(feature = "dtype-extension")]
        DataType::Extension(typ, _) => Ok(map_keys_to_json_rec(series.ext().unwrap().storage())?
            .map(|s| s.into_extension(typ.clone()))),
        _ => Ok(None),
    }
}

#[cfg(feature = "dtype-map")]
fn from_json_decoded_rec(
    series: &Series,
    target: &DataType,
    ignore_errors: bool,
) -> PolarsResult<Series> {
    if !target.contains_map() {
        return cast_leaf(series.clone(), target, ignore_errors);
    }

    match target {
        DataType::Map(key_dtype, value_dtype) => {
            let mut ok_keys: Option<Bitmap> = None;
            let storage = try_apply_map_entries(series.list()?, |key, value| {
                let decoded = decode_keys(key.str()?, key_dtype)?;
                if decoded.null_count() > key.null_count() {
                    let ok = (decoded.is_not_null() | key.is_null()).rechunk().into_owned();
                    if !ignore_errors {
                        let idx = ok.iter().position(|ok| ok == Some(false)).unwrap();
                        let key = key.str()?.get(idx).unwrap();
                        polars_bail!(
                            ComputeError:
                            "cannot decode JSON object key \"{key}\" as Map key of type `{key_dtype}`"
                        );
                    }
                    ok_keys = Some(ok.downcast_as_array().values().clone());
                }
                let value = from_json_decoded_rec(value, value_dtype, ignore_errors)?;
                Ok((decoded, value))
            })?
            .into_series();

            let storage = match ok_keys {
                Some(ok) => {
                    null_rows_where(&storage, |start, len| ok.null_count_range(start, len) > 0)
                },
                None => storage,
            };
            Ok(MapChunked::try_from_storage(target.clone(), storage)?.into_series())
        },
        DataType::List(inner) => {
            let ca = series.list()?;
            let values = from_json_decoded_rec(&ca.get_inner(), inner, ignore_errors)?;
            Ok(ca.with_inner_values(&values).into_series())
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(inner, width) => {
            let ca = series.list()?;
            let values = from_json_decoded_rec(&ca.get_inner(), inner, ignore_errors)?;
            let mut list = ca.with_inner_values(&values).into_series();
            if ignore_errors {
                list = null_rows_where(&list, |_, len| len != *width);
            }
            list.cast_with_options(target, CastOptions::Strict)
        },
        DataType::Struct(fields) => {
            let ca = series.struct_()?;
            let out = ca.try_apply_fields(|field| {
                match fields.iter().find(|f| f.name() == field.name()) {
                    Some(f) => from_json_decoded_rec(field, f.dtype(), ignore_errors),
                    None => Ok(field.clone()),
                }
            })?;
            Ok(out.into_series())
        },
        _ => cast_leaf(series.clone(), target, ignore_errors),
    }
}

/// Null the valid rows of a list series for which `bad(start, len)` holds. `start` indexes
/// the concatenated offset windows of all chunks.
#[cfg(feature = "dtype-map")]
fn null_rows_where(list: &Series, mut bad: impl FnMut(usize, usize) -> bool) -> Series {
    let ca = list.list().unwrap();
    let mut validity = BitmapBuilder::with_capacity(ca.len());
    let mut changed = false;
    let mut base = 0;
    for arr in ca.downcast_iter() {
        let offsets = arr.offsets();
        let first = *offsets.first() as usize;
        for row in 0..arr.len() {
            let (start, end) = offsets.start_end(row);
            let valid = arr.is_valid(row);
            let keep = valid && !bad(base + start - first, end - start);
            changed |= valid != keep;
            validity.push(keep);
        }
        base += offsets.range() as usize;
    }

    if changed {
        list.with_validity(Some(validity.freeze()))
    } else {
        list.clone()
    }
}

#[cfg(feature = "dtype-map")]
fn encode_with<V>(
    name: &PlSmallStr,
    iter: impl Iterator<Item = Option<V>>,
    mut f: impl FnMut(V, &mut String) -> PolarsResult<()>,
) -> PolarsResult<Series> {
    let mut buf = String::new();
    let mut builder = StringChunkedBuilder::new(name.clone(), iter.size_hint().0);
    for v in iter {
        match v {
            Some(v) => {
                buf.clear();
                f(v, &mut buf)?;
                builder.append_value(&buf);
            },
            None => builder.append_null(),
        }
    }
    Ok(builder.finish().into_series())
}

#[cfg(feature = "dtype-map")]
fn encode_integers<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsIntegerType,
    T::Native: itoa::Integer,
{
    encode_with(ca.name(), ca.iter(), |v, buf| {
        buf.push_str(itoa::Buffer::new().format(v));
        Ok(())
    })
}

#[cfg(feature = "dtype-map")]
fn encode_display<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsNumericType,
    T::Native: std::fmt::Display,
{
    encode_with(ca.name(), ca.iter(), |v, buf| {
        write!(buf, "{v}").unwrap();
        Ok(())
    })
}

#[cfg(all(feature = "dtype-map", feature = "dtype-datetime"))]
fn ticks_to_datetime(v: i64, tu: TimeUnit) -> Option<chrono::DateTime<chrono::Utc>> {
    match tu {
        TimeUnit::Nanoseconds => Some(chrono::DateTime::from_timestamp_nanos(v)),
        TimeUnit::Microseconds => chrono::DateTime::from_timestamp_micros(v),
        TimeUnit::Milliseconds => chrono::DateTime::from_timestamp_millis(v),
    }
}

#[cfg(all(feature = "dtype-map", feature = "dtype-datetime"))]
fn datetime_to_ticks(dt: chrono::DateTime<chrono::Utc>, tu: TimeUnit) -> Option<i64> {
    match tu {
        TimeUnit::Nanoseconds => dt.timestamp_nanos_opt(),
        TimeUnit::Microseconds => Some(dt.timestamp_micros()),
        TimeUnit::Milliseconds => Some(dt.timestamp_millis()),
    }
}

#[cfg(feature = "dtype-map")]
fn encode_keys(keys: &Series) -> PolarsResult<Series> {
    use DataType as D;
    let out_of_range = |v: &dyn std::fmt::Display| polars_err!(ComputeError: "cannot encode Map key {v} of type `{}` as a JSON object key", keys.dtype());
    match keys.dtype() {
        D::String => Ok(keys.clone()),
        #[cfg(feature = "dtype-categorical")]
        D::Categorical(..) | D::Enum(..) => keys.cast(&D::String),
        D::Boolean => encode_with(keys.name(), keys.bool()?.iter(), |v, buf| {
            buf.push_str(if v { "true" } else { "false" });
            Ok(())
        }),
        #[cfg(feature = "dtype-i8")]
        D::Int8 => encode_integers(keys.i8()?),
        #[cfg(feature = "dtype-i16")]
        D::Int16 => encode_integers(keys.i16()?),
        D::Int32 => encode_integers(keys.i32()?),
        D::Int64 => encode_integers(keys.i64()?),
        #[cfg(feature = "dtype-i128")]
        D::Int128 => encode_integers(keys.i128()?),
        #[cfg(feature = "dtype-u8")]
        D::UInt8 => encode_integers(keys.u8()?),
        #[cfg(feature = "dtype-u16")]
        D::UInt16 => encode_integers(keys.u16()?),
        D::UInt32 => encode_integers(keys.u32()?),
        D::UInt64 => encode_integers(keys.u64()?),
        #[cfg(feature = "dtype-u128")]
        D::UInt128 => encode_integers(keys.u128()?),
        // `Display` is the shortest round-trip form and keeps `NaN`, `inf` and `-0`.
        #[cfg(feature = "dtype-f16")]
        D::Float16 => encode_display(keys.f16()?),
        D::Float32 => encode_display(keys.f32()?),
        D::Float64 => encode_display(keys.f64()?),
        #[cfg(feature = "dtype-decimal")]
        D::Decimal(_, scale) => {
            let mut fmt = polars_compute::decimal::DecimalFmtBuffer::new();
            let ca = keys.decimal()?.physical();
            // Untrimmed, so distinct values at a fixed scale stay distinct.
            encode_with(ca.name(), ca.iter(), |v, buf| {
                buf.push_str(fmt.format_dec128(v, *scale, false, false));
                Ok(())
            })
        },
        #[cfg(feature = "dtype-date")]
        D::Date => {
            let ca = keys.date()?.physical();
            encode_with(ca.name(), ca.iter(), |v, buf| {
                let date = v
                    .checked_add(EPOCH_DAYS_FROM_CE)
                    .and_then(chrono::NaiveDate::from_num_days_from_ce_opt)
                    .ok_or_else(|| out_of_range(&v))?;
                write!(buf, "{}", date.format("%Y-%m-%d")).unwrap();
                Ok(())
            })
        },
        #[cfg(feature = "dtype-datetime")]
        D::Datetime(tu, tz) => {
            let ca = keys.datetime()?.physical();
            let (tu, has_tz) = (*tu, tz.is_some());
            encode_with(ca.name(), ca.iter(), |v, buf| {
                let dt = ticks_to_datetime(v, tu).ok_or_else(|| out_of_range(&v))?;
                if has_tz {
                    // The dtype carries the time zone, so the key only needs the instant.
                    buf.push_str(&dt.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true));
                } else {
                    write!(buf, "{}", dt.naive_utc().format(NAIVE_DATETIME_FMT)).unwrap();
                }
                Ok(())
            })
        },
        #[cfg(feature = "dtype-time")]
        D::Time => {
            let ca = keys.time()?.physical();
            encode_with(ca.name(), ca.iter(), |v, buf| {
                let time = chrono::NaiveTime::from_num_seconds_from_midnight_opt(
                    (v / 1_000_000_000) as u32,
                    (v % 1_000_000_000) as u32,
                )
                .ok_or_else(|| out_of_range(&v))?;
                write!(buf, "{}", time.format(TIME_FMT)).unwrap();
                Ok(())
            })
        },
        // The same integer count the JSON value reader accepts.
        #[cfg(feature = "dtype-duration")]
        D::Duration(_) => encode_integers(keys.duration()?.physical()),
        dt => polars_bail!(ComputeError: "JSON does not support Map keys of type `{dt}`"),
    }
}

#[cfg(feature = "dtype-map")]
fn decode_with<T: PolarsNumericType>(
    keys: &StringChunked,
    f: impl Fn(&str) -> Option<T::Native>,
) -> ChunkedArray<T> {
    ChunkedArray::from_iter_options(keys.name().clone(), keys.iter().map(|k| k.and_then(&f)))
}

#[cfg(feature = "dtype-map")]
fn decode_parse<T>(keys: &StringChunked) -> Series
where
    T: PolarsNumericType,
    T::Native: std::str::FromStr,
    ChunkedArray<T>: IntoSeries,
{
    decode_with::<T>(keys, |k| k.parse().ok()).into_series()
}

/// Decode JSON object keys as `dtype`; keys that do not decode become null.
#[cfg(feature = "dtype-map")]
fn decode_keys(keys: &StringChunked, dtype: &DataType) -> PolarsResult<Series> {
    use DataType as D;
    Ok(match dtype {
        D::String => keys.clone().into_series(),
        #[cfg(feature = "dtype-categorical")]
        D::Categorical(..) | D::Enum(..) => keys
            .clone()
            .into_series()
            .cast_with_options(dtype, CastOptions::NonStrict)?,
        D::Boolean => BooleanChunked::from_iter_options(
            keys.name().clone(),
            keys.iter().map(|k| match k? {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }),
        )
        .into_series(),
        #[cfg(feature = "dtype-i8")]
        D::Int8 => decode_parse::<Int8Type>(keys),
        #[cfg(feature = "dtype-i16")]
        D::Int16 => decode_parse::<Int16Type>(keys),
        D::Int32 => decode_parse::<Int32Type>(keys),
        D::Int64 => decode_parse::<Int64Type>(keys),
        #[cfg(feature = "dtype-i128")]
        D::Int128 => decode_parse::<Int128Type>(keys),
        #[cfg(feature = "dtype-u8")]
        D::UInt8 => decode_parse::<UInt8Type>(keys),
        #[cfg(feature = "dtype-u16")]
        D::UInt16 => decode_parse::<UInt16Type>(keys),
        D::UInt32 => decode_parse::<UInt32Type>(keys),
        D::UInt64 => decode_parse::<UInt64Type>(keys),
        #[cfg(feature = "dtype-u128")]
        D::UInt128 => decode_parse::<UInt128Type>(keys),
        #[cfg(feature = "dtype-f16")]
        D::Float16 => decode_with::<Float16Type>(keys, |k| {
            k.parse::<f32>().ok().map(polars_utils::float16::pf16::from)
        })
        .into_series(),
        D::Float32 => decode_parse::<Float32Type>(keys),
        D::Float64 => decode_parse::<Float64Type>(keys),
        #[cfg(feature = "dtype-decimal")]
        D::Decimal(p, s) => decode_with::<Int128Type>(keys, |k| {
            polars_compute::decimal::str_to_dec128(k.as_bytes(), *p, *s, false)
        })
        .into_decimal_unchecked(*p, *s)
        .into_series(),
        #[cfg(feature = "dtype-date")]
        D::Date => decode_with::<Int32Type>(keys, |k| {
            use chrono::Datelike;
            let date = chrono::NaiveDate::parse_from_str(k, "%Y-%m-%d").ok()?;
            Some(date.num_days_from_ce() - EPOCH_DAYS_FROM_CE)
        })
        .into_date()
        .into_series(),
        #[cfg(feature = "dtype-datetime")]
        D::Datetime(tu, tz) => {
            let tu = *tu;
            decode_with::<Int64Type>(keys, |k| {
                let dt = if tz.is_some() {
                    chrono::DateTime::parse_from_rfc3339(k).ok()?.to_utc()
                } else {
                    chrono::NaiveDateTime::parse_from_str(k, NAIVE_DATETIME_FMT)
                        .ok()?
                        .and_utc()
                };
                datetime_to_ticks(dt, tu)
            })
            .into_datetime(tu, tz.clone())
            .into_series()
        },
        #[cfg(feature = "dtype-time")]
        D::Time => decode_with::<Int64Type>(keys, |k| {
            use chrono::Timelike;
            let time = chrono::NaiveTime::parse_from_str(k, TIME_FMT).ok()?;
            Some(time.num_seconds_from_midnight() as i64 * 1_000_000_000 + time.nanosecond() as i64)
        })
        .into_time()
        .into_series(),
        #[cfg(feature = "dtype-duration")]
        D::Duration(tu) => decode_with::<Int64Type>(keys, |k| k.parse().ok())
            .into_duration(*tu)
            .into_series(),
        dt => polars_bail!(ComputeError: "JSON does not support Map keys of type `{dt}`"),
    })
}
