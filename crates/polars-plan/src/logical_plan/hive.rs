use std::path::Path;

#[cfg(feature = "dtype-date")]
use chrono::NaiveDate;
#[cfg(feature = "dtype-datetime")]
use chrono::NaiveDateTime;
use percent_encoding::percent_decode_str;
use polars_core::prelude::*;
use polars_io::predicates::{BatchStats, ColumnStats};
use polars_io::utils::{BOOLEAN_RE, DATETIME_RE, DATE_RE, FLOAT_RE, INTEGER_RE};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct HivePartitions {
    /// Single value Series that can be used to run the predicate against.
    /// They are to be broadcasted if the predicates don't filter them out.
    stats: BatchStats,
}

#[cfg(target_os = "windows")]
fn separator(url: &Path) -> char {
    if polars_io::is_cloud_url(url) {
        '/'
    } else {
        '\\'
    }
}

#[cfg(not(target_os = "windows"))]
fn separator(_url: &Path) -> char {
    '/'
}

impl HivePartitions {
    pub fn get_statistics(&self) -> &BatchStats {
        &self.stats
    }

    /// Parse a url and optionally return HivePartitions
    pub(crate) fn parse_url(url: &Path) -> Option<Self> {
        let sep = separator(url);

        let url_string = url.display().to_string();

        let pre_filt = url_string.split(sep);

        let split_count_m1 = pre_filt.clone().count() - 1;

        let partitions = pre_filt
            .enumerate()
            .filter_map(|(index, part)| {
                let mut it = part.split('=');
                let name = it.next()?;
                let value = it.next()?;

                // Don't see files `foo=1.parquet` as hive partitions.
                // So we return globs and paths with extensions.
                if value.contains('*') {
                    return None;
                }

                // Identify file by index location
                if index == split_count_m1 {
                    return None;
                }

                // Having multiple '=' doesn't seem like valid hive partition,
                // continue as url.
                if it.next().is_some() {
                    return None;
                }

                let data_type = if INTEGER_RE.is_match(value) {
                    DataType::Int64
                } else if BOOLEAN_RE.is_match(value) {
                    DataType::Boolean
                } else if FLOAT_RE.is_match(value) {
                    DataType::Float64
                } else if value == "__HIVE_DEFAULT_PARTITION__" {
                    DataType::Null
                } else if DATE_RE.is_match(value) {
                    DataType::Date
                } else if DATETIME_RE.is_match(value) {
                    let tz = value.ends_with('Z').then_some(TimeZone::from("UTC"));
                    let time_unit = match value
                        .chars()
                        .rev()
                        .position(|c| c == '.')
                        .unwrap_or_default()
                        / 3
                    {
                        2 => TimeUnit::Microseconds,
                        3 => TimeUnit::Nanoseconds,
                        // Matches both seconds (no '.' found in the path) and milliseconds (3 digits after '.')
                        _ => TimeUnit::Milliseconds,
                    };
                    DataType::Datetime(time_unit, tz)
                } else {
                    DataType::String
                };

                let s = match data_type {
                    DataType::Int64 => {
                        let value = value.parse::<i64>().ok()?;
                        Series::new(name, &[value])
                    },
                    DataType::Boolean => {
                        let value = value.parse::<bool>().ok()?;
                        Series::new(name, &[value])
                    },
                    DataType::Float64 => {
                        let value = value.parse::<f64>().ok()?;
                        Series::new(name, &[value])
                    },
                    DataType::Null => Series::new_null(name, 1),
                    #[cfg(feature = "dtype-date")]
                    DataType::Date => {
                        let value = value.parse::<NaiveDate>().ok()?;
                        Series::new(name, &[value])
                    },
                    #[cfg(feature = "dtype-datetime")]
                    DataType::Datetime(time_unit, tz) => {
                        let cow_value = percent_decode_str(value).decode_utf8().ok()?;
                        let str_value = cow_value.as_ref();
                        let fmt = if tz.is_some() {
                            "%Y-%m-%d %H:%M:%S%.fZ"
                        } else {
                            "%Y-%m-%d %H:%M:%S%.f"
                        };
                        let value = NaiveDateTime::parse_from_str(str_value, fmt).ok()?;
                        let mut datetime_chunked =
                            DatetimeChunked::from_naive_datetime(name, [value], time_unit);
                        if let Some(tz) = tz {
                            datetime_chunked.set_time_zone(tz).ok()?;
                        }
                        datetime_chunked.into_series()
                    },
                    _ => Series::new(name, &[percent_decode_str(value).decode_utf8().ok()?]),
                };
                Some(s)
            })
            .collect::<Vec<_>>();

        if partitions.is_empty() {
            None
        } else {
            let schema: Schema = partitions.as_slice().into();
            let stats = BatchStats::new(
                Arc::new(schema),
                partitions
                    .into_iter()
                    .map(ColumnStats::from_column_literal)
                    .collect(),
                None,
            );

            Some(HivePartitions { stats })
        }
    }

    pub(crate) fn schema(&self) -> &SchemaRef {
        self.get_statistics().schema()
    }

    pub fn materialize_partition_columns(&self) -> Vec<Series> {
        self.get_statistics()
            .column_stats()
            .iter()
            .map(|cs| cs.get_min_state().unwrap().clone())
            .collect()
    }
}
