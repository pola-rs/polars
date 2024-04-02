use std::path::Path;

use percent_encoding::percent_decode_str;
use polars_core::prelude::*;
use polars_io::predicates::{BatchStats, ColumnStats};
use polars_io::utils::{BOOLEAN_RE, FLOAT_RE, INTEGER_RE};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct HivePartitions {
    /// Single value Series that can be used to run the predicate against.
    /// They are to be broadcasted if the predicates don't filter them out.
    stats: BatchStats,
}

impl HivePartitions {
    /// Constructs a new [`HivePartitions`] from a schema reference.
    pub fn from_schema_ref(schema: SchemaRef) -> Self {
        let column_stats = schema.iter_fields().map(ColumnStats::from_field).collect();
        let stats = BatchStats::new(schema, column_stats, None);
        Self { stats }
    }

    /// Constructs a new [`HivePartitions`] from a path.
    ///
    /// Fails if the path does not contain any Hive partitions.
    pub fn try_from_path(path: &Path, schema: Option<SchemaRef>) -> PolarsResult<Self> {
        let sep = separator(path);

        let path_string = path.display().to_string();
        let path_parts = path_string.split(sep);

        // Last part is the file, which should be skipped.
        let file_index = path_parts.clone().count() - 1;

        let partitions = path_parts
            .enumerate()
            .filter_map(|(index, part)| {
                if index == file_index {
                    return None;
                }
                parse_hive_string(part)
            })
            .map(|(name, value)| hive_info_to_series(name, value, schema.clone()))
            .collect::<PolarsResult<Vec<_>>>()?;

        polars_ensure!(
            !partitions.is_empty(),
            ComputeError: "path does not contain Hive partition information"
        );

        let schema = match schema {
            Some(s) => {
                polars_ensure!(
                    s.len() == partitions.len(),
                    SchemaMismatch: "path does not match the provided Hive schema"
                );
                s
            },
            None => Arc::new(partitions.as_slice().into()),
        };

        let stats = BatchStats::new(
            schema,
            partitions
                .into_iter()
                .map(ColumnStats::from_column_literal)
                .collect(),
            None,
        );

        Ok(HivePartitions { stats })
    }

    pub fn get_statistics(&self) -> &BatchStats {
        &self.stats
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

/// Determine the path separator for identifying Hive partitions.
#[cfg(target_os = "windows")]
fn separator(url: &Path) -> char {
    if polars_io::is_cloud_url(url) {
        '/'
    } else {
        '\\'
    }
}

/// Determine the path separator for identifying Hive partitions.
#[cfg(not(target_os = "windows"))]
fn separator(_url: &Path) -> char {
    '/'
}

/// Parse a Hive partition string (e.g. "column=1.5") into a name and value part.
fn parse_hive_string(part: &'_ str) -> Option<(&'_ str, &'_ str)> {
    let mut it = part.split('=');
    let name = it.next()?;
    let value = it.next()?;

    // Having multiple '=' doesn't seem like a valid Hive partition.
    if it.next().is_some() {
        return None;
    }

    // Files are not Hive partitions, so globs are not valid.
    if value.contains('*') {
        return None;
    }

    Some((name, value))
}

/// Convert Hive partition string information to a single-value [`Series`].
fn hive_info_to_series(name: &str, value: &str, schema: Option<SchemaRef>) -> PolarsResult<Series> {
    let dtype = match schema {
        Some(ref s) => {
            let dtype = s.try_get(name).map_err(|_| {
                polars_err!(
                    SchemaFieldNotFound:
                    "path contains column not present in the given Hive schema: {:?}", name
                )
            })?;
            Some(dtype)
        },
        None => None,
    };

    value_to_series(name, value, dtype).ok_or_else(|| polars_err!(ComputeError: "..."))
}

fn value_to_series(name: &str, value: &str, dtype: Option<&DataType>) -> Option<Series> {
    let mut s = if INTEGER_RE.is_match(value) {
        let value = value.parse::<i64>().ok()?;
        Series::new(name, &[value])
    } else if BOOLEAN_RE.is_match(value) {
        let value = value.parse::<bool>().ok()?;
        Series::new(name, &[value])
    } else if FLOAT_RE.is_match(value) {
        let value = value.parse::<f64>().ok()?;
        Series::new(name, &[value])
    } else if value == "__HIVE_DEFAULT_PARTITION__" {
        Series::new_null(name, 1)
    } else {
        Series::new(name, &[percent_decode_str(value).decode_utf8().ok()?])
    };

    // TODO: Avoid expensive logic above when dtype is known
    if let Some(dt) = dtype {
        s = s.strict_cast(dt).ok()?;
    }

    Some(s)
}
