use std::path::{Path, PathBuf};

use percent_encoding::percent_decode_str;
use polars_core::prelude::*;
use polars_io::predicates::{BatchStats, ColumnStats};
use polars_io::prelude::schema_inference::{finish_infer_field_schema, infer_field_schema};
use polars_io::utils::{BOOLEAN_RE, FLOAT_RE, INTEGER_RE};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct HivePartitions {
    /// Single value Series that can be used to run the predicate against.
    /// They are to be broadcasted if the predicates don't filter them out.
    stats: BatchStats,
}

impl HivePartitions {
    pub fn get_projection_schema_and_indices<T: AsRef<str>>(
        &self,
        names: &[T],
    ) -> (SchemaRef, Vec<usize>) {
        let names = names.iter().map(T::as_ref).collect::<PlHashSet<&str>>();
        let mut out_schema = Schema::with_capacity(self.stats.schema().len());
        let mut out_indices = Vec::with_capacity(self.stats.column_stats().len());

        for (i, cs) in self.stats.column_stats().iter().enumerate() {
            let name = cs.field_name();
            if names.contains(name.as_str()) {
                out_indices.push(i);
                out_schema
                    .insert_at_index(out_schema.len(), name.clone(), cs.dtype().clone())
                    .unwrap();
            }
        }

        (out_schema.into(), out_indices)
    }

    pub fn apply_projection(&mut self, new_schema: SchemaRef, column_indices: &[usize]) {
        self.stats.with_schema(new_schema);
        self.stats.take_indices(column_indices);
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

/// # Safety
/// `hive_start_idx <= [min path length]`
pub fn hive_partitions_from_paths(
    paths: &[PathBuf],
    hive_start_idx: usize,
    schema: Option<SchemaRef>,
) -> PolarsResult<Option<Arc<[HivePartitions]>>> {
    let Some(path) = paths.first() else {
        return Ok(None);
    };

    let sep = separator(path);
    let path_string = path.to_str().unwrap();

    macro_rules! get_hive_parts_iter {
        ($e:expr) => {{
            let path_parts = $e[hive_start_idx..].split(sep);
            let file_index = path_parts.clone().count() - 1;

            path_parts.enumerate().filter_map(move |(index, part)| {
                if index == file_index {
                    return None;
                }
                parse_hive_string(part)
            })
        }};
    }

    let hive_schema = if let Some(v) = schema {
        v
    } else {
        let mut hive_schema = Schema::with_capacity(16);
        let mut schema_inference_map: PlHashMap<&str, PlHashSet<DataType>> =
            PlHashMap::with_capacity(16);

        for (name, _) in get_hive_parts_iter!(path_string) {
            hive_schema.insert_at_index(hive_schema.len(), name.into(), DataType::String)?;
            schema_inference_map.insert(name, PlHashSet::with_capacity(4));
        }

        if hive_schema.is_empty() && schema_inference_map.is_empty() {
            return Ok(None);
        }

        if !schema_inference_map.is_empty() {
            for path in paths {
                for (name, value) in get_hive_parts_iter!(path.to_str().unwrap()) {
                    let Some(entry) = schema_inference_map.get_mut(name) else {
                        continue;
                    };

                    entry.insert(infer_field_schema(value, false, false));
                }
            }

            for (name, ref possibilities) in schema_inference_map.drain() {
                let dtype = finish_infer_field_schema(possibilities);
                *hive_schema.try_get_mut(name).unwrap() = dtype;
            }
        }
        Arc::new(hive_schema)
    };

    let mut hive_partitions = Vec::with_capacity(paths.len());

    for path in paths {
        let path = path.to_str().unwrap();

        let column_stats = get_hive_parts_iter!(path)
            .map(|(name, value)| {
                let Some(dtype) = hive_schema.as_ref().get(name) else {
                    polars_bail!(
                        SchemaFieldNotFound:
                        "path contains column not present in the given Hive schema: {:?}, path = {:?}",
                        name,
                        path
                    )
                };

                Ok(ColumnStats::from_column_literal(value_to_series(
                    name,
                    value,
                    Some(dtype),
                )?))
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        if column_stats.is_empty() {
            polars_bail!(
                ComputeError: "expected Hive partitioned path, got {}\n\n\
                This error occurs if some paths are Hive partitioned and some paths are not.",
                path
            )
        }

        let stats = BatchStats::new(hive_schema.clone(), column_stats, None);

        hive_partitions.push(HivePartitions { stats });
    }

    Ok(Some(Arc::from(hive_partitions)))
}

/// Determine the path separator for identifying Hive partitions.
#[cfg(target_os = "windows")]
fn separator(url: &Path) -> char {
    if polars_io::utils::is_cloud_url(url) {
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
///
/// Returns `None` if the string is not a Hive partition string.
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

/// Parse a string value into a single-value [`Series`].
fn value_to_series(name: &str, value: &str, dtype: Option<&DataType>) -> PolarsResult<Series> {
    let fn_err = || polars_err!(ComputeError: "unable to parse Hive partition value: {:?}", value);

    let mut s = if INTEGER_RE.is_match(value) {
        let value = value.parse::<i64>().map_err(|_| fn_err())?;
        Series::new(name, &[value])
    } else if BOOLEAN_RE.is_match(value) {
        let value = value.parse::<bool>().map_err(|_| fn_err())?;
        Series::new(name, &[value])
    } else if FLOAT_RE.is_match(value) {
        let value = value.parse::<f64>().map_err(|_| fn_err())?;
        Series::new(name, &[value])
    } else if value == "__HIVE_DEFAULT_PARTITION__" {
        Series::new_null(name, 1)
    } else {
        let value = percent_decode_str(value)
            .decode_utf8()
            .map_err(|_| fn_err())?;
        Series::new(name, &[value])
    };

    // TODO: Avoid expensive logic above when dtype is known
    if let Some(dt) = dtype {
        s = s.strict_cast(dt)?;
    }

    Ok(s)
}
