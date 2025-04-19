use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::prelude::schema_inference::{finish_infer_field_schema, infer_field_schema};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct HivePartitionsDf(DataFrame);

impl HivePartitionsDf {
    pub fn get_projection_schema_and_indices(
        &self,
        names: &PlHashSet<PlSmallStr>,
    ) -> (SchemaRef, Vec<usize>) {
        let mut out_schema = Schema::with_capacity(self.schema().len());
        let mut out_indices = Vec::with_capacity(self.0.get_columns().len());

        for (i, column) in self.0.get_columns().iter().enumerate() {
            let name = column.name();
            if names.contains(name.as_str()) {
                out_indices.push(i);
                out_schema
                    .insert_at_index(out_schema.len(), name.clone(), column.dtype().clone())
                    .unwrap();
            }
        }

        (out_schema.into(), out_indices)
    }

    pub fn apply_projection(&mut self, column_indices: &[usize]) {
        let schema = self.schema();
        let projected_schema = schema.try_project_indices(column_indices).unwrap();
        self.0 = self.0.select(projected_schema.iter_names_cloned()).unwrap();
    }

    pub fn take_indices(&self, row_indexes: &[IdxSize]) -> Self {
        if !row_indexes.is_empty() {
            let mut max_idx = 0;
            for &i in row_indexes {
                max_idx = max_idx.max(i);
            }
            assert!(max_idx < self.0.height() as IdxSize);
        }
        // SAFETY: Checked bounds before.
        Self(unsafe { self.0.take_slice_unchecked(row_indexes) })
    }

    pub fn df(&self) -> &DataFrame {
        &self.0
    }

    pub fn schema(&self) -> &SchemaRef {
        self.0.schema()
    }
}

impl From<DataFrame> for HivePartitionsDf {
    fn from(value: DataFrame) -> Self {
        Self(value)
    }
}

/// Note: Returned hive partitions are ordered by their position in the `reader_schema`
///
/// # Safety
/// `hive_start_idx <= [min path length]`
pub fn hive_partitions_from_paths(
    paths: &[PathBuf],
    hive_start_idx: usize,
    schema: Option<SchemaRef>,
    reader_schema: &Schema,
    try_parse_dates: bool,
) -> PolarsResult<Option<HivePartitionsDf>> {
    let Some(path) = paths.first() else {
        return Ok(None);
    };

    let sep = separator(path);
    let path_string = path.to_str().unwrap();

    fn parse_hive_string_and_decode(part: &'_ str) -> Option<(&'_ str, std::borrow::Cow<'_, str>)> {
        let (k, v) = parse_hive_string(part)?;
        let v = percent_encoding::percent_decode(v.as_bytes())
            .decode_utf8()
            .ok()?;

        Some((k, v))
    }

    macro_rules! get_hive_parts_iter {
        ($e:expr) => {{
            let path_parts = $e[hive_start_idx..].split(sep);
            let file_index = path_parts.clone().count() - 1;

            path_parts.enumerate().filter_map(move |(index, part)| {
                if index == file_index {
                    return None;
                }

                parse_hive_string_and_decode(part)
            })
        }};
    }

    let hive_schema = if let Some(ref schema) = schema {
        Arc::new(get_hive_parts_iter!(path_string).map(|(name, _)| {
                let Some(dtype) = schema.get(name) else {
                    polars_bail!(
                        SchemaFieldNotFound:
                        "path contains column not present in the given Hive schema: {:?}, path = {:?}",
                        name,
                        path
                    )
                };

                let dtype = if !try_parse_dates && dtype.is_temporal() {
                    DataType::String
                } else {
                    dtype.clone()
                };

                Ok(Field::new(PlSmallStr::from_str(name), dtype))
            }).collect::<PolarsResult<Schema>>()?)
    } else {
        let mut hive_schema = Schema::with_capacity(16);
        let mut schema_inference_map: PlHashMap<&str, PlHashSet<DataType>> =
            PlHashMap::with_capacity(16);

        for (name, _) in get_hive_parts_iter!(path_string) {
            // If the column is also in the file we can use the dtype stored there.
            if let Some(dtype) = reader_schema.get(name) {
                let dtype = if !try_parse_dates && dtype.is_temporal() {
                    DataType::String
                } else {
                    dtype.clone()
                };

                hive_schema.insert_at_index(hive_schema.len(), name.into(), dtype.clone())?;
                continue;
            }

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

                    if value.is_empty() || value == "__HIVE_DEFAULT_PARTITION__" {
                        continue;
                    }

                    entry.insert(infer_field_schema(value.as_ref(), try_parse_dates, false));
                }
            }

            for (name, ref possibilities) in schema_inference_map.drain() {
                let dtype = finish_infer_field_schema(possibilities);
                *hive_schema.try_get_mut(name).unwrap() = dtype;
            }
        }
        Arc::new(hive_schema)
    };

    let mut buffers = polars_io::csv::read::buffer::init_buffers(
        &(0..hive_schema.len()).collect::<Vec<_>>(),
        paths.len(),
        hive_schema.as_ref(),
        None,
        polars_io::prelude::CsvEncoding::Utf8,
        false,
    )?;

    for path in paths {
        let path = path.to_str().unwrap();

        for (name, value) in get_hive_parts_iter!(path) {
            let Some(index) = hive_schema.index_of(name) else {
                polars_bail!(
                    SchemaFieldNotFound:
                    "path contains column not present in the given Hive schema: {:?}, path = {:?}",
                    name,
                    path
                )
            };

            let buf = buffers.get_mut(index).unwrap();

            if !value.is_empty() && value != "__HIVE_DEFAULT_PARTITION__" {
                buf.add(value.as_bytes(), false, false, false)?;
            } else {
                buf.add_null(false);
            }
        }
    }

    let mut buffers = buffers
        .into_iter()
        .map(|x| Ok(x.into_series()?.into_column()))
        .collect::<PolarsResult<Vec<_>>>()?;
    buffers.sort_by_key(|s| reader_schema.index_of(s.name()).unwrap_or(usize::MAX));

    Ok(Some(HivePartitionsDf(DataFrame::new_with_height(
        paths.len(),
        buffers,
    )?)))
}

/// Determine the path separator for identifying Hive partitions.
fn separator(url: &Path) -> &[char] {
    if cfg!(target_family = "windows") {
        if polars_io::path_utils::is_cloud_url(url) {
            &['/']
        } else {
            &['/', '\\']
        }
    } else {
        &['/']
    }
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
    };

    Some((name, value))
}
