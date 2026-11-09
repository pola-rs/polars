use std::borrow::Cow;
use std::path::{Component, Path};

use polars_core::prelude::*;
use polars_io::prelude::schema_inference::{finish_infer_field_schema, infer_field_schema};
use polars_utils::pl_path::PlRefPath;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::dsl::HivePartitions;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct HivePartitionsDf(DataFrame);

impl HivePartitionsDf {
    pub fn try_new(partitions: &HivePartitions, num_sources: usize) -> PolarsResult<Self> {
        polars_ensure!(
            partitions.0.height() == num_sources,
            ShapeMismatch:
            "number of hive partition rows ({}) does not match number of scan sources ({})",
            partitions.0.height(),
            num_sources,
        );
        Ok(Self(partitions.0.as_ref().clone()))
    }

    /// Filter the columns to those contained in `projected_columns`.
    pub fn filter_columns(&self, projected_columns: &Schema) -> Self {
        let columns: Vec<_> = self
            .df()
            .columns()
            .iter()
            .filter(|c| projected_columns.contains(c.name()))
            .cloned()
            .collect();

        let height = self.df().height();
        unsafe { DataFrame::new_unchecked(height, columns) }.into()
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

// generate an iterator for path segments
fn get_normal_components(path: &Path) -> Box<dyn Iterator<Item = &str> + '_> {
    Box::new(path.components().filter_map(|c| match c {
        Component::Normal(seg) => Some(seg.to_str().unwrap()),
        _ => None,
    }))
}

fn parse_hive_string_and_decode(part: &'_ str) -> Option<(&'_ str, std::borrow::Cow<'_, str>)> {
    let (k, v) = parse_hive_string(part)?;
    let v = percent_encoding::percent_decode(v.as_bytes())
        .decode_utf8()
        .ok()?;

    Some((k, v))
}

// generate (k,v) tuples from 'k=v' partition strings
pub fn get_hive_parts_iter(path: &PlRefPath) -> impl Iterator<Item = (&str, Cow<'_, str>)> {
    let path = path.as_std_path();
    let file_index = get_normal_components(path).count() - 1;
    let path_parts = get_normal_components(path);

    path_parts.enumerate().filter_map(move |(index, part)| {
        if index == file_index {
            return None;
        }

        parse_hive_string_and_decode(part)
    })
}

/// Note: Returned hive partitions are ordered by their position in the `reader_schema`
///
/// # Safety
/// `hive_start_idx <= [min path length]`
pub fn hive_partitions_from_paths(
    paths: &[PlRefPath],
    hive_start_idx: usize,
    schema: Option<SchemaRef>,
    reader_schema: &Schema,
    try_parse_dates: bool,
) -> PolarsResult<Option<HivePartitionsDf>> {
    let Some(path) = paths.first() else {
        return Ok(None);
    };

    let hive_schema = if let Some(ref schema) = schema {
        let path = path.sliced(hive_start_idx..path.as_str().len());
        Arc::new(get_hive_parts_iter(&path).map(|(name, _)| {
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
        let path = path.sliced(hive_start_idx..path.as_str().len());

        let mut hive_schema = Schema::with_capacity(16);
        let mut schema_inference_map: PlIndexMap<&str, PlIndexSet<DataType>> =
            PlIndexMap::with_capacity(16);

        for (name, _) in get_hive_parts_iter(&path) {
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
            schema_inference_map.insert(name, PlIndexSet::with_capacity(4));
        }

        if hive_schema.is_empty() && schema_inference_map.is_empty() {
            return Ok(None);
        }

        if !schema_inference_map.is_empty() {
            for path in paths {
                let path = path.sliced(hive_start_idx..path.as_str().len());
                for (name, value) in get_hive_parts_iter(&path) {
                    let Some(entry) = schema_inference_map.get_mut(name) else {
                        continue;
                    };

                    if value.is_empty() || value == "__HIVE_DEFAULT_PARTITION__" {
                        continue;
                    }

                    entry.insert(infer_field_schema(value.as_ref(), try_parse_dates, false));
                }
            }

            for (name, ref possibilities) in schema_inference_map.drain(..) {
                let dtype = finish_infer_field_schema(possibilities);
                *hive_schema.try_get_mut(name).unwrap() = dtype;
            }
        }
        Arc::new(hive_schema)
    };

    let parse_schema: Schema = hive_schema
        .iter()
        .map(|(name, dtype)| Field::new(name.clone(), parse_dtype(dtype)))
        .collect();

    let mut buffers = polars_io::csv::read::builder::init_builders(
        &(0..hive_schema.len()).collect::<Vec<_>>(),
        paths.len(),
        &parse_schema,
        None,
        polars_io::prelude::CsvEncoding::Utf8,
        false,
    )?;

    for path in paths {
        let path = path.sliced(hive_start_idx..path.as_str().len());
        for (name, value) in get_hive_parts_iter(&path) {
            let Some(index) = hive_schema.index_of(name) else {
                polars_bail!(
                    SchemaFieldNotFound:
                    "path contains column not present in the given Hive schema: {:?}, path = {:?}",
                    name,
                    path
                )
            };

            let buf = buffers.get_mut(index).unwrap();

            if value != "__HIVE_DEFAULT_PARTITION__" {
                // An empty value is null, except for `String` columns.
                let missing_is_null = !hive_schema.get_at_index(index).unwrap().1.is_string();
                buf.add(value.as_bytes(), false, false, missing_is_null)?;
            } else {
                buf.add_null(false);
            }
        }
    }

    let mut buffers = buffers
        .into_iter()
        .zip(hive_schema.iter_values())
        .map(|(x, dtype)| finish_parsed_column(x.into_series()?, dtype))
        .collect::<PolarsResult<Vec<_>>>()?;
    buffers.sort_by_key(|s| reader_schema.index_of(s.name()).unwrap_or(usize::MAX));

    Ok(Some(HivePartitionsDf(DataFrame::new(
        paths.len(),
        buffers,
    )?)))
}

/// The dtype used to parse Hive values of `dtype`. The CSV builders cannot parse `Time`, so it is
/// parsed as `String` and converted in [`finish_parsed_column`], as the CSV reader does.
fn parse_dtype(dtype: &DataType) -> DataType {
    match dtype {
        #[cfg(feature = "dtype-time")]
        DataType::Time => DataType::String,
        dt => dt.clone(),
    }
}

fn finish_parsed_column(s: Series, dtype: &DataType) -> PolarsResult<Column> {
    match dtype {
        #[cfg(feature = "dtype-time")]
        DataType::Time => {
            use polars_core::chunked_array::temporal::string::StringMethods;

            let out = s.str()?.as_time(None, false)?.into_series();
            if out.null_count() != s.null_count() {
                polars_core::utils::handle_casting_failures(&s, &out)?;
            }
            Ok(out.into_column())
        },
        _ => Ok(s.into_column()),
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
