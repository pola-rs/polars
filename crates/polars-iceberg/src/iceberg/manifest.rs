//! Manifest list and manifest decoding (Avro), for format versions 1–3.
//!
//! Fields are identified by their Iceberg field ID (the `field-id` attribute in the writer's Avro
//! schema), falling back to the spec name only for writers that omit field IDs. Names are not
//! reliable: writers may sanitize them. Both v1 and v2+ layouts decode with the same code; absent
//! fields take their spec defaults, and unneeded fields are skipped without being materialized.
use polars_utils::aliases::PlHashSet;

use crate::iceberg::avro::{
    self, AvroFile, Datum, Record, Schema, for_each_item, read_datum, read_opt_bool,
    read_opt_bytes, read_opt_long, read_opt_str, skip,
};
use crate::iceberg::error::{IcebergResult, err_invalid_data};

pub const CONTENT_DATA: i32 = 0;
pub const CONTENT_POSITION_DELETES: i32 = 1;
pub const CONTENT_EQUALITY_DELETES: i32 = 2;

pub const STATUS_EXISTING: i32 = 0;
pub const STATUS_ADDED: i32 = 1;
pub const STATUS_DELETED: i32 = 2;

/// Manifest content in the manifest list.
pub const MANIFEST_CONTENT_DATA: i32 = 0;
pub const MANIFEST_CONTENT_DELETES: i32 = 1;

/// Reserved field ID of `file_path` in position delete files.
pub const DELETE_FILE_PATH_FIELD_ID: i32 = 2147483546;

/// Reserved field IDs of the manifest list, manifest entry and data file structs (Iceberg spec,
/// "Manifests" / "Manifest Lists"), with their spec names for writers that omit field IDs.
mod field_id {
    // manifest_file (manifest list)
    pub const MANIFEST_PATH: i32 = 500;
    pub const MANIFEST_LENGTH: i32 = 501;
    pub const PARTITION_SPEC_ID: i32 = 502;
    pub const ADDED_SNAPSHOT_ID: i32 = 503;
    pub const PARTITIONS: i32 = 507;
    pub const SEQUENCE_NUMBER_LIST: i32 = 515;
    pub const MIN_SEQUENCE_NUMBER: i32 = 516;
    pub const MANIFEST_CONTENT: i32 = 517;
    // field_summary
    pub const CONTAINS_NULL: i32 = 509;
    pub const LOWER_BOUND: i32 = 510;
    pub const UPPER_BOUND: i32 = 511;
    pub const CONTAINS_NAN: i32 = 518;
    // manifest_entry
    pub const STATUS: i32 = 0;
    pub const SNAPSHOT_ID: i32 = 1;
    pub const DATA_FILE: i32 = 2;
    pub const SEQUENCE_NUMBER: i32 = 3;
    pub const FILE_SEQUENCE_NUMBER: i32 = 4;
    // data_file
    pub const FILE_PATH: i32 = 100;
    pub const FILE_FORMAT: i32 = 101;
    pub const PARTITION: i32 = 102;
    pub const RECORD_COUNT: i32 = 103;
    pub const FILE_SIZE_IN_BYTES: i32 = 104;
    pub const VALUE_COUNTS: i32 = 109;
    pub const VALUE_COUNTS_KV: (i32, i32) = (119, 120);
    pub const NULL_VALUE_COUNTS: i32 = 110;
    pub const NULL_VALUE_COUNTS_KV: (i32, i32) = (121, 122);
    pub const NAN_VALUE_COUNTS: i32 = 137;
    pub const NAN_VALUE_COUNTS_KV: (i32, i32) = (138, 139);
    pub const LOWER_BOUNDS: i32 = 125;
    pub const LOWER_BOUNDS_KV: (i32, i32) = (126, 127);
    pub const UPPER_BOUNDS: i32 = 128;
    pub const UPPER_BOUNDS_KV: (i32, i32) = (129, 130);
    pub const CONTENT: i32 = 134;
    pub const REFERENCED_DATA_FILE: i32 = 143;
    pub const CONTENT_OFFSET: i32 = 144;
    pub const CONTENT_SIZE_IN_BYTES: i32 = 145;

    pub const MANIFEST_FILE_NAMES: &[(&str, i32)] = &[
        ("manifest_path", MANIFEST_PATH),
        ("manifest_length", MANIFEST_LENGTH),
        ("partition_spec_id", PARTITION_SPEC_ID),
        ("added_snapshot_id", ADDED_SNAPSHOT_ID),
        ("partitions", PARTITIONS),
        ("sequence_number", SEQUENCE_NUMBER_LIST),
        ("min_sequence_number", MIN_SEQUENCE_NUMBER),
        ("content", MANIFEST_CONTENT),
    ];
    pub const FIELD_SUMMARY_NAMES: &[(&str, i32)] = &[
        ("contains_null", CONTAINS_NULL),
        ("lower_bound", LOWER_BOUND),
        ("upper_bound", UPPER_BOUND),
        ("contains_nan", CONTAINS_NAN),
    ];
    pub const MANIFEST_ENTRY_NAMES: &[(&str, i32)] = &[
        ("status", STATUS),
        ("snapshot_id", SNAPSHOT_ID),
        ("data_file", DATA_FILE),
        ("sequence_number", SEQUENCE_NUMBER),
        ("file_sequence_number", FILE_SEQUENCE_NUMBER),
    ];
    pub const DATA_FILE_NAMES: &[(&str, i32)] = &[
        ("file_path", FILE_PATH),
        ("file_format", FILE_FORMAT),
        ("partition", PARTITION),
        ("record_count", RECORD_COUNT),
        ("file_size_in_bytes", FILE_SIZE_IN_BYTES),
        ("value_counts", VALUE_COUNTS),
        ("null_value_counts", NULL_VALUE_COUNTS),
        ("nan_value_counts", NAN_VALUE_COUNTS),
        ("lower_bounds", LOWER_BOUNDS),
        ("upper_bounds", UPPER_BOUNDS),
        ("content", CONTENT),
        ("referenced_data_file", REFERENCED_DATA_FILE),
        ("content_offset", CONTENT_OFFSET),
        ("content_size_in_bytes", CONTENT_SIZE_IN_BYTES),
    ];
}

/// Sentinel for fields this decoder does not use.
const UNKNOWN_FIELD: i32 = i32::MIN;

/// Field ID of each field of a writer record: its `field-id` attribute, or the spec ID for its
/// name if the writer omitted it.
fn record_field_ids(record: &Record, names: &[(&str, i32)]) -> Vec<i32> {
    record
        .fields
        .iter()
        .map(|f| {
            f.field_id.unwrap_or_else(|| {
                names
                    .iter()
                    .find(|(n, _)| *n == f.name)
                    .map_or(UNKNOWN_FIELD, |(_, id)| *id)
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct ManifestFile {
    pub path: String,
    pub length: i64,
    pub spec_id: i32,
    pub content: i32,
    pub sequence_number: i64,
    pub min_sequence_number: i64,
    pub added_snapshot_id: Option<i64>,
    pub partitions: Vec<FieldSummary>,
}

#[derive(Debug, Clone, Default)]
pub struct FieldSummary {
    pub contains_null: bool,
    pub contains_nan: Option<bool>,
    pub lower_bound: Option<Vec<u8>>,
    pub upper_bound: Option<Vec<u8>>,
}

pub fn parse_manifest_list(bytes: &[u8]) -> IcebergResult<Vec<ManifestFile>> {
    use field_id::*;

    let file = AvroFile::parse(bytes)?;
    let record = file.record()?;
    let ids = record_field_ids(record, MANIFEST_FILE_NAMES);
    require_fields(&ids, &[(MANIFEST_PATH, "manifest_path")], "manifest list")?;

    let mut out = Vec::with_capacity(file.objects_capacity_hint());
    file.for_each_object(|buf| {
        let mut m = ManifestFile {
            path: String::new(),
            length: 0,
            spec_id: 0,
            content: MANIFEST_CONTENT_DATA,
            sequence_number: 0,
            min_sequence_number: 0,
            added_snapshot_id: None,
            partitions: vec![],
        };

        for (field, id) in record.fields.iter().zip(&ids) {
            let s = &field.schema;
            match *id {
                MANIFEST_PATH => m.path = req(read_opt_str(s, buf)?, "manifest_path")?.to_owned(),
                MANIFEST_LENGTH => m.length = req(read_opt_long(s, buf)?, "manifest_length")?,
                PARTITION_SPEC_ID => m.spec_id = read_opt_long(s, buf)?.unwrap_or(0) as i32,
                MANIFEST_CONTENT => m.content = read_opt_long(s, buf)?.unwrap_or(0) as i32,
                SEQUENCE_NUMBER_LIST => m.sequence_number = read_opt_long(s, buf)?.unwrap_or(0),
                MIN_SEQUENCE_NUMBER => m.min_sequence_number = read_opt_long(s, buf)?.unwrap_or(0),
                ADDED_SNAPSHOT_ID => m.added_snapshot_id = read_opt_long(s, buf)?,
                PARTITIONS => m.partitions = read_field_summaries(s, buf)?,
                _ => skip(s, buf)?,
            }
        }

        out.push(m);
        Ok(())
    })?;

    Ok(out)
}

fn read_field_summaries(schema: &Schema, buf: &mut &[u8]) -> IcebergResult<Vec<FieldSummary>> {
    use field_id::*;

    let items = match avro::resolve(schema, buf)? {
        None => return Ok(vec![]),
        Some(Schema::Array(items)) => items,
        Some(_) => {
            return Err(err_invalid_data(
                "manifest list: partitions is not an array",
            ));
        },
    };
    let Schema::Record(record) = items.non_null() else {
        return Err(err_invalid_data("manifest list: invalid partition summary"));
    };
    let ids = record_field_ids(record, FIELD_SUMMARY_NAMES);
    let mut out = vec![];
    for_each_item(buf, |buf| {
        // `contains_null` is required: a writer without it says nothing about nulls.
        let mut s = FieldSummary {
            contains_null: true,
            ..Default::default()
        };
        for (field, id) in record.fields.iter().zip(&ids) {
            let schema = &field.schema;
            match *id {
                CONTAINS_NULL => s.contains_null = read_opt_bool(schema, buf)?.unwrap_or(true),
                CONTAINS_NAN => s.contains_nan = read_opt_bool(schema, buf)?,
                LOWER_BOUND => s.lower_bound = read_opt_bytes(schema, buf)?.map(<[u8]>::to_vec),
                UPPER_BOUND => s.upper_bound = read_opt_bytes(schema, buf)?.map(<[u8]>::to_vec),
                _ => skip(schema, buf)?,
            }
        }
        out.push(s);
        Ok(())
    })?;
    Ok(out)
}

/// A manifest with its live entries.
pub struct Manifest {
    /// Field IDs of the partition tuple fields, in the order of [`DataFile::partition`] (`None`
    /// if the writer did not record them).
    pub partition_field_ids: Vec<Option<i32>>,
    pub entries: Vec<ManifestEntry>,
}

#[derive(Debug)]
pub struct ManifestEntry {
    pub status: i32,
    pub snapshot_id: Option<i64>,
    pub sequence_number: Option<i64>,
    pub file_sequence_number: Option<i64>,
    pub file: DataFile,
}

#[derive(Debug, Default)]
pub struct DataFile {
    pub content: i32,
    pub file_path: String,
    pub file_format: String,
    pub partition: Vec<Option<Datum>>,
    pub record_count: i64,
    pub file_size_in_bytes: i64,
    /// Only for the requested field IDs.
    pub value_counts: Vec<(i32, i64)>,
    /// Only for the requested field IDs.
    pub null_value_counts: Vec<(i32, i64)>,
    /// Only for the requested field IDs.
    pub nan_value_counts: Vec<(i32, i64)>,
    /// Only for the requested field IDs.
    pub lower_bounds: Vec<(i32, Vec<u8>)>,
    /// Only for the requested field IDs.
    pub upper_bounds: Vec<(i32, Vec<u8>)>,
    pub referenced_data_file: Option<String>,
    pub content_offset: Option<i64>,
    pub content_size_in_bytes: Option<i64>,
}

impl DataFile {
    pub fn lower_bound(&self, id: i32) -> Option<&[u8]> {
        find(&self.lower_bounds, id)
    }

    pub fn upper_bound(&self, id: i32) -> Option<&[u8]> {
        find(&self.upper_bounds, id)
    }

    pub fn value_count(&self, id: i32) -> Option<i64> {
        find_count(&self.value_counts, id)
    }

    pub fn null_value_count(&self, id: i32) -> Option<i64> {
        find_count(&self.null_value_counts, id)
    }

    pub fn nan_value_count(&self, id: i32) -> Option<i64> {
        find_count(&self.nan_value_counts, id)
    }
}

fn find_count(kvs: &[(i32, i64)], id: i32) -> Option<i64> {
    kvs.iter().find(|(k, _)| *k == id).map(|(_, v)| *v)
}

fn find(kvs: &[(i32, Vec<u8>)], id: i32) -> Option<&[u8]> {
    kvs.iter()
        .find(|(k, _)| *k == id)
        .map(|(_, v)| v.as_slice())
}

/// Decode a manifest, dropping entries with status `DELETED`. Column statistics are kept only
/// for `stats_field_ids`.
pub fn parse_manifest(bytes: &[u8], stats_field_ids: &PlHashSet<i32>) -> IcebergResult<Manifest> {
    use field_id::*;

    let file = AvroFile::parse(bytes)?;
    let record = file.record()?;
    let entry_ids = record_field_ids(record, MANIFEST_ENTRY_NAMES);
    require_fields(&entry_ids, &[(STATUS, "status")], "manifest")?;

    let data_file_record = record
        .fields
        .iter()
        .zip(&entry_ids)
        .find(|(_, id)| **id == DATA_FILE)
        .and_then(|(f, _)| match f.schema.non_null() {
            Schema::Record(r) => Some(r),
            _ => None,
        })
        .ok_or_else(|| err_invalid_data("manifest without data_file record"))?;
    let data_file_ids = record_field_ids(data_file_record, DATA_FILE_NAMES);
    require_fields(
        &data_file_ids,
        &[(FILE_PATH, "file_path"), (RECORD_COUNT, "record_count")],
        "manifest data_file",
    )?;

    let partition_field_ids = data_file_record
        .fields
        .iter()
        .zip(&data_file_ids)
        .find(|(_, id)| **id == PARTITION)
        .map(|(f, _)| match f.schema.non_null() {
            Schema::Record(r) => r.fields.iter().map(|f| f.field_id).collect(),
            _ => vec![],
        })
        .unwrap_or_default();

    let data_file = DataFileDecoder {
        record: data_file_record,
        ids: &data_file_ids,
        stats_field_ids,
    };

    let mut entries = Vec::with_capacity(file.objects_capacity_hint());
    file.for_each_object(|buf| {
        let mut entry = ManifestEntry {
            status: STATUS_EXISTING,
            snapshot_id: None,
            sequence_number: None,
            file_sequence_number: None,
            file: DataFile::default(),
        };

        for (field, id) in record.fields.iter().zip(&entry_ids) {
            let s = &field.schema;
            match *id {
                STATUS => entry.status = req(read_opt_long(s, buf)?, "status")? as i32,
                SNAPSHOT_ID => entry.snapshot_id = read_opt_long(s, buf)?,
                SEQUENCE_NUMBER => entry.sequence_number = read_opt_long(s, buf)?,
                FILE_SEQUENCE_NUMBER => entry.file_sequence_number = read_opt_long(s, buf)?,
                DATA_FILE => {
                    if avro::resolve(s, buf)?.is_none() {
                        return Err(err_invalid_data("manifest: data_file is null"));
                    }
                    entry.file = data_file.decode(buf)?;
                },
                _ => skip(s, buf)?,
            }
        }

        if entry.status != STATUS_DELETED {
            entries.push(entry);
        }
        Ok(())
    })?;

    Ok(Manifest {
        partition_field_ids,
        entries,
    })
}

struct DataFileDecoder<'a> {
    record: &'a Record,
    ids: &'a [i32],
    stats_field_ids: &'a PlHashSet<i32>,
}

impl DataFileDecoder<'_> {
    fn decode(&self, buf: &mut &[u8]) -> IcebergResult<DataFile> {
        use field_id::*;

        let mut f = DataFile::default();
        // Bounds of `file_path` are needed to match position delete files to data files.
        let wanted_bounds =
            |id: i32| id == DELETE_FILE_PATH_FIELD_ID || self.stats_field_ids.contains(&id);

        for (field, id) in self.record.fields.iter().zip(self.ids) {
            let s = &field.schema;
            match *id {
                CONTENT => f.content = read_opt_long(s, buf)?.unwrap_or(0) as i32,
                FILE_PATH => f.file_path = req(read_opt_str(s, buf)?, "file_path")?.to_owned(),
                FILE_FORMAT => {
                    f.file_format = req(read_opt_str(s, buf)?, "file_format")?.to_ascii_uppercase()
                },
                PARTITION => match avro::resolve(s, buf)? {
                    None => {},
                    Some(Schema::Record(r)) => {
                        f.partition = r
                            .fields
                            .iter()
                            .map(|pf| read_datum(&pf.schema, buf))
                            .collect::<IcebergResult<_>>()?;
                    },
                    Some(_) => return Err(err_invalid_data("manifest: partition is not a record")),
                },
                RECORD_COUNT => f.record_count = req(read_opt_long(s, buf)?, "record_count")?,
                FILE_SIZE_IN_BYTES => {
                    f.file_size_in_bytes = req(read_opt_long(s, buf)?, "file_size_in_bytes")?
                },
                VALUE_COUNTS | NULL_VALUE_COUNTS | NAN_VALUE_COUNTS => {
                    let (kv_ids, out) = match *id {
                        VALUE_COUNTS => (VALUE_COUNTS_KV, &mut f.value_counts),
                        NULL_VALUE_COUNTS => (NULL_VALUE_COUNTS_KV, &mut f.null_value_counts),
                        _ => (NAN_VALUE_COUNTS_KV, &mut f.nan_value_counts),
                    };
                    read_int_map(s, buf, kv_ids, |id, value_schema, buf| {
                        let v = read_opt_long(value_schema, buf)?;
                        if let Some(v) = v
                            && self.stats_field_ids.contains(&id)
                        {
                            out.push((id, v));
                        }
                        Ok(())
                    })?;
                },
                LOWER_BOUNDS | UPPER_BOUNDS => {
                    let kv_ids = if *id == LOWER_BOUNDS {
                        LOWER_BOUNDS_KV
                    } else {
                        UPPER_BOUNDS_KV
                    };
                    let mut out = vec![];
                    read_int_map(s, buf, kv_ids, |id, value_schema, buf| {
                        if wanted_bounds(id) {
                            if let Some(v) = read_opt_bytes(value_schema, buf)? {
                                out.push((id, v.to_vec()));
                            }
                        } else {
                            skip(value_schema, buf)?;
                        }
                        Ok(())
                    })?;
                    if *id == LOWER_BOUNDS {
                        f.lower_bounds = out;
                    } else {
                        f.upper_bounds = out;
                    }
                },
                REFERENCED_DATA_FILE => {
                    f.referenced_data_file = read_opt_str(s, buf)?.map(str::to_owned)
                },
                CONTENT_OFFSET => f.content_offset = read_opt_long(s, buf)?,
                CONTENT_SIZE_IN_BYTES => f.content_size_in_bytes = read_opt_long(s, buf)?,
                _ => skip(s, buf)?,
            }
        }

        Ok(f)
    }
}

/// Read an Iceberg int-keyed map, stored either as an array of `{key, value}` records (the key
/// and value fields identified by `kv_ids`, or by name) or as an Avro map with string keys.
fn read_int_map(
    schema: &Schema,
    buf: &mut &[u8],
    (key_id, value_id): (i32, i32),
    mut f: impl FnMut(i32, &Schema, &mut &[u8]) -> IcebergResult<()>,
) -> IcebergResult<()> {
    match avro::resolve(schema, buf)? {
        None => Ok(()),
        Some(Schema::Array(items)) => {
            let Schema::Record(kv) = items.non_null() else {
                return Err(err_invalid_data("manifest: invalid map encoding"));
            };
            let ids = record_field_ids(kv, &[("key", key_id), ("value", value_id)]);
            let (Some(key_idx), Some(value_idx)) = (
                ids.iter().position(|id| *id == key_id),
                ids.iter().position(|id| *id == value_id),
            ) else {
                return Err(err_invalid_data("manifest: map entry without key or value"));
            };
            if value_idx < key_idx {
                return Err(err_invalid_data("manifest: map value before key"));
            }
            for_each_item(buf, |buf| {
                let mut key = None;
                for (i, field) in kv.fields.iter().enumerate() {
                    if i == key_idx {
                        key = read_opt_long(&field.schema, buf)?;
                    } else if i == value_idx {
                        let k = key.ok_or_else(|| err_invalid_data("manifest: null map key"))?;
                        f(k as i32, &field.schema, buf)?;
                    } else {
                        skip(&field.schema, buf)?;
                    }
                }
                Ok(())
            })
        },
        Some(Schema::Map(values)) => for_each_item(buf, |buf| {
            let key = avro::read_str(buf)?;
            let key: i32 = key
                .parse()
                .map_err(|_| err_invalid_data(format!("manifest: invalid map key '{key}'")))?;
            f(key, values, buf)
        }),
        Some(other) => Err(err_invalid_data(format!(
            "manifest: unexpected map schema {other:?}"
        ))),
    }
}

fn require_fields(ids: &[i32], required: &[(i32, &str)], what: &str) -> IcebergResult<()> {
    match required.iter().find(|(id, _)| !ids.contains(id)) {
        Some((_, name)) => Err(err_invalid_data(format!("{what} without field '{name}'"))),
        None => Ok(()),
    }
}

fn req<T>(v: Option<T>, name: &str) -> IcebergResult<T> {
    v.ok_or_else(|| err_invalid_data(format!("manifest field '{name}' is null")))
}
