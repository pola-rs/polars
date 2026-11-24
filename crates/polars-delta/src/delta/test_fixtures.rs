//! Generate small, physical Parquet checkpoint fixtures for the Delta planner tests.
use std::fmt::Display;
use std::sync::Arc;

use delta_kernel::arrow::array::{ArrayRef, BinaryArray};
use delta_kernel::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use delta_kernel::arrow::error::ArrowError;
use delta_kernel::arrow::json::ReaderBuilder;
use delta_kernel::arrow::json::reader::{
    ArrayDecoder, DecoderContext, DecoderFactory, Tape, TapeElement,
};
use delta_kernel::parquet::arrow::ArrowWriter;
use delta_kernel::parquet::basic::{Compression, ZstdLevel};
use delta_kernel::parquet::file::properties::WriterProperties;
use serde_json::{Map, Value, json};

pub(super) fn structure<N: Into<String>>(
    fields: impl IntoIterator<Item = (N, DataType)>,
) -> DataType {
    DataType::Struct(
        fields
            .into_iter()
            .map(|(name, dtype)| Arc::new(Field::new(name, dtype, true)))
            .collect(),
    )
}

fn map(value: DataType) -> DataType {
    let entries = DataType::Struct(
        [
            Arc::new(Field::new("key", DataType::Utf8, false)),
            Arc::new(Field::new("value", value, true)),
        ]
        .into(),
    );
    DataType::Map(Arc::new(Field::new("key_value", entries, false)), false)
}

pub(super) fn string_map() -> DataType {
    map(DataType::Utf8)
}

fn string_list() -> DataType {
    DataType::List(Arc::new(Field::new("element", DataType::Utf8, true)))
}

pub(super) fn metadata_type() -> DataType {
    structure([
        ("id", DataType::Utf8),
        (
            "format",
            structure([("provider", DataType::Utf8), ("options", string_map())]),
        ),
        ("schemaString", DataType::Utf8),
        ("partitionColumns", string_list()),
        ("configuration", string_map()),
    ])
}

pub(super) fn protocol_type(features: bool) -> DataType {
    let mut fields = vec![
        ("minReaderVersion", DataType::Int32),
        ("minWriterVersion", DataType::Int32),
    ];
    if features {
        fields.extend([
            ("readerFeatures", string_list()),
            ("writerFeatures", string_list()),
        ]);
    }
    structure(fields)
}

pub(super) fn column(name: &str, dtype: &str) -> Value {
    json!({"name": name, "type": dtype, "nullable": true, "metadata": {}})
}

pub(super) fn metadata(
    id: &str,
    columns: Vec<Value>,
    partitions: &[&str],
    configuration: Value,
) -> Value {
    json!({
        "id": id,
        "format": {"provider": "parquet", "options": {}},
        "schemaString": json!({"type": "struct", "fields": columns}).to_string(),
        "partitionColumns": partitions,
        "configuration": configuration,
    })
}

pub(super) fn protocol() -> Value {
    json!({"minReaderVersion": 1, "minWriterVersion": 2})
}

pub(super) fn write(
    schema: Vec<(&str, DataType)>,
    rows: Vec<Value>,
    row_group_size: Option<usize>,
) -> Vec<u8> {
    let schema = Arc::new(Schema::new(
        schema
            .into_iter()
            .map(|(name, dtype)| Field::new(name, dtype, true))
            .collect::<Vec<_>>(),
    ));
    let mut json = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut json, row).unwrap();
        json.push(b'\n');
    }
    let batch = ReaderBuilder::new(schema.clone())
        .with_decoder_factory(Arc::new(ByteArrayDecoder))
        .with_batch_size(rows.len())
        .build(json.as_slice())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let mut properties =
        WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::default()));
    if let Some(size) = row_group_size {
        properties = properties.set_max_row_group_row_count(Some(size));
    }
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, Some(properties.build())).unwrap();
    writer.write(&batch).unwrap();
    writer.into_inner().unwrap()
}

// Fixture inputs describe binary values as bytes, including deliberately invalid UTF-8.
#[derive(Debug)]
struct ByteArrayDecoder;

impl DecoderFactory for ByteArrayDecoder {
    fn make_default_decoder(
        &self,
        _: &DecoderContext,
        field: &Arc<Field>,
        _: bool,
    ) -> Result<Option<Box<dyn ArrayDecoder>>, ArrowError> {
        Ok((field.data_type() == &DataType::Binary)
            .then(|| Box::new(ByteArrayDecoder) as Box<dyn ArrayDecoder>))
    }
}

impl ArrayDecoder for ByteArrayDecoder {
    fn decode(&mut self, tape: &Tape<'_>, positions: &[u32]) -> Result<ArrayRef, ArrowError> {
        let mut values = Vec::with_capacity(positions.len());
        for &position in positions {
            match tape.get(position) {
                TapeElement::Null => values.push(None),
                TapeElement::StartList(end) => {
                    let bytes = (position + 1..end)
                        .map(|i| match tape.get(i) {
                            TapeElement::Number(index) => {
                                Ok(tape.get_string(index).parse::<u8>().unwrap())
                            },
                            _ => Err(tape.error(i, "byte")),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    values.push(Some(bytes));
                },
                _ => return Err(tape.error(position, "list of bytes")),
            }
        }
        Ok(Arc::new(BinaryArray::from_iter(
            values.iter().map(|v| v.as_deref()),
        )))
    }
}

fn add_fields(partitions: DataType) -> Vec<(&'static str, DataType)> {
    vec![
        ("path", DataType::Utf8),
        ("size", DataType::Int64),
        ("partitionValues", partitions),
    ]
}

fn checkpoint_schema(
    fields: Vec<(&str, DataType)>,
    features: bool,
) -> Vec<(&'static str, DataType)> {
    vec![
        ("add", structure(fields)),
        ("metaData", metadata_type()),
        ("protocol", protocol_type(features)),
    ]
}

fn add(path: impl Display) -> Value {
    json!({"path": format!("{path}.parquet"), "size": 10, "partitionValues": {}})
}

fn x_stats(count: i64, low: i64, high: i64) -> Value {
    json!({"numRecords": count, "minValues": {"x": low}, "maxValues": {"x": high}, "nullCount": {"x": 0}})
}

fn stats_type(values: DataType, counts: DataType, tight: bool) -> DataType {
    let mut fields = vec![
        ("numRecords", DataType::Int64),
        ("minValues", values.clone()),
        ("maxValues", values),
        ("nullCount", counts),
    ];
    if tight {
        fields.push(("tightBounds", DataType::Boolean));
    }
    structure(fields)
}

fn x_stats_type(dtype: DataType) -> DataType {
    stats_type(
        structure([("x", dtype)]),
        structure([("x", DataType::Int64)]),
        false,
    )
}

fn parsed(path: impl Display, count: i64, low: i64, high: i64) -> Value {
    let mut add = add(path);
    add["stats_parsed"] = x_stats(count, low, high);
    json!({"add": add})
}

fn textual(path: impl Display, count: i64, low: i64, high: i64) -> Value {
    let mut add = add(path);
    add["stats"] = json!(x_stats(count, low, high).to_string());
    json!({"add": add})
}

fn parsed_range(paths: &str) -> Vec<Value> {
    paths
        .chars()
        .enumerate()
        .map(|(i, path)| parsed(path, 5, i as i64 * 10, i as i64 * 10 + 5))
        .collect()
}

fn wide_checkpoint() -> Vec<u8> {
    let names: Vec<_> = (0..128).map(|i| format!("c{i}")).collect();
    let mut columns: Vec<_> = names.iter().map(|name| column(name, "long")).collect();
    columns.push(column("p", "binary"));
    let bounds = structure(names.iter().map(|name| (name.as_str(), DataType::Int64)));
    let values = |value| {
        names
            .iter()
            .map(|name| (name.clone(), json!(value)))
            .collect::<Map<_, _>>()
    };
    let mut fields = add_fields(string_map());
    fields.extend([
        (
            "partitionValues_parsed",
            structure([("p", DataType::Binary)]),
        ),
        ("stats_parsed", stats_type(bounds.clone(), bounds, true)),
    ]);
    write(
        checkpoint_schema(fields, false),
        vec![
            json!({"add": {
                "path": "a.parquet", "size": 10, "partitionValues": {"p": "hi"},
                "partitionValues_parsed": {"p": b"hi"},
                "stats_parsed": {
                    "numRecords": 5, "tightBounds": true, "minValues": values(1),
                    "maxValues": values(5), "nullCount": values(0)
                }
            }}),
            json!({"metaData": metadata("wide-checkpoint-regression", columns, &["p"], json!({}))}),
            json!({"protocol": protocol()}),
        ],
        None,
    )
}

fn widened_checkpoint(name: &str) -> Vec<u8> {
    let before = name == "before-date-widening";
    let after = name == "after-date-widening";
    let mut x = column("x", if before { "date" } else { "timestamp_ntz" });
    let mut configuration = json!({});
    if !before {
        let mut change = json!({"fromType": "date", "toType": "timestamp_ntz"});
        if after {
            change["tableVersion"] = json!(1);
        }
        x["metadata"]["delta.typeChanges"] = json!([change]);
    }
    let mut fields = add_fields(string_map());
    let dtype = if after {
        DataType::Timestamp(TimeUnit::Microsecond, None)
    } else {
        DataType::Date32
    };
    fields.push(("stats_parsed", x_stats_type(dtype)));
    let day_scale = if after { 86_400_000_000 } else { 1 };
    // January 1 through January 31, 2020, as dates or rewritten timestamps.
    let mut row = parsed("a", 5, 18_262 * day_scale, 18_292 * day_scale);
    if before || after {
        fields.extend([
            ("modificationTime", DataType::Int64),
            ("dataChange", DataType::Boolean),
        ]);
        row["add"]["modificationTime"] = json!(1000);
        row["add"]["dataChange"] = json!(true);
        configuration["delta.checkpoint.writeStatsAsJson"] = json!("false");
        configuration["delta.checkpoint.writeStatsAsStruct"] = json!("true");
    }
    let protocol = if after {
        configuration["delta.enableTypeWidening"] = json!("true");
        json!({
            "minReaderVersion": 3, "minWriterVersion": 7,
            "readerFeatures": ["timestampNtz", "typeWidening"],
            "writerFeatures": ["appendOnly", "invariants", "timestampNtz", "typeWidening"],
        })
    } else {
        protocol()
    };
    write(
        checkpoint_schema(fields, after),
        vec![
            row,
            json!({"metaData": metadata(name, vec![x], &[], configuration)}),
            json!({"protocol": protocol}),
        ],
        None,
    )
}

fn binary_stats() -> Vec<u8> {
    let mut fields = add_fields(string_map());
    let bounds = structure([("t", DataType::Int64), ("b", DataType::Binary)]);
    let counts = structure([("t", DataType::Int64), ("b", DataType::Int64)]);
    fields.push(("stats_parsed", stats_type(bounds, counts, false)));
    write(
        checkpoint_schema(fields, false),
        vec![
            json!({"add": {
                "path": "a.parquet", "size": 10, "partitionValues": {},
                "stats_parsed": {"numRecords": 5, "minValues": {"t": 1, "b": [0]},
                    "maxValues": {"t": 5, "b": [255]}, "nullCount": {"t": 0, "b": 0}}
            }}),
            json!({"metaData": metadata("binary-stats", vec![column("t", "long"), column("b", "binary")], &[], json!({}))}),
            json!({"protocol": protocol()}),
        ],
        None,
    )
}

fn mixed_root() -> Vec<u8> {
    // Missing format is intentional: the file action sharing the metadata row
    // must be rejected before required metadata fields are interpreted.
    let schema = vec![
        ("protocol", protocol_type(false)),
        (
            "metaData",
            structure([
                ("id", DataType::Utf8),
                ("schemaString", DataType::Utf8),
                ("partitionColumns", string_list()),
            ]),
        ),
        ("add", structure(add_fields(string_map()))),
        ("remove", structure([("path", DataType::Utf8)])),
    ];
    let meta = metadata("id", vec![column("x", "long")], &[], json!({}));
    write(
        schema,
        vec![
            json!({"protocol": protocol()}),
            json!({"add": add("a")}),
            json!({"remove": {"path": "b.parquet"}}),
            json!({"metaData": meta, "add": add("c")}),
        ],
        Some(2),
    )
}

fn long_checkpoint(name: &str) -> Vec<u8> {
    let mut fields = add_fields(string_map());
    fields.extend([
        ("stats", DataType::Utf8),
        ("stats_parsed", x_stats_type(DataType::Int64)),
    ]);
    let schema = checkpoint_schema(fields, false);
    let meta_name = if name == "rg-mixed-protocol-metadata" {
        "rg-meta-only"
    } else {
        name
    };
    let meta = metadata(meta_name, vec![column("x", "long")], &[], json!({}));
    let meta_rows = [json!({"metaData": meta}), json!({"protocol": protocol()})];
    let mut rows = match name {
        "partial-parsed" => {
            // The second file deliberately disagrees with its own JSON stats.
            let mut b = parsed("b", 7, 10, 20);
            b["add"]["stats"] = json!(x_stats(70, 100, 200).to_string());
            vec![textual("a", 5, 1, 5), b, parsed("c", 2, 20, 21)]
        },
        "multipart-1" => vec![parsed("a", 5, 1, 5)],
        "multipart-2" => vec![parsed("b", 7, 10, 20)],
        "rg-spread" => parsed_range("abcdef"),
        "rg-meta-only" => parsed_range("abcd"),
        "rg-stats-text" => {
            let mut adds = parsed_range("abcde");
            adds[2] = textual("c", 5, 20, 25);
            adds
        },
        "rg-multipart-1" => parsed_range("abcde"),
        "rg-multipart-2" => vec![
            textual("f", 5, 50, 55),
            parsed("g", 5, 60, 65),
            parsed("h", 5, 70, 75),
            textual("i", 5, 80, 85),
        ],
        "rg-mixed-protocol-metadata" => {
            vec![json!({"metaData": meta, "protocol": protocol()})]
        },
        _ => panic!("unknown checkpoint fixture: {name}"),
    };
    match name {
        "multipart-1" | "rg-multipart-1" => {},
        "rg-meta-only" | "rg-multipart-2" | "rg-mixed-protocol-metadata" => {
            rows.splice(..0, meta_rows);
        },
        _ => rows.extend(meta_rows),
    }
    write(schema, rows, name.starts_with("rg-").then_some(2))
}

pub(super) fn bytes(name: &str) -> Vec<u8> {
    if name == "v2-sidecar.sidecar.parquet" {
        let mut fields = add_fields(string_map());
        fields.push(("stats", DataType::Utf8));
        return write(
            vec![("add", structure(fields))],
            vec![textual("a", 5, 1, 5), textual("b", 5, 1, 5)],
            None,
        );
    }
    let name = name.strip_suffix(".checkpoint.parquet").unwrap();
    match name {
        "mixed-root" => mixed_root(),
        "wide-binary" => wide_checkpoint(),
        "binary-stats" => binary_stats(),
        "date-widened" | "before-date-widening" | "after-date-widening" => widened_checkpoint(name),
        "binary-partition" | "binary-partition-invalid" => {
            let value: &[u8] = if name == "binary-partition" {
                b"north"
            } else {
                &[u8::MAX]
            };
            let mut row = add("a");
            row["partitionValues"] = json!({"p": value});
            write(
                checkpoint_schema(add_fields(map(DataType::Binary)), false),
                vec![
                    json!({"add": row}),
                    json!({"metaData": metadata(name, vec![column("p", "string")], &["p"], json!({}))}),
                    json!({"protocol": protocol()}),
                ],
                None,
            )
        },
        name => long_checkpoint(name),
    }
}
