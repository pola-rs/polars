//! Checkpoint fixtures for timestamp statistics and stored partition instants.

use std::sync::Arc;

use bytes::Bytes;
use chrono::{TimeZone, Utc};
use delta_kernel::arrow::datatypes::{DataType, TimeUnit};
use delta_kernel::parquet::basic::{LogicalType, Type as PhysicalType};
use delta_kernel::parquet::column::reader::get_typed_column_reader;
use delta_kernel::parquet::column::writer::ColumnCloseResult;
use delta_kernel::parquet::data_type::{Int64Type, Int96, Int96Type};
use delta_kernel::parquet::file::properties::WriterProperties;
use delta_kernel::parquet::file::reader::{FileReader, SerializedFileReader};
use delta_kernel::parquet::file::writer::SerializedFileWriter;
use delta_kernel::parquet::schema::types::{Type, TypePtr};
use serde_json::json;

use super::test_fixtures::{
    column, metadata, metadata_type, protocol, protocol_type, string_map, structure, write,
};

pub(super) fn bytes(name: &str) -> Vec<u8> {
    let recent = timestamp_micros(2020, 1, 20);
    let ancient = timestamp_micros(1500, 1, 20);
    let wall_time = "2020-01-01 12:00:00";
    match name {
        "timestamp-ms.checkpoint.parquet" => statistics_checkpoint(TimeUnit::Millisecond),
        "timestamp-us.checkpoint.parquet" => statistics_checkpoint(TimeUnit::Microsecond),
        "timestamp-ns.checkpoint.parquet" => statistics_checkpoint(TimeUnit::Nanosecond),
        "timestamp-int96.checkpoint.parquet" => {
            with_int96_timestamps(statistics_checkpoint(TimeUnit::Microsecond))
        },
        "partition-timestamp-typed.checkpoint.parquet" => {
            partition_checkpoint(wall_time, Some(recent), PartitionLayout::Single)
        },
        "partition-timestamp-raw.checkpoint.parquet" => {
            partition_checkpoint(wall_time, None, PartitionLayout::Single)
        },
        "partition-timestamp-mapped.checkpoint.parquet" => {
            partition_checkpoint(wall_time, Some(recent), PartitionLayout::Mapped)
        },
        "partition-timestamp-multipart-raw.checkpoint.parquet" => {
            partition_checkpoint("2020-01-01T20:00:00Z", None, PartitionLayout::Single)
        },
        "partition-timestamp-multipart-typed.checkpoint.parquet" => {
            partition_checkpoint(wall_time, Some(recent), PartitionLayout::MultipartLast)
        },
        "partition-timestamp-int96-wall.checkpoint.parquet" => {
            with_int96_timestamps(partition_checkpoint(
                "1500-01-01 12:00:00",
                Some(ancient),
                PartitionLayout::Single,
            ))
        },
        "partition-timestamp-int96-offset.checkpoint.parquet" => {
            with_int96_timestamps(partition_checkpoint(
                "1500-01-01T12:00:00-08:00",
                Some(ancient),
                PartitionLayout::Single,
            ))
        },
        _ => panic!("unknown timestamp checkpoint fixture {name}"),
    }
}

fn timestamp_micros(year: i32, day: u32, hour: u32) -> i64 {
    Utc.with_ymd_and_hms(year, 1, day, hour, 0, 0)
        .unwrap()
        .timestamp_micros()
}

fn statistics_checkpoint(unit: TimeUnit) -> Vec<u8> {
    let timestamp = |year, day| {
        let micros = timestamp_micros(year, day, 0);
        match unit {
            TimeUnit::Millisecond => micros / 1_000,
            TimeUnit::Microsecond => micros,
            TimeUnit::Nanosecond => micros.checked_mul(1_000).unwrap(),
            TimeUnit::Second => unreachable!(),
        }
    };
    let values = structure([("t", DataType::Timestamp(unit, Some("UTC".into())))]);
    let stats = structure([
        ("numRecords", DataType::Int64),
        ("tightBounds", DataType::Boolean),
        ("minValues", values.clone()),
        ("maxValues", values),
        ("nullCount", structure([("t", DataType::Int64)])),
    ]);
    let add = structure([
        ("path", DataType::Utf8),
        ("size", DataType::Int64),
        ("partitionValues", string_map()),
        ("stats_parsed", stats),
    ]);
    let row = |path, year| {
        json!({"add": {
            "path": path, "size": 10, "partitionValues": {},
            "stats_parsed": {
                "numRecords": 5, "tightBounds": true,
                "minValues": {"t": timestamp(year, 1)},
                "maxValues": {"t": timestamp(year, 2)},
                "nullCount": {"t": 0},
            },
        }})
    };
    let mut rows = vec![row("a.parquet", 2024)];
    // Year 1500 falls outside the signed nanosecond range.
    if unit != TimeUnit::Nanosecond {
        rows.push(row("b.parquet", 1500));
    }
    rows.extend([
        json!({"metaData": metadata("timestamp-checkpoint", vec![column("t", "timestamp")], &[], json!({}))}),
        json!({"protocol": protocol()}),
    ]);
    write(
        vec![
            ("add", add),
            ("metaData", metadata_type()),
            ("protocol", protocol_type(false)),
        ],
        rows,
        None,
    )
}

#[derive(Clone, Copy)]
enum PartitionLayout {
    Single,
    Mapped,
    MultipartLast,
}

fn partition_checkpoint(raw: &str, stored_micros: Option<i64>, layout: PartitionLayout) -> Vec<u8> {
    let mapped = matches!(layout, PartitionLayout::Mapped);
    let multipart_last = matches!(layout, PartitionLayout::MultipartLast);
    let typed = stored_micros.is_some();
    let part = if mapped { "col-t" } else { "t" };
    let mut add_fields = vec![
        ("path", DataType::Utf8),
        ("partitionValues", string_map()),
        ("size", DataType::Int64),
        ("modificationTime", DataType::Int64),
        ("dataChange", DataType::Boolean),
        ("stats", DataType::Utf8),
    ];
    if typed {
        add_fields.extend([
            (
                "partitionValues_parsed",
                structure([(
                    part,
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                )]),
            ),
            ("stats_parsed", structure([("numRecords", DataType::Int64)])),
        ]);
    }

    let row = |path: &str, raw: Option<&str>, micros: Option<i64>| {
        let stats = json!({"numRecords": 1});
        let mut add = json!({
            "path": path, "partitionValues": {(part): raw}, "size": 512,
            "modificationTime": 0, "dataChange": false, "stats": stats.to_string(),
        });
        if typed {
            add["partitionValues_parsed"] = json!({(part): micros});
            add["stats_parsed"] = stats;
        }
        json!({"add": add})
    };
    let mut schema = Vec::new();
    let mut rows = Vec::new();
    if !multipart_last {
        let mut columns = vec![column("x", "long"), column("t", "timestamp")];
        let mut configuration = json!({"delta.checkpoint.writeStatsAsStruct": typed.to_string()});
        let mut versions = protocol();
        if mapped {
            versions = json!({"minReaderVersion": 2, "minWriterVersion": 5});
            configuration["delta.columnMapping.mode"] = json!("name");
            configuration["delta.columnMapping.maxColumnId"] = json!("2");
            for (index, (column, physical)) in
                columns.iter_mut().zip(["col-x", "col-t"]).enumerate()
            {
                column["metadata"] = json!({
                    "delta.columnMapping.id": index + 1,
                    "delta.columnMapping.physicalName": physical,
                });
            }
        }
        schema.extend([
            ("protocol", protocol_type(false)),
            ("metaData", metadata_type()),
        ]);
        rows.push(json!({"protocol": versions}));
        rows.push(json!({"metaData": metadata(
            "314f71cb-07ee-45be-b2dc-6315cc83abd4", columns, &["t"], configuration,
        )}));
    }
    schema.push(("add", structure(add_fields)));
    let path = if multipart_last {
        "b.parquet"
    } else {
        "a.parquet"
    };
    rows.push(row(path, Some(raw), stored_micros));
    if mapped {
        rows.push(row("b.parquet", None, None));
    }
    write(schema, rows, None)
}

fn int96_schema(field: &TypePtr) -> TypePtr {
    match field.as_ref() {
        Type::GroupType { basic_info, fields } => Arc::new(Type::GroupType {
            basic_info: basic_info.clone(),
            fields: fields.iter().map(int96_schema).collect(),
        }),
        Type::PrimitiveType { basic_info, .. }
            if matches!(
                basic_info.logical_type_ref(),
                Some(LogicalType::Timestamp(_))
            ) =>
        {
            Arc::new(
                Type::primitive_type_builder(field.name(), PhysicalType::INT96)
                    .with_repetition(basic_info.repetition())
                    .with_id(basic_info.has_id().then(|| basic_info.id()))
                    .build()
                    .unwrap(),
            )
        },
        _ => field.clone(),
    }
}

fn int96_micros(micros: i64) -> Int96 {
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    const JULIAN_DAY_OF_EPOCH: i64 = 2_440_588;
    let julian_day = micros.div_euclid(MICROS_PER_DAY) + JULIAN_DAY_OF_EPOCH;
    let nanos_of_day = micros.rem_euclid(MICROS_PER_DAY) as u64 * 1_000;
    let mut timestamp = Int96::new();
    timestamp.set_data(
        nanos_of_day as u32,
        (nanos_of_day >> 32) as u32,
        julian_day.try_into().unwrap(),
    );
    timestamp
}

/// ArrowWriter emits TIMESTAMP(MICROS), so use the Parquet column writer for the
/// legacy INT96 leaves. Copy the other chunks and preserve the null nesting levels.
fn with_int96_timestamps(bytes: Vec<u8>) -> Vec<u8> {
    let source = Bytes::from(bytes);
    let reader = SerializedFileReader::new(source.clone()).unwrap();
    assert_eq!(reader.num_row_groups(), 1);
    let schema = reader.metadata().file_metadata().schema_descr();
    let mut writer = SerializedFileWriter::new(
        Vec::new(),
        int96_schema(&schema.root_schema_ptr()),
        Arc::new(WriterProperties::default()),
    )
    .unwrap();
    let input = reader.get_row_group(0).unwrap();
    let rows = usize::try_from(input.metadata().num_rows()).unwrap();
    let mut output = writer.next_row_group().unwrap();
    for (index, column) in input.metadata().columns().iter().enumerate() {
        if matches!(
            column.column_descr().logical_type_ref(),
            Some(LogicalType::Timestamp(_))
        ) {
            let mut values = Vec::new();
            let mut definitions = Vec::new();
            let mut input =
                get_typed_column_reader::<Int64Type>(input.get_column_reader(index).unwrap());
            let (read, _, _) = input
                .read_records(rows, Some(&mut definitions), None, &mut values)
                .unwrap();
            assert_eq!(read, rows);
            let values: Vec<_> = values.into_iter().map(int96_micros).collect();
            let mut output = output.next_column().unwrap().unwrap();
            output
                .typed::<Int96Type>()
                .write_batch(&values, Some(&definitions), None)
                .unwrap();
            output.close().unwrap();
        } else {
            output
                .append_column(
                    &source,
                    ColumnCloseResult {
                        bytes_written: column.compressed_size().try_into().unwrap(),
                        rows_written: rows.try_into().unwrap(),
                        metadata: column.clone(),
                        bloom_filter: None,
                        column_index: None,
                        offset_index: None,
                    },
                )
                .unwrap();
        }
    }
    output.close().unwrap();
    writer.into_inner().unwrap()
}
