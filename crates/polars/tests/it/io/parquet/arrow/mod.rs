mod read;
mod write;

use std::io::{Cursor, Read, Seek};
use std::sync::Arc;

use ethnum::AsI256;
use polars::prelude::{PlSmallStr, get_encodings};
use polars_arrow::array::*;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::datatypes::*;
use polars_arrow::record_batch::RecordBatchT;
use polars_arrow::types::{NativeType, i256};
use polars_error::PolarsResult;
use polars_parquet::read::expr::{ParquetColumnExpr, ParquetScalar, SpecializedParquetColumnExpr};
use polars_parquet::read::{self as p_read};
use polars_parquet::write::*;

use super::read::file::FileReader;
use super::read::row_group::read_columns_many;

fn new_struct(
    arrays: Vec<Box<dyn Array>>,
    length: usize,
    names: Vec<String>,
    validity: Option<Bitmap>,
) -> StructArray {
    let fields = names
        .into_iter()
        .zip(arrays.iter())
        .map(|(n, a)| Field::new(n.into(), a.dtype().clone(), true))
        .collect();
    StructArray::new(ArrowDataType::Struct(fields), length, arrays, validity)
}

pub fn read_column<R: Read + Seek>(mut reader: R, column: &str) -> PolarsResult<Box<dyn Array>> {
    let metadata = p_read::read_metadata(&mut reader)?;
    let mut schema = p_read::infer_schema(&metadata)?;

    schema.retain(|_, f| f.name == column);

    let mut reader = FileReader::new(reader, metadata.row_groups, schema, None);

    let array = reader.next().unwrap()?.into_arrays().pop().unwrap();

    Ok(array)
}

pub fn pyarrow_nested_edge(column: &str) -> Box<dyn Array> {
    match column {
        "simple" => {
            // [[0, 1]]
            let data = [Some(vec![Some(0), Some(1)])];
            let mut a = MutableListArray::<i64, MutablePrimitiveArray<i64>>::new();
            a.try_extend(data).unwrap();
            let array: ListArray<i64> = a.into();
            Box::new(array)
        },
        "null" => {
            // [None]
            let data = [None::<Vec<Option<i64>>>];
            let mut a = MutableListArray::<i64, MutablePrimitiveArray<i64>>::new();
            a.try_extend(data).unwrap();
            let array: ListArray<i64> = a.into();
            Box::new(array)
        },
        "empty" => {
            // [None]
            let data: [Option<Vec<Option<i64>>>; 0] = [];
            let mut a = MutableListArray::<i64, MutablePrimitiveArray<i64>>::new();
            a.try_extend(data).unwrap();
            let array: ListArray<i64> = a.into();
            Box::new(array)
        },
        "struct_list_nullable" => {
            // [
            //      {"f1": ["a", "b", None, "c"]}
            // ]
            let a = ListArray::<i64>::new(
                ArrowDataType::LargeList(Box::new(Field::new(
                    "item".into(),
                    ArrowDataType::Utf8View,
                    true,
                ))),
                vec![0, 4].try_into().unwrap(),
                Utf8ViewArray::from_slice([Some("a"), Some("b"), None, Some("c")]).boxed(),
                None,
            );
            StructArray::new(
                ArrowDataType::Struct(vec![Field::new("f1".into(), a.dtype().clone(), true)]),
                a.len(),
                vec![a.boxed()],
                None,
            )
            .boxed()
        },
        "list_struct_list_nullable" => {
            let values = pyarrow_nested_edge("struct_list_nullable");
            ListArray::<i64>::new(
                ArrowDataType::LargeList(Box::new(Field::new(
                    "item".into(),
                    values.dtype().clone(),
                    true,
                ))),
                vec![0, 1].try_into().unwrap(),
                values,
                None,
            )
            .boxed()
        },
        _ => todo!(),
    }
}

pub fn pyarrow_nested_nullable(column: &str) -> Box<dyn Array> {
    let i64_values = &[
        Some(0),
        Some(1),
        Some(2),
        None,
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(8),
        Some(9),
        Some(10),
    ];
    let offsets = vec![0, 2, 2, 5, 8, 8, 11, 11, 12].try_into().unwrap();

    let values = match column {
        "list_int64" => {
            // [[0, 1], None, [2, None, 3], [4, 5, 6], [], [7, 8, 9], None, [10]]
            PrimitiveArray::<i64>::from(i64_values).boxed()
        },
        "list_int64_required" | "list_int64_optional_required" | "list_int64_required_required" => {
            // [[0, 1], None, [2, 0, 3], [4, 5, 6], [], [7, 8, 9], None, [10]]
            PrimitiveArray::<i64>::from(&[
                Some(0),
                Some(1),
                Some(2),
                Some(0),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(8),
                Some(9),
                Some(10),
            ])
            .boxed()
        },
        "list_int16" => PrimitiveArray::<i16>::from(&[
            Some(0),
            Some(1),
            Some(2),
            None,
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            Some(10),
        ])
        .boxed(),
        "list_bool" => BooleanArray::from(&[
            Some(false),
            Some(true),
            Some(true),
            None,
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ])
        .boxed(),
        /*
            string = [
                ["Hello", "bbb"],
                None,
                ["aa", None, ""],
                ["bbb", "aa", "ccc"],
                [],
                ["abc", "bbb", "bbb"],
                None,
                [""],
            ]
        */
        "list_utf8" => Utf8ViewArray::from_slice([
            Some("Hello".to_string()),
            Some("bbb".to_string()),
            Some("aa".to_string()),
            None,
            Some("".to_string()),
            Some("bbb".to_string()),
            Some("aa".to_string()),
            Some("ccc".to_string()),
            Some("abc".to_string()),
            Some("bbb".to_string()),
            Some("bbb".to_string()),
            Some("".to_string()),
        ])
        .boxed(),
        "list_large_binary" => Box::new(BinaryArray::<i64>::from([
            Some(b"Hello".to_vec()),
            Some(b"bbb".to_vec()),
            Some(b"aa".to_vec()),
            None,
            Some(b"".to_vec()),
            Some(b"bbb".to_vec()),
            Some(b"aa".to_vec()),
            Some(b"ccc".to_vec()),
            Some(b"abc".to_vec()),
            Some(b"bbb".to_vec()),
            Some(b"bbb".to_vec()),
            Some(b"".to_vec()),
        ])),
        "list_decimal" => {
            let values = i64_values
                .iter()
                .map(|x| x.map(|x| x as i128))
                .collect::<Vec<_>>();
            Box::new(PrimitiveArray::<i128>::from(values).to(ArrowDataType::Decimal(9, 0)))
        },
        "list_decimal256" => {
            let values = i64_values
                .iter()
                .map(|x| x.map(|x| i256(x.as_i256())))
                .collect::<Vec<_>>();
            let array = PrimitiveArray::<i256>::from(values).to(ArrowDataType::Decimal256(9, 0));
            Box::new(array)
        },
        "list_nested_i64"
        | "list_nested_inner_required_i64"
        | "list_nested_inner_required_required_i64" => {
            Box::new(NullArray::new(ArrowDataType::Null, 1))
        },
        "struct_list_nullable" => pyarrow_nested_nullable("list_utf8"),
        "list_struct_nullable" => {
            let array = Utf8ViewArray::from_slice([
                Some("a"),
                Some("b"),
                //
                Some("b"),
                None,
                Some("b"),
                //
                None,
                None,
                None,
                //
                Some("d"),
                Some("d"),
                Some("d"),
                //
                Some("e"),
            ])
            .boxed();

            let len = array.len();
            new_struct(
                vec![array],
                len,
                vec!["a".to_string()],
                Some(
                    [
                        true, true, //
                        true, false, true, //
                        true, true, true, //
                        true, true, true, //
                        true,
                    ]
                    .into(),
                ),
            )
            .boxed()
        },
        "list_struct_list_nullable" => {
            /*
            [
                [{"a": ["a"]}, {"a": ["b"]}],
                None,
                [{"a": ["b"]}, None, {"a": ["b"]}],
                [{"a": None}, {"a": None}, {"a": None}],
                [],
                [{"a": ["d"]}, {"a": [None]}, {"a": ["c", "d"]}],
                None,
                [{"a": []}],
            ]
            */
            let array = Utf8ViewArray::from_slice([
                Some("a"),
                Some("b"),
                //
                Some("b"),
                Some("b"),
                //
                Some("d"),
                None,
                Some("c"),
                Some("d"),
            ])
            .boxed();

            let array = ListArray::<i64>::new(
                ArrowDataType::LargeList(Box::new(Field::new(
                    "item".into(),
                    array.dtype().clone(),
                    true,
                ))),
                vec![0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 6, 8, 8]
                    .try_into()
                    .unwrap(),
                array,
                Some(
                    [
                        true, true, true, false, true, false, false, false, true, true, true, true,
                    ]
                    .into(),
                ),
            )
            .boxed();

            let len = array.len();
            new_struct(
                vec![array],
                len,
                vec!["a".to_string()],
                Some(
                    [
                        true, true, //
                        true, false, true, //
                        true, true, true, //
                        true, true, true, //
                        true,
                    ]
                    .into(),
                ),
            )
            .boxed()
        },
        other => unreachable!("{}", other),
    };

    match column {
        "list_int64_required_required" => {
            // [[0, 1], [], [2, 0, 3], [4, 5, 6], [], [7, 8, 9], [], [10]]
            let dtype = ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Int64,
                false,
            )));
            ListArray::<i64>::new(dtype, offsets, values, None).boxed()
        },
        "list_int64_optional_required" => {
            // [[0, 1], [], [2, 0, 3], [4, 5, 6], [], [7, 8, 9], [], [10]]
            let dtype = ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Int64,
                true,
            )));
            ListArray::<i64>::new(dtype, offsets, values, None).boxed()
        },
        "list_nested_i64" => {
            // [[0, 1]], None, [[2, None], [3]], [[4, 5], [6]], [], [[7], None, [9]], [[], [None], None], [[10]]
            let data = [
                Some(vec![Some(vec![Some(0), Some(1)])]),
                None,
                Some(vec![Some(vec![Some(2), None]), Some(vec![Some(3)])]),
                Some(vec![Some(vec![Some(4), Some(5)]), Some(vec![Some(6)])]),
                Some(vec![]),
                Some(vec![Some(vec![Some(7)]), None, Some(vec![Some(9)])]),
                Some(vec![Some(vec![]), Some(vec![None]), None]),
                Some(vec![Some(vec![Some(10)])]),
            ];
            let mut a =
                MutableListArray::<i64, MutableListArray<i64, MutablePrimitiveArray<i64>>>::new();
            a.try_extend(data).unwrap();
            let array: ListArray<i64> = a.into();
            Box::new(array)
        },
        "list_nested_inner_required_i64" => {
            let data = [
                Some(vec![Some(vec![Some(0), Some(1)])]),
                None,
                Some(vec![Some(vec![Some(2), Some(3)]), Some(vec![Some(3)])]),
                Some(vec![Some(vec![Some(4), Some(5)]), Some(vec![Some(6)])]),
                Some(vec![]),
                Some(vec![Some(vec![Some(7)]), None, Some(vec![Some(9)])]),
                None,
                Some(vec![Some(vec![Some(10)])]),
            ];
            let mut a =
                MutableListArray::<i64, MutableListArray<i64, MutablePrimitiveArray<i64>>>::new();
            a.try_extend(data).unwrap();
            let array: ListArray<i64> = a.into();
            Box::new(array)
        },
        "list_nested_inner_required_required_i64" => {
            let data = [
                Some(vec![Some(vec![Some(0), Some(1)])]),
                None,
                Some(vec![Some(vec![Some(2), Some(3)]), Some(vec![Some(3)])]),
                Some(vec![Some(vec![Some(4), Some(5)]), Some(vec![Some(6)])]),
                Some(vec![]),
                Some(vec![
                    Some(vec![Some(7)]),
                    Some(vec![Some(8)]),
                    Some(vec![Some(9)]),
                ]),
                None,
                Some(vec![Some(vec![Some(10)])]),
            ];
            let mut a =
                MutableListArray::<i64, MutableListArray<i64, MutablePrimitiveArray<i64>>>::new();
            a.try_extend(data).unwrap();
            let array: ListArray<i64> = a.into();
            Box::new(array)
        },
        "struct_list_nullable" => {
            let len = values.len();
            new_struct(vec![values], len, vec!["a".to_string()], None).boxed()
        },
        _ => {
            let field = match column {
                "list_int64" => Field::new("item".into(), ArrowDataType::Int64, true),
                "list_int64_required" => Field::new("item".into(), ArrowDataType::Int64, false),
                "list_int16" => Field::new("item".into(), ArrowDataType::Int16, true),
                "list_bool" => Field::new("item".into(), ArrowDataType::Boolean, true),
                "list_utf8" => Field::new("item".into(), ArrowDataType::Utf8View, true),
                "list_large_binary" => Field::new("item".into(), ArrowDataType::LargeBinary, true),
                "list_decimal" => Field::new("item".into(), ArrowDataType::Decimal(9, 0), true),
                "list_decimal256" => {
                    Field::new("item".into(), ArrowDataType::Decimal256(9, 0), true)
                },
                "list_struct_nullable" => Field::new("item".into(), values.dtype().clone(), true),
                "list_struct_list_nullable" => {
                    Field::new("item".into(), values.dtype().clone(), true)
                },
                other => unreachable!("{}", other),
            };

            let validity = Some(Bitmap::from([
                true, false, true, true, true, true, false, true,
            ]));
            // [0, 2, 2, 5, 8, 8, 11, 11, 12]
            // [[a1, a2], None, [a3, a4, a5], [a6, a7, a8], [], [a9, a10, a11], None, [a12]]
            let dtype = ArrowDataType::LargeList(Box::new(field));
            ListArray::<i64>::new(dtype, offsets, values, validity).boxed()
        },
    }
}

pub fn pyarrow_nullable(column: &str) -> Box<dyn Array> {
    let i64_values = &[
        Some(-256),
        Some(-1),
        None,
        Some(3),
        None,
        Some(5),
        Some(6),
        Some(7),
        None,
        Some(9),
    ];
    let u32_values = &[
        Some(0),
        Some(1),
        None,
        Some(3),
        None,
        Some(5),
        Some(6),
        Some(7),
        None,
        Some(9),
    ];

    match column {
        "int64" => Box::new(PrimitiveArray::<i64>::from(i64_values)),
        "float64" => Box::new(PrimitiveArray::<f64>::from(&[
            Some(0.0),
            Some(1.0),
            None,
            Some(3.0),
            None,
            Some(5.0),
            Some(6.0),
            Some(7.0),
            None,
            Some(9.0),
        ])),
        "string" => Box::new(Utf8ViewArray::from_slice([
            Some("Hello".to_string()),
            None,
            Some("aa".to_string()),
            Some("".to_string()),
            None,
            Some("abc".to_string()),
            None,
            None,
            Some("def".to_string()),
            Some("aaa".to_string()),
        ])),
        "bool" => Box::new(BooleanArray::from([
            Some(true),
            None,
            Some(false),
            Some(false),
            None,
            Some(true),
            None,
            None,
            Some(true),
            Some(true),
        ])),
        "timestamp_ms" => Box::new(
            PrimitiveArray::<i64>::from_iter(u32_values.iter().map(|x| x.map(|x| x as i64)))
                .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None)),
        ),
        "uint32" => Box::new(PrimitiveArray::<u32>::from(u32_values)),
        "int32_dict" => {
            let keys = PrimitiveArray::<i32>::from([Some(0), Some(1), None, Some(1)]);
            let values = Box::new(PrimitiveArray::<i32>::from_slice([10, 200]));
            Box::new(DictionaryArray::try_from_keys(keys, values, false).unwrap())
        },
        "timestamp_us" => Box::new(
            PrimitiveArray::<i64>::from(i64_values)
                .to(ArrowDataType::Timestamp(TimeUnit::Microsecond, None)),
        ),
        "timestamp_s" => Box::new(
            PrimitiveArray::<i64>::from(i64_values)
                .to(ArrowDataType::Timestamp(TimeUnit::Second, None)),
        ),
        "timestamp_s_utc" => Box::new(PrimitiveArray::<i64>::from(i64_values).to(
            ArrowDataType::Timestamp(TimeUnit::Second, Some(PlSmallStr::from_static("UTC"))),
        )),
        _ => unreachable!(),
    }
}

// these values match the values in `integration`
pub fn pyarrow_required(column: &str) -> Box<dyn Array> {
    let i64_values = &[
        Some(-256),
        Some(-1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(8),
        Some(9),
    ];

    match column {
        "int64" => Box::new(PrimitiveArray::<i64>::from(i64_values)),
        "bool" => Box::new(BooleanArray::from_slice([
            true, true, false, false, false, true, true, true, true, true,
        ])),
        "string" => Box::new(Utf8ViewArray::from_slice([
            Some("Hello"),
            Some("bbb"),
            Some("aa"),
            Some(""),
            Some("bbb"),
            Some("abc"),
            Some("bbb"),
            Some("bbb"),
            Some("def"),
            Some("aaa"),
        ])),
        _ => unreachable!(),
    }
}

pub fn pyarrow_struct(column: &str) -> Box<dyn Array> {
    let boolean = [
        Some(true),
        None,
        Some(false),
        Some(false),
        None,
        Some(true),
        None,
        None,
        Some(true),
        Some(true),
    ];
    let boolean = BooleanArray::from(boolean).boxed();

    let string = [
        Some("Hello"),
        None,
        Some("aa"),
        Some(""),
        None,
        Some("abc"),
        None,
        None,
        Some("def"),
        Some("aaa"),
    ];
    let string = Utf8ViewArray::from_slice(string).boxed();

    let mask = [true, true, false, true, true, true, true, true, true, true];

    let fields = vec![
        Field::new("f1".into(), ArrowDataType::Utf8View, true),
        Field::new("f2".into(), ArrowDataType::Boolean, true),
    ];
    match column {
        "struct" => StructArray::new(
            ArrowDataType::Struct(fields),
            string.len(),
            vec![string, boolean],
            None,
        )
        .boxed(),
        "struct_nullable" => {
            let len = string.len();
            let values = vec![string, boolean];
            StructArray::new(
                ArrowDataType::Struct(fields),
                len,
                values,
                Some(mask.into()),
            )
            .boxed()
        },
        "struct_struct" => {
            let struct_ = pyarrow_struct("struct");
            Box::new(StructArray::new(
                ArrowDataType::Struct(vec![
                    Field::new("f1".into(), ArrowDataType::Struct(fields), true),
                    Field::new("f2".into(), ArrowDataType::Boolean, true),
                ]),
                struct_.len(),
                vec![struct_, boolean],
                None,
            ))
        },
        "struct_struct_nullable" => {
            let struct_ = pyarrow_struct("struct");
            Box::new(StructArray::new(
                ArrowDataType::Struct(vec![
                    Field::new("f1".into(), ArrowDataType::Struct(fields), true),
                    Field::new("f2".into(), ArrowDataType::Boolean, true),
                ]),
                struct_.len(),
                vec![struct_, boolean],
                Some(mask.into()),
            ))
        },
        _ => todo!(),
    }
}

fn integration_write(
    schema: &ArrowSchema,
    chunks: &[RecordBatchT<Box<dyn Array>>],
) -> PolarsResult<Vec<u8>> {
    let options = WriteOptions {
        statistics: StatisticsOptions::full(),
        compression: CompressionOptions::Uncompressed,
        version: Version::V1,
        data_page_size: None,
    };

    let encodings = get_encodings(schema);

    let row_groups =
        RowGroupIterator::try_new(chunks.iter().cloned().map(Ok), schema, options, encodings)?;

    let writer = Cursor::new(vec![]);

    let mut writer = FileWriter::try_new(writer, schema.clone(), options, None)?;

    for group in row_groups {
        writer.write(u64::MAX, group?)?;
    }
    writer.end(None)?;

    Ok(writer.into_inner().into_inner())
}

type IntegrationRead = (ArrowSchema, Vec<RecordBatchT<Box<dyn Array>>>);

fn integration_read(data: &[u8], limit: Option<usize>) -> PolarsResult<IntegrationRead> {
    let metadata = p_read::read_metadata(&mut Cursor::new(data))?;
    let schema = p_read::infer_schema(&metadata)?;
    let batches = read_with_schema(data, schema.clone(), limit)?;

    Ok((schema, batches))
}

/// Decodes `data` as `schema`, instead of the schema inferred from the file.
fn read_with_schema(
    data: &[u8],
    schema: ArrowSchema,
    limit: Option<usize>,
) -> PolarsResult<Vec<RecordBatchT<Box<dyn Array>>>> {
    let metadata = p_read::read_metadata(&mut Cursor::new(data))?;
    FileReader::new(Cursor::new(data), metadata.row_groups, schema, limit).collect()
}

fn assert_roundtrip(
    schema: ArrowSchema,
    chunk: RecordBatchT<Box<dyn Array>>,
    limit: Option<usize>,
) -> PolarsResult<()> {
    let r = integration_write(&schema, std::slice::from_ref(&chunk))?;

    let (new_schema, new_chunks) = integration_read(&r, limit)?;

    let expected = if let Some(limit) = limit {
        let length = chunk.len().min(limit);
        let expected = chunk
            .into_arrays()
            .into_iter()
            .map(|x| x.sliced(0, limit))
            .collect::<Vec<_>>();
        RecordBatchT::new(length, Arc::new(schema.clone()), expected)
    } else {
        chunk
    };

    assert_eq!(new_schema, schema);
    assert_eq!(new_chunks, vec![expected]);
    Ok(())
}

fn data<T: NativeType, I: Iterator<Item = T>>(
    mut iter: I,
    inner_is_nullable: bool,
) -> Box<dyn Array> {
    // [[0, 1], [], [2, 0, 3], [4, 5, 6], [], [7, 8, 9], [], [10]]
    let data = vec![
        Some(vec![Some(iter.next().unwrap()), Some(iter.next().unwrap())]),
        Some(vec![]),
        Some(vec![
            Some(iter.next().unwrap()),
            Some(iter.next().unwrap()),
            Some(iter.next().unwrap()),
        ]),
        Some(vec![
            Some(iter.next().unwrap()),
            Some(iter.next().unwrap()),
            Some(iter.next().unwrap()),
        ]),
        Some(vec![]),
        Some(vec![
            Some(iter.next().unwrap()),
            Some(iter.next().unwrap()),
            Some(iter.next().unwrap()),
        ]),
        Some(vec![]),
        Some(vec![Some(iter.next().unwrap())]),
    ];
    let mut array = MutableListArray::<i64, _>::new_with_field(
        MutablePrimitiveArray::<T>::new(),
        "item".into(),
        inner_is_nullable,
    );
    array.try_extend(data).unwrap();
    array.into_box()
}

fn assert_array_roundtrip(
    is_nullable: bool,
    array: Box<dyn Array>,
    limit: Option<usize>,
) -> PolarsResult<()> {
    let schema =
        ArrowSchema::from_iter([Field::new("a1".into(), array.dtype().clone(), is_nullable)]);
    let chunk = RecordBatchT::try_new(array.len(), Arc::new(schema.clone()), vec![array])?;

    assert_roundtrip(schema, chunk, limit)
}

fn test_list_array_required_required(limit: Option<usize>) -> PolarsResult<()> {
    assert_array_roundtrip(false, data(0..12i8, false), limit)?;
    assert_array_roundtrip(false, data(0..12i16, false), limit)?;
    assert_array_roundtrip(false, data(0..12i64, false), limit)?;
    assert_array_roundtrip(false, data(0..12i64, false), limit)?;
    assert_array_roundtrip(false, data(0..12u8, false), limit)?;
    assert_array_roundtrip(false, data(0..12u16, false), limit)?;
    assert_array_roundtrip(false, data(0..12u32, false), limit)?;
    assert_array_roundtrip(false, data(0..12u64, false), limit)?;
    assert_array_roundtrip(false, data((0..12).map(|x| (x as f32) * 1.0), false), limit)?;
    assert_array_roundtrip(
        false,
        data((0..12).map(|x| (x as f64) * 1.0f64), false),
        limit,
    )
}

#[test]
fn list_array_required_required() -> PolarsResult<()> {
    test_list_array_required_required(None)
}

#[test]
fn list_array_optional_optional() -> PolarsResult<()> {
    assert_array_roundtrip(true, data(0..12, true), None)
}

#[test]
fn list_array_required_optional() -> PolarsResult<()> {
    assert_array_roundtrip(true, data(0..12, false), None)
}

#[test]
fn list_array_optional_required() -> PolarsResult<()> {
    assert_array_roundtrip(false, data(0..12, true), None)
}

#[test]
fn list_slice() -> PolarsResult<()> {
    let data = vec![
        Some(vec![None, Some(2)]),
        Some(vec![Some(3), Some(4)]),
        Some(vec![Some(5), Some(6)]),
    ];
    let mut array = MutableListArray::<i64, _>::new_with_field(
        MutablePrimitiveArray::<i64>::new(),
        "item".into(),
        true,
    );
    array.try_extend(data).unwrap();
    let a: ListArray<i64> = array.into();
    let a = a.sliced(2, 1);
    assert_array_roundtrip(false, a.boxed(), None)
}

#[test]
fn struct_slice() -> PolarsResult<()> {
    let a = pyarrow_nested_nullable("struct_list_nullable");

    let a = a.sliced(2, 1);
    assert_array_roundtrip(true, a, None)
}

#[test]
fn list_struct_slice() -> PolarsResult<()> {
    let a = pyarrow_nested_nullable("list_struct_nullable");

    let a = a.sliced(2, 1);
    assert_array_roundtrip(true, a, None)
}

#[test]
fn list_int_nullable() -> PolarsResult<()> {
    let data = vec![
        Some(vec![Some(1)]),
        None,
        Some(vec![None, Some(2)]),
        Some(vec![]),
        Some(vec![Some(3)]),
        None,
    ];
    let mut array = MutableListArray::<i64, _>::new_with_field(
        MutablePrimitiveArray::<i64>::new(),
        "item".into(),
        true,
    );
    array.try_extend(data).unwrap();
    assert_array_roundtrip(true, array.into_box(), None)
}

#[test]
fn list_nesting_mismatch_is_refused() -> PolarsResult<()> {
    let array = data(0..12i64, true);
    let schema = ArrowSchema::from_iter([Field::new("a1".into(), array.dtype().clone(), true)]);
    let chunk = RecordBatchT::try_new(array.len(), Arc::new(schema.clone()), vec![array])?;
    let data = integration_write(&schema, &[chunk])?;

    let mismatched = ArrowSchema::from_iter([Field::new(
        "a1".into(),
        ArrowDataType::LargeList(Box::new(Field::new(
            "item".into(),
            ArrowDataType::Int64,
            false,
        ))),
        true,
    )]);
    let err = read_with_schema(&data, mismatched, None).unwrap_err();
    assert!(
        err.to_string().contains("nesting mismatch"),
        "unexpected error: {err}"
    );

    Ok(())
}

#[test]
fn limit_list() -> PolarsResult<()> {
    test_list_array_required_required(Some(2))
}

#[test]
fn filter_chunk() -> PolarsResult<()> {
    let field = Field::new("c1".into(), ArrowDataType::Int16, true);
    let schema = ArrowSchema::from_iter([field]);
    let chunk1 = RecordBatchT::new(
        2,
        Arc::new(schema.clone()),
        vec![PrimitiveArray::from_slice([1i16, 3]).boxed()],
    );
    let chunk2 = RecordBatchT::new(
        2,
        Arc::new(schema.clone()),
        vec![PrimitiveArray::from_slice([2i16, 4]).boxed()],
    );

    let r = integration_write(&schema, &[chunk1.clone(), chunk2.clone()])?;

    let mut reader = Cursor::new(r);

    let metadata = p_read::read_metadata(&mut reader)?;

    let new_schema = p_read::infer_schema(&metadata)?;
    assert_eq!(new_schema, schema);

    // select chunk 1
    let row_groups = metadata
        .row_groups
        .into_iter()
        .enumerate()
        .filter(|(index, _)| *index == 0)
        .map(|(_, row_group)| row_group)
        .collect();

    let reader = FileReader::new(reader, row_groups, schema, None);

    let new_chunks = reader.collect::<PolarsResult<Vec<_>>>()?;

    assert_eq!(new_chunks, vec![chunk1]);
    Ok(())
}

/// Writes an array as a row group and reads it back with `filter`.
fn filtered_roundtrip(
    array: Box<dyn Array>,
    is_nullable: bool,
    filter: p_read::Filter,
) -> PolarsResult<Box<dyn Array>> {
    let schema =
        ArrowSchema::from_iter([Field::new("a".into(), array.dtype().clone(), is_nullable)]);
    let chunk = RecordBatchT::try_new(array.len(), Arc::new(schema.clone()), vec![array])?;
    let mut reader = Cursor::new(integration_write(&schema, &[chunk])?);
    let metadata = p_read::read_metadata(&mut reader)?;
    Ok(
        read_columns_many(&mut reader, &metadata.row_groups[0], &schema, Some(filter))?
            .remove(0)
            .remove(0),
    )
}

/// Checks that `array` holds `expected` in an allocation of exactly that many values.
fn assert_exactly_allocated(mut array: Box<dyn Array>, expected: &[i64]) {
    let array = std::mem::take(array.as_any_mut().downcast_mut::<Int64Array>().unwrap());
    assert_eq!(array, Int64Array::from_slice(expected));
    let values = array.into_inner().1.into_mut().right().unwrap();
    assert_eq!(values.capacity(), expected.len());
}

#[test]
fn masked_read_allocates_only_kept_values() -> PolarsResult<()> {
    // A masked read allocates the kept values of a flat column and of a struct field, and
    // decoding them must not grow that allocation.
    let values = Int64Array::from_vec((0..1000).collect());
    let keep = |v: &i64| v % 3 != 1;
    let mask = Bitmap::from_iter(values.values_iter().map(keep));
    let expected = values
        .values_iter()
        .copied()
        .filter(keep)
        .collect::<Vec<_>>();

    let struct_dtype =
        ArrowDataType::Struct(vec![Field::new("a".into(), ArrowDataType::Int64, false)]);
    let structs = StructArray::new(
        struct_dtype.clone(),
        values.len(),
        vec![values.clone().boxed()],
        None,
    );
    let filter = p_read::Filter::Mask(mask);
    assert_exactly_allocated(
        filtered_roundtrip(values.boxed(), false, filter.clone())?,
        &expected,
    );

    let mut column = filtered_roundtrip(structs.boxed(), false, filter)?;
    let structs = column.as_any_mut().downcast_mut::<StructArray>().unwrap();
    let empty = StructArray::new_empty(structs.dtype().clone());
    let (_, _, mut fields, _) = std::mem::replace(structs, empty).into_data();
    assert_exactly_allocated(fields.pop().unwrap(), &expected);

    Ok(())
}

#[test]
fn masked_list_read_allocates_kept_share_of_values() -> PolarsResult<()> {
    // Each row has 3 values. Check both a dense selection and one row from a large page.
    for sparse in [false, true] {
        let keep = |i: &i64| if sparse { *i == 0 } else { i % 3 != 1 };
        let mut lists = MutableListArray::<i64, _>::new_with_field(
            MutablePrimitiveArray::<i64>::new(),
            "item".into(),
            false,
        );
        let rows = (0..1000i64).map(|i| Some((3 * i..3 * i + 3).map(Some).collect::<Vec<_>>()));
        lists.try_extend(rows)?;
        let lists: ListArray<i64> = lists.into();
        let mask = Bitmap::from_iter((0..1000).map(|i| keep(&i)));
        let expected = (0..1000i64)
            .filter(keep)
            .flat_map(|i| 3 * i..3 * i + 3)
            .collect::<Vec<_>>();

        let mut column = filtered_roundtrip(lists.boxed(), false, p_read::Filter::Mask(mask))?;
        let lists = column
            .as_any_mut()
            .downcast_mut::<ListArray<i64>>()
            .unwrap();
        let empty = ListArray::new_empty(lists.dtype().clone());
        let (_, values, offsets, _) = std::mem::replace(lists, empty).into_inner();
        assert_exactly_allocated(values, &expected);
        let offsets = offsets.into_mut().right().unwrap();
        assert!(offsets.capacity() <= 2 * (expected.len() / 3 + 1));
    }

    Ok(())
}

#[test]
fn masked_read_refuses_chunk_with_wrong_number_of_rows() -> PolarsResult<()> {
    for array in [pyarrow_struct("struct"), data(0..12i64, false)] {
        let rows = array.len();
        for (mask_len, message) in [
            (rows - 1, "more rows than the filter mask".to_string()),
            (
                rows + 1,
                format!("{rows} rows where {} were expected", rows + 1),
            ),
        ] {
            let mask = Bitmap::from_iter((0..mask_len).map(|i| i == 0));
            let err =
                filtered_roundtrip(array.clone(), false, p_read::Filter::Mask(mask)).unwrap_err();
            assert!(err.to_string().contains(&message), "{err}");
        }
    }
    Ok(())
}

/// `column == needle` on an `Int64` column. As it is specialized, the decoders evaluate it on
/// the encoded values.
struct Int64Equals(SpecializedParquetColumnExpr);

impl ParquetColumnExpr for Int64Equals {
    fn try_evaluate_mut(&self, values: &dyn Array, bm: &mut BitmapBuilder) -> PolarsResult<()> {
        let SpecializedParquetColumnExpr::Equal(ParquetScalar::Int64(needle)) = &self.0 else {
            unreachable!()
        };
        let values = values.as_any().downcast_ref::<Int64Array>().unwrap();
        bm.extend_trusted_len_iter(values.values_iter().map(|v| v == needle));
        Ok(())
    }

    fn evaluate_null(&self) -> PolarsResult<bool> {
        Ok(false)
    }

    fn as_specialized(&self) -> Option<&SpecializedParquetColumnExpr> {
        Some(&self.0)
    }
}

#[test]
fn equality_predicate_read_returns_valid_needles() -> PolarsResult<()> {
    // A read that keeps the values matching `column == 2` writes the needle for each match,
    // which must be valid, for a required and an optional column.
    let predicate = p_read::PredicateFilter {
        predicate: Arc::new(Int64Equals(SpecializedParquetColumnExpr::Equal(
            ParquetScalar::Int64(2),
        ))),
        include_values: true,
    };
    for is_nullable in [false, true] {
        let array = Int64Array::from_vec((0..1000).map(|i| i % 4).collect());
        let filter = p_read::Filter::Predicate(predicate.clone());
        let array = filtered_roundtrip(array.boxed(), is_nullable, filter)?;
        let column = array.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(column, &Int64Array::from_vec(vec![2; 250]));
    }

    Ok(())
}
