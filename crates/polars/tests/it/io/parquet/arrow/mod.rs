mod read;
mod write;

use std::io::{Cursor, Read, Seek};

use arrow::array::*;
use arrow::bitmap::Bitmap;
use arrow::datatypes::*;
use arrow::legacy::prelude::LargeListArray;
use arrow::record_batch::RecordBatchT;
use arrow::types::{i256, NativeType};
use ethnum::AsI256;
use polars_error::PolarsResult;
use polars_parquet::read as p_read;
use polars_parquet::read::statistics::*;
use polars_parquet::write::*;

use super::read::file::FileReader;

fn new_struct(
    arrays: Vec<Box<dyn Array>>,
    names: Vec<String>,
    validity: Option<Bitmap>,
) -> StructArray {
    let fields = names
        .into_iter()
        .zip(arrays.iter())
        .map(|(n, a)| Field::new(n, a.data_type().clone(), true))
        .collect();
    StructArray::new(ArrowDataType::Struct(fields), arrays, validity)
}

pub fn read_column<R: Read + Seek>(mut reader: R, column: &str) -> PolarsResult<Box<dyn Array>> {
    let metadata = p_read::read_metadata(&mut reader)?;
    let schema = p_read::infer_schema(&metadata)?;

    let schema = schema.filter(|_, f| f.name == column);

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
                    "item",
                    ArrowDataType::Utf8View,
                    true,
                ))),
                vec![0, 4].try_into().unwrap(),
                Utf8ViewArray::from_slice([Some("a"), Some("b"), None, Some("c")]).boxed(),
                None,
            );
            StructArray::new(
                ArrowDataType::Struct(vec![Field::new("f1", a.data_type().clone(), true)]),
                vec![a.boxed()],
                None,
            )
            .boxed()
        },
        "list_struct_list_nullable" => {
            let values = pyarrow_nested_edge("struct_list_nullable");
            ListArray::<i64>::new(
                ArrowDataType::LargeList(Box::new(Field::new(
                    "item",
                    values.data_type().clone(),
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
            new_struct(
                vec![array],
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
                    "item",
                    array.data_type().clone(),
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

            new_struct(
                vec![array],
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
            let data_type =
                ArrowDataType::LargeList(Box::new(Field::new("item", ArrowDataType::Int64, false)));
            ListArray::<i64>::new(data_type, offsets, values, None).boxed()
        },
        "list_int64_optional_required" => {
            // [[0, 1], [], [2, 0, 3], [4, 5, 6], [], [7, 8, 9], [], [10]]
            let data_type =
                ArrowDataType::LargeList(Box::new(Field::new("item", ArrowDataType::Int64, true)));
            ListArray::<i64>::new(data_type, offsets, values, None).boxed()
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
        "struct_list_nullable" => new_struct(vec![values], vec!["a".to_string()], None).boxed(),
        _ => {
            let field = match column {
                "list_int64" => Field::new("item", ArrowDataType::Int64, true),
                "list_int64_required" => Field::new("item", ArrowDataType::Int64, false),
                "list_int16" => Field::new("item", ArrowDataType::Int16, true),
                "list_bool" => Field::new("item", ArrowDataType::Boolean, true),
                "list_utf8" => Field::new("item", ArrowDataType::Utf8View, true),
                "list_large_binary" => Field::new("item", ArrowDataType::LargeBinary, true),
                "list_decimal" => Field::new("item", ArrowDataType::Decimal(9, 0), true),
                "list_decimal256" => Field::new("item", ArrowDataType::Decimal256(9, 0), true),
                "list_struct_nullable" => Field::new("item", values.data_type().clone(), true),
                "list_struct_list_nullable" => Field::new("item", values.data_type().clone(), true),
                other => unreachable!("{}", other),
            };

            let validity = Some(Bitmap::from([
                true, false, true, true, true, true, false, true,
            ]));
            // [0, 2, 2, 5, 8, 8, 11, 11, 12]
            // [[a1, a2], None, [a3, a4, a5], [a6, a7, a8], [], [a9, a10, a11], None, [a12]]
            let data_type = ArrowDataType::LargeList(Box::new(field));
            ListArray::<i64>::new(data_type, offsets, values, validity).boxed()
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
            Box::new(DictionaryArray::try_from_keys(keys, values).unwrap())
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
            ArrowDataType::Timestamp(TimeUnit::Second, Some("UTC".to_string())),
        )),
        _ => unreachable!(),
    }
}

pub fn pyarrow_nullable_statistics(column: &str) -> Statistics {
    match column {
        "int64" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(3)]).boxed(),
            min_value: Box::new(Int64Array::from_slice([-256])),
            max_value: Box::new(Int64Array::from_slice([9])),
        },
        "float64" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(3)]).boxed(),
            min_value: Box::new(Float64Array::from_slice([0.0])),
            max_value: Box::new(Float64Array::from_slice([9.0])),
        },
        "string" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(4)]).boxed(),
            min_value: Box::new(Utf8ViewArray::from_slice([Some("")])),
            max_value: Box::new(Utf8ViewArray::from_slice([Some("def")])),
        },
        "bool" => Statistics {
            distinct_count: UInt64Array::from([Some(2)]).boxed(),
            null_count: UInt64Array::from([Some(4)]).boxed(),
            min_value: Box::new(BooleanArray::from_slice([false])),
            max_value: Box::new(BooleanArray::from_slice([true])),
        },
        "timestamp_ms" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(3)]).boxed(),
            min_value: Box::new(
                Int64Array::from_slice([0])
                    .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None)),
            ),
            max_value: Box::new(
                Int64Array::from_slice([9])
                    .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None)),
            ),
        },
        "uint32" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(3)]).boxed(),
            min_value: Box::new(UInt32Array::from_slice([0])),
            max_value: Box::new(UInt32Array::from_slice([9])),
        },
        "int32_dict" => {
            let new_dict = |array: Box<dyn Array>| -> Box<dyn Array> {
                Box::new(DictionaryArray::try_from_keys(vec![Some(0)].into(), array).unwrap())
            };

            Statistics {
                distinct_count: UInt64Array::from([None]).boxed(),
                null_count: UInt64Array::from([Some(1)]).boxed(),
                min_value: new_dict(Box::new(Int32Array::from_slice([10]))),
                max_value: new_dict(Box::new(Int32Array::from_slice([200]))),
            }
        },
        "timestamp_us" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(3)]).boxed(),
            min_value: Box::new(
                Int64Array::from_slice([-256])
                    .to(ArrowDataType::Timestamp(TimeUnit::Microsecond, None)),
            ),
            max_value: Box::new(
                Int64Array::from_slice([9])
                    .to(ArrowDataType::Timestamp(TimeUnit::Microsecond, None)),
            ),
        },
        "timestamp_s" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(3)]).boxed(),
            min_value: Box::new(
                Int64Array::from_slice([-256]).to(ArrowDataType::Timestamp(TimeUnit::Second, None)),
            ),
            max_value: Box::new(
                Int64Array::from_slice([9]).to(ArrowDataType::Timestamp(TimeUnit::Second, None)),
            ),
        },
        "timestamp_s_utc" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(3)]).boxed(),
            min_value: Box::new(Int64Array::from_slice([-256]).to(ArrowDataType::Timestamp(
                TimeUnit::Second,
                Some("UTC".to_string()),
            ))),
            max_value: Box::new(Int64Array::from_slice([9]).to(ArrowDataType::Timestamp(
                TimeUnit::Second,
                Some("UTC".to_string()),
            ))),
        },
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

pub fn pyarrow_required_statistics(column: &str) -> Statistics {
    let mut s = pyarrow_nullable_statistics(column);
    s.null_count = UInt64Array::from([Some(0)]).boxed();
    s
}

pub fn pyarrow_nested_nullable_statistics(column: &str) -> Statistics {
    let new_list = |array: Box<dyn Array>, nullable: bool| {
        ListArray::<i64>::new(
            ArrowDataType::LargeList(Box::new(Field::new(
                "item",
                array.data_type().clone(),
                nullable,
            ))),
            vec![0, array.len() as i64].try_into().unwrap(),
            array,
            None,
        )
    };

    match column {
        "list_int16" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(Box::new(Int16Array::from_slice([0])), true).boxed(),
            max_value: new_list(Box::new(Int16Array::from_slice([10])), true).boxed(),
        },
        "list_bool" => Statistics {
            distinct_count: new_list(UInt64Array::from([Some(2)]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(Box::new(BooleanArray::from_slice([false])), true).boxed(),
            max_value: new_list(Box::new(BooleanArray::from_slice([true])), true).boxed(),
        },
        "list_utf8" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(Box::new(Utf8ViewArray::from_slice([Some("")])), true).boxed(),
            max_value: new_list(Box::new(Utf8ViewArray::from_slice([Some("ccc")])), true).boxed(),
        },
        "list_large_binary" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(Box::new(BinaryArray::<i64>::from_slice([b""])), true).boxed(),
            max_value: new_list(Box::new(BinaryArray::<i64>::from_slice([b"ccc"])), true).boxed(),
        },
        "list_decimal" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(
                Box::new(Int128Array::from_slice([0]).to(ArrowDataType::Decimal(9, 0))),
                true,
            )
            .boxed(),
            max_value: new_list(
                Box::new(Int128Array::from_slice([10]).to(ArrowDataType::Decimal(9, 0))),
                true,
            )
            .boxed(),
        },
        "list_decimal256" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(
                Box::new(
                    Int256Array::from_slice([i256(0.as_i256())])
                        .to(ArrowDataType::Decimal256(9, 0)),
                ),
                true,
            )
            .boxed(),
            max_value: new_list(
                Box::new(
                    Int256Array::from_slice([i256(10.as_i256())])
                        .to(ArrowDataType::Decimal256(9, 0)),
                ),
                true,
            )
            .boxed(),
        },
        "list_int64" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(Box::new(Int64Array::from_slice([0])), true).boxed(),
            max_value: new_list(Box::new(Int64Array::from_slice([10])), true).boxed(),
        },
        "list_int64_required" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed(),
            min_value: new_list(Box::new(Int64Array::from_slice([0])), false).boxed(),
            max_value: new_list(Box::new(Int64Array::from_slice([10])), false).boxed(),
        },
        "list_int64_required_required" | "list_int64_optional_required" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), false).boxed(),
            null_count: new_list(UInt64Array::from([Some(0)]).boxed(), false).boxed(),
            min_value: new_list(Box::new(Int64Array::from_slice([0])), false).boxed(),
            max_value: new_list(Box::new(Int64Array::from_slice([10])), false).boxed(),
        },
        "list_nested_i64" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed(), true).boxed(),
            null_count: new_list(UInt64Array::from([Some(2)]).boxed(), true).boxed(),
            min_value: new_list(
                new_list(Box::new(Int64Array::from_slice([0])), true).boxed(),
                true,
            )
            .boxed(),
            max_value: new_list(
                new_list(Box::new(Int64Array::from_slice([10])), true).boxed(),
                true,
            )
            .boxed(),
        },
        "list_nested_inner_required_required_i64" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(0)]).boxed(),
            min_value: new_list(
                new_list(Box::new(Int64Array::from_slice([0])), true).boxed(),
                true,
            )
            .boxed(),
            max_value: new_list(
                new_list(Box::new(Int64Array::from_slice([10])), true).boxed(),
                true,
            )
            .boxed(),
        },
        "list_nested_inner_required_i64" => Statistics {
            distinct_count: UInt64Array::from([None]).boxed(),
            null_count: UInt64Array::from([Some(0)]).boxed(),
            min_value: new_list(
                new_list(Box::new(Int64Array::from_slice([0])), true).boxed(),
                true,
            )
            .boxed(),
            max_value: new_list(
                new_list(Box::new(Int64Array::from_slice([10])), true).boxed(),
                true,
            )
            .boxed(),
        },
        "list_struct_nullable" => Statistics {
            distinct_count: new_list(
                new_struct(
                    vec![UInt64Array::from([None]).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
            null_count: new_list(
                new_struct(
                    vec![UInt64Array::from([Some(4)]).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
            min_value: new_list(
                new_struct(
                    vec![Utf8ViewArray::from_slice([Some("a")]).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
            max_value: new_list(
                new_struct(
                    vec![Utf8ViewArray::from_slice([Some("e")]).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
        },
        "list_struct_list_nullable" => Statistics {
            distinct_count: new_list(
                new_struct(
                    vec![new_list(UInt64Array::from([None]).boxed(), true).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
            null_count: new_list(
                new_struct(
                    vec![new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
            min_value: new_list(
                new_struct(
                    vec![new_list(Utf8ViewArray::from_slice([Some("a")]).boxed(), true).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
            max_value: new_list(
                new_struct(
                    vec![new_list(Utf8ViewArray::from_slice([Some("d")]).boxed(), true).boxed()],
                    vec!["a".to_string()],
                    None,
                )
                .boxed(),
                true,
            )
            .boxed(),
        },
        "struct_list_nullable" => Statistics {
            distinct_count: new_struct(
                vec![new_list(UInt64Array::from([None]).boxed(), true).boxed()],
                vec!["a".to_string()],
                None,
            )
            .boxed(),
            null_count: new_struct(
                vec![new_list(UInt64Array::from([Some(1)]).boxed(), true).boxed()],
                vec!["a".to_string()],
                None,
            )
            .boxed(),
            min_value: new_struct(
                vec![new_list(Utf8ViewArray::from_slice([Some("")]).boxed(), true).boxed()],
                vec!["a".to_string()],
                None,
            )
            .boxed(),
            max_value: new_struct(
                vec![new_list(Utf8ViewArray::from_slice([Some("ccc")]).boxed(), true).boxed()],
                vec!["a".to_string()],
                None,
            )
            .boxed(),
        },
        other => todo!("{}", other),
    }
}

pub fn pyarrow_nested_edge_statistics(column: &str) -> Statistics {
    let new_list = |array: Box<dyn Array>| {
        ListArray::<i64>::new(
            ArrowDataType::LargeList(Box::new(Field::new(
                "item",
                array.data_type().clone(),
                true,
            ))),
            vec![0, array.len() as i64].try_into().unwrap(),
            array,
            None,
        )
    };

    let new_struct = |arrays: Vec<Box<dyn Array>>, names: Vec<String>| {
        let fields = names
            .into_iter()
            .zip(arrays.iter())
            .map(|(n, a)| Field::new(n, a.data_type().clone(), true))
            .collect();
        StructArray::new(ArrowDataType::Struct(fields), arrays, None)
    };

    let names = vec!["f1".to_string()];

    match column {
        "simple" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed()).boxed(),
            null_count: new_list(UInt64Array::from([Some(0)]).boxed()).boxed(),
            min_value: new_list(Box::new(Int64Array::from([Some(0)]))).boxed(),
            max_value: new_list(Box::new(Int64Array::from([Some(1)]))).boxed(),
        },
        "null" | "empty" => Statistics {
            distinct_count: new_list(UInt64Array::from([None]).boxed()).boxed(),
            null_count: new_list(UInt64Array::from([Some(0)]).boxed()).boxed(),
            min_value: new_list(Box::new(Int64Array::from([None]))).boxed(),
            max_value: new_list(Box::new(Int64Array::from([None]))).boxed(),
        },
        "struct_list_nullable" => Statistics {
            distinct_count: new_struct(
                vec![new_list(Box::new(UInt64Array::from([None]))).boxed()],
                names.clone(),
            )
            .boxed(),
            null_count: new_struct(
                vec![new_list(Box::new(UInt64Array::from([Some(1)]))).boxed()],
                names.clone(),
            )
            .boxed(),
            min_value: Box::new(new_struct(
                vec![new_list(Box::new(Utf8ViewArray::from_slice([Some("a")]))).boxed()],
                names.clone(),
            )),
            max_value: Box::new(new_struct(
                vec![new_list(Box::new(Utf8ViewArray::from_slice([Some("c")]))).boxed()],
                names,
            )),
        },
        "list_struct_list_nullable" => Statistics {
            distinct_count: new_list(
                new_struct(
                    vec![new_list(Box::new(UInt64Array::from([None]))).boxed()],
                    names.clone(),
                )
                .boxed(),
            )
            .boxed(),
            null_count: new_list(
                new_struct(
                    vec![new_list(Box::new(UInt64Array::from([Some(1)]))).boxed()],
                    names.clone(),
                )
                .boxed(),
            )
            .boxed(),
            min_value: new_list(Box::new(new_struct(
                vec![new_list(Box::new(Utf8ViewArray::from_slice([Some("a")]))).boxed()],
                names.clone(),
            )))
            .boxed(),
            max_value: new_list(Box::new(new_struct(
                vec![new_list(Box::new(Utf8ViewArray::from_slice([Some("c")]))).boxed()],
                names,
            )))
            .boxed(),
        },
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
        Field::new("f1", ArrowDataType::Utf8View, true),
        Field::new("f2", ArrowDataType::Boolean, true),
    ];
    match column {
        "struct" => {
            StructArray::new(ArrowDataType::Struct(fields), vec![string, boolean], None).boxed()
        },
        "struct_nullable" => {
            let values = vec![string, boolean];
            StructArray::new(ArrowDataType::Struct(fields), values, Some(mask.into())).boxed()
        },
        "struct_struct" => {
            let struct_ = pyarrow_struct("struct");
            Box::new(StructArray::new(
                ArrowDataType::Struct(vec![
                    Field::new("f1", ArrowDataType::Struct(fields), true),
                    Field::new("f2", ArrowDataType::Boolean, true),
                ]),
                vec![struct_, boolean],
                None,
            ))
        },
        "struct_struct_nullable" => {
            let struct_ = pyarrow_struct("struct");
            Box::new(StructArray::new(
                ArrowDataType::Struct(vec![
                    Field::new("f1", ArrowDataType::Struct(fields), true),
                    Field::new("f2", ArrowDataType::Boolean, true),
                ]),
                vec![struct_, boolean],
                Some(mask.into()),
            ))
        },
        _ => todo!(),
    }
}

pub fn pyarrow_struct_statistics(column: &str) -> Statistics {
    let new_struct =
        |arrays: Vec<Box<dyn Array>>, names: Vec<String>| new_struct(arrays, names, None);

    let names = vec!["f1".to_string(), "f2".to_string()];

    match column {
        "struct" | "struct_nullable" => Statistics {
            distinct_count: new_struct(
                vec![
                    Box::new(UInt64Array::from([None])),
                    Box::new(UInt64Array::from([Some(2)])),
                ],
                names.clone(),
            )
            .boxed(),
            null_count: new_struct(
                vec![
                    Box::new(UInt64Array::from([Some(4)])),
                    Box::new(UInt64Array::from([Some(4)])),
                ],
                names.clone(),
            )
            .boxed(),
            min_value: Box::new(new_struct(
                vec![
                    Box::new(Utf8ViewArray::from_slice([Some("")])),
                    Box::new(BooleanArray::from_slice([false])),
                ],
                names.clone(),
            )),
            max_value: Box::new(new_struct(
                vec![
                    Box::new(Utf8ViewArray::from_slice([Some("def")])),
                    Box::new(BooleanArray::from_slice([true])),
                ],
                names,
            )),
        },
        "struct_struct" => Statistics {
            distinct_count: new_struct(
                vec![
                    new_struct(
                        vec![
                            Box::new(UInt64Array::from([None])),
                            Box::new(UInt64Array::from([Some(2)])),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    UInt64Array::from([None]).boxed(),
                ],
                names.clone(),
            )
            .boxed(),
            null_count: new_struct(
                vec![
                    new_struct(
                        vec![
                            Box::new(UInt64Array::from([Some(4)])),
                            Box::new(UInt64Array::from([Some(4)])),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    UInt64Array::from([Some(4)]).boxed(),
                ],
                names.clone(),
            )
            .boxed(),
            min_value: new_struct(
                vec![
                    new_struct(
                        vec![
                            Utf8ViewArray::from_slice([Some("")]).boxed(),
                            BooleanArray::from_slice([false]).boxed(),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    BooleanArray::from_slice([false]).boxed(),
                ],
                names.clone(),
            )
            .boxed(),
            max_value: new_struct(
                vec![
                    new_struct(
                        vec![
                            Utf8ViewArray::from_slice([Some("def")]).boxed(),
                            BooleanArray::from_slice([true]).boxed(),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    BooleanArray::from_slice([true]).boxed(),
                ],
                names,
            )
            .boxed(),
        },
        "struct_struct_nullable" => Statistics {
            distinct_count: new_struct(
                vec![
                    new_struct(
                        vec![
                            Box::new(UInt64Array::from([None])),
                            Box::new(UInt64Array::from([None])),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    UInt64Array::from([None]).boxed(),
                ],
                names.clone(),
            )
            .boxed(),
            null_count: new_struct(
                vec![
                    new_struct(
                        vec![
                            Box::new(UInt64Array::from([Some(5)])),
                            Box::new(UInt64Array::from([Some(5)])),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    UInt64Array::from([Some(5)]).boxed(),
                ],
                names.clone(),
            )
            .boxed(),
            min_value: new_struct(
                vec![
                    new_struct(
                        vec![
                            Utf8ViewArray::from_slice([Some("")]).boxed(),
                            BooleanArray::from_slice([false]).boxed(),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    BooleanArray::from_slice([false]).boxed(),
                ],
                names.clone(),
            )
            .boxed(),
            max_value: new_struct(
                vec![
                    new_struct(
                        vec![
                            Utf8ViewArray::from_slice([Some("def")]).boxed(),
                            BooleanArray::from_slice([true]).boxed(),
                        ],
                        names.clone(),
                    )
                    .boxed(),
                    BooleanArray::from_slice([true]).boxed(),
                ],
                names,
            )
            .boxed(),
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

    let encodings = schema
        .fields
        .iter()
        .map(|f| {
            transverse(&f.data_type, |x| {
                if let ArrowDataType::Dictionary(..) = x {
                    Encoding::RleDictionary
                } else {
                    Encoding::Plain
                }
            })
        })
        .collect();

    let row_groups =
        RowGroupIterator::try_new(chunks.iter().cloned().map(Ok), schema, options, encodings)?;

    let writer = Cursor::new(vec![]);

    let mut writer = FileWriter::try_new(writer, schema.clone(), options)?;

    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None)?;

    Ok(writer.into_inner().into_inner())
}

type IntegrationRead = (ArrowSchema, Vec<RecordBatchT<Box<dyn Array>>>);

fn integration_read(data: &[u8], limit: Option<usize>) -> PolarsResult<IntegrationRead> {
    let mut reader = Cursor::new(data);
    let metadata = p_read::read_metadata(&mut reader)?;
    let schema = p_read::infer_schema(&metadata)?;

    let reader = FileReader::new(
        Cursor::new(data),
        metadata.row_groups,
        schema.clone(),
        limit,
    );

    let batches = reader.collect::<PolarsResult<Vec<_>>>()?;

    Ok((schema, batches))
}

fn generic_data() -> PolarsResult<(ArrowSchema, RecordBatchT<Box<dyn Array>>)> {
    let array1 = PrimitiveArray::<i64>::from([Some(1), None, Some(2)])
        .to(ArrowDataType::Duration(TimeUnit::Second));
    let array2 = Utf8ViewArray::from_slice([Some("a"), None, Some("bb")]);

    let indices = PrimitiveArray::from_values((0..3u64).map(|x| x % 2));
    let values = PrimitiveArray::from_slice([1.0f32, 3.0]).boxed();
    let array3 = DictionaryArray::try_from_keys(indices.clone(), values).unwrap();

    let array4 = BinaryViewArray::from_slice([Some(b"ab"), Some(b"aa"), Some(b"ac")]);

    let values = PrimitiveArray::from_slice([1i16, 3]).boxed();
    let array6 = DictionaryArray::try_from_keys(indices.clone(), values).unwrap();

    let values = PrimitiveArray::from_slice([1i64, 3])
        .to(ArrowDataType::Timestamp(
            TimeUnit::Millisecond,
            Some("UTC".to_string()),
        ))
        .boxed();
    let array7 = DictionaryArray::try_from_keys(indices.clone(), values).unwrap();

    let values = PrimitiveArray::from_slice([1.0f64, 3.0]).boxed();
    let array8 = DictionaryArray::try_from_keys(indices.clone(), values).unwrap();

    let values = PrimitiveArray::from_slice([1u8, 3]).boxed();
    let array9 = DictionaryArray::try_from_keys(indices.clone(), values).unwrap();

    let values = PrimitiveArray::from_slice([1u16, 3]).boxed();
    let array10 = DictionaryArray::try_from_keys(indices.clone(), values).unwrap();

    let values = PrimitiveArray::from_slice([1u32, 3]).boxed();
    let array11 = DictionaryArray::try_from_keys(indices.clone(), values).unwrap();

    let values = PrimitiveArray::from_slice([1u64, 3]).boxed();
    let array12 = DictionaryArray::try_from_keys(indices, values).unwrap();

    let array13 = PrimitiveArray::<i32>::from_slice([1, 2, 3])
        .to(ArrowDataType::Interval(IntervalUnit::YearMonth));

    let schema = ArrowSchema::from(vec![
        Field::new("a1", array1.data_type().clone(), true),
        Field::new("a2", array2.data_type().clone(), true),
        Field::new("a3", array3.data_type().clone(), true),
        Field::new("a4", array4.data_type().clone(), true),
        Field::new("a6", array6.data_type().clone(), true),
        Field::new("a7", array7.data_type().clone(), true),
        Field::new("a8", array8.data_type().clone(), true),
        Field::new("a9", array9.data_type().clone(), true),
        Field::new("a10", array10.data_type().clone(), true),
        Field::new("a11", array11.data_type().clone(), true),
        Field::new("a12", array12.data_type().clone(), true),
        Field::new("a13", array13.data_type().clone(), true),
    ]);
    let chunk = RecordBatchT::try_new(vec![
        array1.boxed(),
        array2.boxed(),
        array3.boxed(),
        array4.boxed(),
        array6.boxed(),
        array7.boxed(),
        array8.boxed(),
        array9.boxed(),
        array10.boxed(),
        array11.boxed(),
        array12.boxed(),
        array13.boxed(),
    ])?;

    Ok((schema, chunk))
}

fn assert_roundtrip(
    schema: ArrowSchema,
    chunk: RecordBatchT<Box<dyn Array>>,
    limit: Option<usize>,
) -> PolarsResult<()> {
    let r = integration_write(&schema, &[chunk.clone()])?;

    let (new_schema, new_chunks) = integration_read(&r, limit)?;

    let expected = if let Some(limit) = limit {
        let expected = chunk
            .into_arrays()
            .into_iter()
            .map(|x| x.sliced(0, limit))
            .collect::<Vec<_>>();
        RecordBatchT::new(expected)
    } else {
        chunk
    };

    assert_eq!(new_schema, schema);
    assert_eq!(new_chunks, vec![expected]);
    Ok(())
}

/// Tests that when arrow-specific types (Duration and LargeUtf8) are written to parquet, we can roundtrip its
/// logical types.
#[test]
fn arrow_type() -> PolarsResult<()> {
    let (schema, chunk) = generic_data()?;
    assert_roundtrip(schema, chunk, None)
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
        "item",
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
    let schema = ArrowSchema::from(vec![Field::new(
        "a1",
        array.data_type().clone(),
        is_nullable,
    )]);
    let chunk = RecordBatchT::try_new(vec![array])?;

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
        "item",
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
        "item",
        true,
    );
    array.try_extend(data).unwrap();
    assert_array_roundtrip(true, array.into_box(), None)
}

#[test]
fn limit() -> PolarsResult<()> {
    let (schema, chunk) = generic_data()?;
    assert_roundtrip(schema, chunk, Some(2))
}

#[test]
fn limit_list() -> PolarsResult<()> {
    test_list_array_required_required(Some(2))
}

fn nested_dict_data(
    data_type: ArrowDataType,
) -> PolarsResult<(ArrowSchema, RecordBatchT<Box<dyn Array>>)> {
    let values = match data_type {
        ArrowDataType::Float32 => PrimitiveArray::from_slice([1.0f32, 3.0]).boxed(),
        ArrowDataType::Utf8View => Utf8ViewArray::from_slice([Some("a"), Some("b")]).boxed(),
        _ => unreachable!(),
    };

    let indices = PrimitiveArray::from_values((0..3u64).map(|x| x % 2));
    let values = DictionaryArray::try_from_keys(indices, values).unwrap();
    let values = LargeListArray::try_new(
        ArrowDataType::LargeList(Box::new(Field::new(
            "item",
            values.data_type().clone(),
            false,
        ))),
        vec![0i64, 0, 0, 2, 3].try_into().unwrap(),
        values.boxed(),
        Some([true, false, true, true].into()),
    )?;

    let schema = ArrowSchema::from(vec![Field::new("c1", values.data_type().clone(), true)]);
    let chunk = RecordBatchT::try_new(vec![values.boxed()])?;

    Ok((schema, chunk))
}

#[test]
fn nested_dict() -> PolarsResult<()> {
    let (schema, chunk) = nested_dict_data(ArrowDataType::Float32)?;

    assert_roundtrip(schema, chunk, None)
}

#[test]
fn nested_dict_utf8() -> PolarsResult<()> {
    let (schema, chunk) = nested_dict_data(ArrowDataType::Utf8View)?;

    assert_roundtrip(schema, chunk, None)
}

#[test]
fn nested_dict_limit() -> PolarsResult<()> {
    let (schema, chunk) = nested_dict_data(ArrowDataType::Float32)?;

    assert_roundtrip(schema, chunk, Some(2))
}

#[test]
fn filter_chunk() -> PolarsResult<()> {
    let chunk1 = RecordBatchT::new(vec![PrimitiveArray::from_slice([1i16, 3]).boxed()]);
    let chunk2 = RecordBatchT::new(vec![PrimitiveArray::from_slice([2i16, 4]).boxed()]);
    let schema = ArrowSchema::from(vec![Field::new("c1", ArrowDataType::Int16, true)]);

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
