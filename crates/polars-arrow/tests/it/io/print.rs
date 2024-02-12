use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::buffer::Buffer;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::{ArrowDataType, Field, TimeUnit, UnionMode};
use polars_arrow::error::Result;
use polars_arrow::io::print::*;

#[test]
fn write_basics() -> Result<()> {
    let a = Utf8Array::<i32>::from(vec![Some("a"), Some("b"), None, Some("d")]);
    let b = Int32Array::from(vec![Some(1), None, Some(10), Some(100)]);

    let batch = Chunk::try_new(vec![&a as &dyn Array, &b])?;

    let table = write(&[batch], &["a".to_string(), "b".to_string()]);

    let expected = vec![
        "+---+-----+",
        "| a | b   |",
        "+---+-----+",
        "| a | 1   |",
        "| b |     |",
        "|   | 10  |",
        "| d | 100 |",
        "+---+-----+",
    ];

    let actual: Vec<&str> = table.lines().collect();

    assert_eq!(expected, actual, "Actual result:\n{table}");

    Ok(())
}

#[test]
fn write_null() -> Result<()> {
    let num_rows = 4;
    let arrays = [
        ArrowDataType::Utf8,
        ArrowDataType::Int32,
        ArrowDataType::Null,
    ]
    .iter()
    .map(|dt| new_null_array(dt.clone(), num_rows))
    .collect();

    // define data (null)
    let columns = Chunk::try_new(arrays)?;

    let table = write(&[columns], &["a", "b", "c"]);

    let expected = vec![
        "+---+---+---+",
        "| a | b | c |",
        "+---+---+---+",
        "|   |   |   |",
        "|   |   |   |",
        "|   |   |   |",
        "|   |   |   |",
        "+---+---+---+",
    ];

    let actual: Vec<&str> = table.lines().collect();

    assert_eq!(expected, actual, "Actual result:\n{table:#?}");
    Ok(())
}

#[test]
fn write_dictionary() -> Result<()> {
    let mut array = MutableDictionaryArray::<i32, MutableUtf8Array<i32>>::new();

    array.try_extend(vec![Some("one"), None, Some("three")])?;
    let array = array.into_box();

    let batch = Chunk::new(vec![array]);

    let table = write(&[batch], &["d1"]);

    let expected = vec![
        "+-------+",
        "| d1    |",
        "+-------+",
        "| one   |",
        "|       |",
        "| three |",
        "+-------+",
    ];

    let actual: Vec<&str> = table.lines().collect();

    assert_eq!(expected, actual, "Actual result:\n{table}");

    Ok(())
}

#[test]
fn dictionary_validities() -> Result<()> {
    let keys = PrimitiveArray::<i32>::from([Some(1), None, Some(0)]);
    let values = PrimitiveArray::<i32>::from([None, Some(10)]);
    let array = DictionaryArray::try_from_keys(keys, Box::new(values)).unwrap();

    let columns = Chunk::new(vec![&array as &dyn Array]);

    let table = write(&[columns], &["d1"]);

    let expected = vec![
        "+----+", "| d1 |", "+----+", "| 10 |", "|    |", "|    |", "+----+",
    ];

    let actual: Vec<&str> = table.lines().collect();

    assert_eq!(expected, actual, "Actual result:\n{table}");

    Ok(())
}

/// Generate an array with type $ARRAYTYPE with a numeric value of
/// $VALUE, and compare $EXPECTED_RESULT to the output of
/// formatting that array with `write`
macro_rules! check_datetime {
    ($ty:ty, $datatype:expr, $value:expr, $EXPECTED_RESULT:expr) => {
        let array = PrimitiveArray::<$ty>::from(&[Some($value), None]).to($datatype);
        let batch = Chunk::new(vec![&array as &dyn Array]);

        let table = write(&[batch], &["f"]);

        let expected = $EXPECTED_RESULT;
        let actual: Vec<&str> = table.lines().collect();

        assert_eq!(expected, actual, "Actual result:\n\n{:#?}\n\n", actual);
    };
}

#[test]
fn write_timestamp_second() {
    let expected = vec![
        "+---------------------+",
        "| f                   |",
        "+---------------------+",
        "| 1970-05-09 14:25:11 |",
        "|                     |",
        "+---------------------+",
    ];
    check_datetime!(
        i64,
        ArrowDataType::Timestamp(TimeUnit::Second, None),
        11111111,
        expected
    );
}

#[test]
fn write_timestamp_second_with_tz() {
    let expected = vec![
        "+----------------------------+",
        "| f                          |",
        "+----------------------------+",
        "| 1970-05-09 14:25:11 +00:00 |",
        "|                            |",
        "+----------------------------+",
    ];
    check_datetime!(
        i64,
        ArrowDataType::Timestamp(TimeUnit::Second, Some("UTC".to_string())),
        11111111,
        expected
    );
}

#[test]
fn write_timestamp_millisecond() {
    let expected = vec![
        "+-------------------------+",
        "| f                       |",
        "+-------------------------+",
        "| 1970-01-01 03:05:11.111 |",
        "|                         |",
        "+-------------------------+",
    ];
    check_datetime!(
        i64,
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
        11111111,
        expected
    );
}

#[test]
fn write_timestamp_microsecond() {
    let expected = vec![
        "+----------------------------+",
        "| f                          |",
        "+----------------------------+",
        "| 1970-01-01 00:00:11.111111 |",
        "|                            |",
        "+----------------------------+",
    ];
    check_datetime!(
        i64,
        ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
        11111111,
        expected
    );
}

#[test]
fn write_timestamp_nanosecond() {
    let expected = vec![
        "+-------------------------------+",
        "| f                             |",
        "+-------------------------------+",
        "| 1970-01-01 00:00:00.011111111 |",
        "|                               |",
        "+-------------------------------+",
    ];
    check_datetime!(
        i64,
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
        11111111,
        expected
    );
}

#[test]
fn write_date_32() {
    let expected = vec![
        "+------------+",
        "| f          |",
        "+------------+",
        "| 1973-05-19 |",
        "|            |",
        "+------------+",
    ];
    check_datetime!(i32, ArrowDataType::Date32, 1234, expected);
}

#[test]
fn write_date_64() {
    let expected = vec![
        "+------------+",
        "| f          |",
        "+------------+",
        "| 2005-03-18 |",
        "|            |",
        "+------------+",
    ];
    check_datetime!(i64, ArrowDataType::Date64, 1111111100000, expected);
}

#[test]
fn write_time_32_second() {
    let expected = vec![
        "+----------+",
        "| f        |",
        "+----------+",
        "| 00:18:31 |",
        "|          |",
        "+----------+",
    ];
    check_datetime!(i32, ArrowDataType::Time32(TimeUnit::Second), 1111, expected);
}

#[test]
fn write_time_32_millisecond() {
    let expected = vec![
        "+--------------+",
        "| f            |",
        "+--------------+",
        "| 03:05:11.111 |",
        "|              |",
        "+--------------+",
    ];
    check_datetime!(
        i32,
        ArrowDataType::Time32(TimeUnit::Millisecond),
        11111111,
        expected
    );
}

#[test]
fn write_time_64_microsecond() {
    let expected = vec![
        "+-----------------+",
        "| f               |",
        "+-----------------+",
        "| 00:00:11.111111 |",
        "|                 |",
        "+-----------------+",
    ];
    check_datetime!(
        i64,
        ArrowDataType::Time64(TimeUnit::Microsecond),
        11111111,
        expected
    );
}

#[test]
fn write_time_64_nanosecond() {
    let expected = vec![
        "+--------------------+",
        "| f                  |",
        "+--------------------+",
        "| 00:00:00.011111111 |",
        "|                    |",
        "+--------------------+",
    ];
    check_datetime!(
        i64,
        ArrowDataType::Time64(TimeUnit::Nanosecond),
        11111111,
        expected
    );
}

#[test]
fn write_struct() -> Result<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let values = vec![
        Int32Array::from(&[Some(1), None, Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let validity = Some(Bitmap::from(&[true, false, true]));

    let array = StructArray::new(ArrowDataType::Struct(fields), values, validity);

    let columns = Chunk::new(vec![&array as &dyn Array]);

    let table = write(&[columns], &["a"]);

    let expected = vec![
        "+--------------+",
        "| a            |",
        "+--------------+",
        "| {a: 1, b: a} |",
        "|              |",
        "| {a: 2, b: c} |",
        "+--------------+",
    ];

    let actual: Vec<&str> = table.lines().collect();

    assert_eq!(expected, actual, "Actual result:\n{table}");

    Ok(())
}

#[test]
fn write_union() -> Result<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let types = Buffer::from(vec![0, 0, 1]);
    let fields = vec![
        Int32Array::from(&[Some(1), None, Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type, types, fields, None);

    let batch = Chunk::new(vec![&array as &dyn Array]);

    let table = write(&[batch], &["a"]);

    let expected = vec![
        "+---+", "| a |", "+---+", "| 1 |", "|   |", "| c |", "+---+",
    ];

    let actual: Vec<&str> = table.lines().collect();

    assert_eq!(expected, actual, "Actual result:\n{table}");

    Ok(())
}
