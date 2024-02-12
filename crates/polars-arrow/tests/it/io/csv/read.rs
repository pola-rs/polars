use std::io::Cursor;

use polars_arrow::array::*;
use polars_arrow::datatypes::*;
use polars_arrow::error::Result;
use polars_arrow::io::csv::read::*;
use proptest::prelude::*;

#[test]
fn read() -> Result<()> {
    let data = r#"city,lat,lng
"Elgin, Scotland, the UK",57.653484,-3.335724
"Stoke-on-Trent, Staffordshire, the UK",53.002666,-2.179404
"Solihull, Birmingham, UK",52.412811,-1.778197
"Cardiff, Cardiff county, UK",51.481583,-3.179090
"Eastbourne, East Sussex, UK",50.768036,0.290472
"Oxford, Oxfordshire, UK",51.752022,-1.257677
"London, UK",51.509865,-0.118092
"Swindon, Swindon, UK",51.568535,-1.772232
"Gravesend, Kent, UK",51.441883,0.370759
"Northampton, Northamptonshire, UK",52.240479,-0.902656
"Rugby, Warwickshire, UK",52.370876,-1.265032
"Sutton Coldfield, West Midlands, UK",52.570385,-1.824042
"Harlow, Essex, UK",51.772938,0.102310
"Aberdeen, Aberdeen City, UK",57.149651,-2.099075"#;
    let mut reader = ReaderBuilder::new().from_reader(Cursor::new(data));

    let (fields, _) = infer_schema(&mut reader, None, true, &infer)?;

    let mut rows = vec![ByteRecord::default(); 100];
    let rows_read = read_rows(&mut reader, 0, &mut rows)?;

    let columns = deserialize_batch(&rows[..rows_read], &fields, None, 0, deserialize_column)?;

    assert_eq!(14, columns.len());
    assert_eq!(3, columns.arrays().len());

    let lat = columns.arrays()[1]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!((57.653484 - lat.value(0)).abs() < f64::EPSILON);

    let city = columns.arrays()[0]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap();

    assert_eq!("Elgin, Scotland, the UK", city.value(0));
    assert_eq!("Aberdeen, Aberdeen City, UK", city.value(13));
    Ok(())
}

#[test]
fn infer_basics() -> Result<()> {
    let file = Cursor::new("1,2,3\na,b,c\na,,c");
    let mut reader = ReaderBuilder::new().from_reader(file);

    let (fields, _) = infer_schema(&mut reader, Some(10), false, &infer)?;

    assert_eq!(
        fields,
        vec![
            Field::new("column_1", ArrowDataType::Utf8, true),
            Field::new("column_2", ArrowDataType::Utf8, true),
            Field::new("column_3", ArrowDataType::Utf8, true),
        ]
    );
    Ok(())
}

#[test]
fn infer_ints() -> Result<()> {
    let file = Cursor::new("1,2,3\n1,a,5\n2,,4");
    let mut reader = ReaderBuilder::new().from_reader(file);

    let (fields, _) = infer_schema(&mut reader, Some(10), false, &infer)?;

    assert_eq!(
        fields,
        vec![
            Field::new("column_1", ArrowDataType::Int64, true),
            Field::new("column_2", ArrowDataType::Utf8, true),
            Field::new("column_3", ArrowDataType::Int64, true),
        ]
    );
    Ok(())
}

#[test]
fn infer_ints_with_empty_fields() -> Result<()> {
    let file = Cursor::new("1,2,3\n1,3,5\n2,,4");
    let mut reader = ReaderBuilder::new().from_reader(file);

    let (fields, _) = infer_schema(&mut reader, Some(10), false, &infer)?;

    assert_eq!(
        fields,
        vec![
            Field::new("column_1", ArrowDataType::Int64, true),
            Field::new("column_2", ArrowDataType::Int64, true),
            Field::new("column_3", ArrowDataType::Int64, true),
        ]
    );
    Ok(())
}

fn test_deserialize(input: &str, data_type: ArrowDataType) -> Result<Box<dyn Array>> {
    let reader = std::io::Cursor::new(input);
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(reader);

    let mut rows = vec![ByteRecord::default(); 10];
    let rows_read = read_rows(&mut reader, 0, &mut rows)?;
    deserialize_column(&rows[..rows_read], 0, data_type, 0)
}

#[test]
fn utf8() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = Utf8Array::<i32>::from([Some("1"), Some(""), Some("3")]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn large_utf8() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = Utf8Array::<i64>::from([Some("1"), Some(""), Some("3")]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn binary() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = BinaryArray::<i32>::from([Some(b"1".as_ref()), Some(b"".as_ref()), Some(b"3")]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn large_binary() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = BinaryArray::<i64>::from([Some(b"1".as_ref()), Some(b"".as_ref()), Some(b"3")]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn u8() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = UInt8Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn u16() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = UInt16Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn u32() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = UInt32Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn u64() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = UInt64Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn i8() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = Int8Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn i16() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = Int16Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn i32() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = Int32Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn i64() -> Result<()> {
    let data = "1,\n,\n3,";
    let expected = Int64Array::from(&[Some(1), None, Some(3)]);

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn ts_ns() -> Result<()> {
    let data = "1970-01-01T00:00:00.000000001\n";
    let expected =
        Int64Array::from_slice([1]).to(ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn ts_us() -> Result<()> {
    let data = "1970-01-01T00:00:00.000001\n";
    let expected =
        Int64Array::from_slice([1]).to(ArrowDataType::Timestamp(TimeUnit::Microsecond, None));

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn ts_ms() -> Result<()> {
    let data = "1970-01-01T00:00:00.001\n";
    let expected =
        Int64Array::from_slice([1]).to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None));

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn ts_s() -> Result<()> {
    let data = "1970-01-01T00:00:01\n";
    let expected = Int64Array::from_slice([1]).to(ArrowDataType::Timestamp(TimeUnit::Second, None));

    let result = test_deserialize(data, expected.data_type().clone())?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn date32() -> Result<()> {
    let result = test_deserialize(
        "1970-01-01,\n2020-03-15,\n1945-05-08,\n",
        ArrowDataType::Date32,
    )?;
    let expected = Int32Array::from(&[Some(0), Some(18336), Some(-9004)]).to(ArrowDataType::Date32);
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn date64() -> Result<()> {
    let input = "1970-01-01T00:00:00,\n \
        2018-11-13T17:11:10,\n \
        2018-11-13T17:11:10.011,\n \
        1900-02-28T12:34:56,\n";

    let result = test_deserialize(input, ArrowDataType::Date64)?;
    let expected = Int64Array::from(&[
        Some(0),
        Some(1542129070000),
        Some(1542129070011),
        Some(-2203932304000),
    ])
    .to(ArrowDataType::Date64);
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn time32_s() -> Result<()> {
    let result = test_deserialize(
        "00:00:00,\n23:59:59,\n11:00:11,\n",
        ArrowDataType::Time32(TimeUnit::Second),
    )?;
    let expected = Int32Array::from(&[Some(0), Some(86399), Some(39611)])
        .to(ArrowDataType::Time32(TimeUnit::Second));
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn time32_ms() -> Result<()> {
    let result = test_deserialize(
        "00:00:00.000,\n23:59:59.999,\n00:00:00.999,\n",
        ArrowDataType::Time32(TimeUnit::Millisecond),
    )?;
    let expected = Int32Array::from(&[Some(0), Some(86_399_999), Some(999)])
        .to(ArrowDataType::Time32(TimeUnit::Millisecond));
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn time64_us() -> Result<()> {
    let result = test_deserialize(
        "00:00:00.000000,\n23:59:59.999999,\n00:00:00.000001,\n",
        ArrowDataType::Time64(TimeUnit::Microsecond),
    )?;
    let expected = Int64Array::from(&[Some(0), Some(86_399_999_999), Some(1)])
        .to(ArrowDataType::Time64(TimeUnit::Microsecond));
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn time64_ns() -> Result<()> {
    let result = test_deserialize(
        "00:00:00.000000000,\n23:59:59.999999999,\n00:00:00.000000001,\n",
        ArrowDataType::Time64(TimeUnit::Nanosecond),
    )?;
    let expected = Int64Array::from(&[Some(0), Some(86_399_999_999_999), Some(1)])
        .to(ArrowDataType::Time64(TimeUnit::Nanosecond));
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn decimal() -> Result<()> {
    let result = test_deserialize("1.1,\n1.2,\n1.22,\n1.3,\n", ArrowDataType::Decimal(2, 1))?;
    let expected =
        Int128Array::from(&[Some(11), Some(12), None, Some(13)]).to(ArrowDataType::Decimal(2, 1));
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn decimal_only_scale() -> Result<()> {
    let result = test_deserialize(
        "0.01,\n0.12,\n0.222,\n0.13,\n",
        ArrowDataType::Decimal(2, 2),
    )?;
    let expected =
        Int128Array::from(&[Some(1), Some(12), None, Some(13)]).to(ArrowDataType::Decimal(2, 2));
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn decimal_only_integer() -> Result<()> {
    let result = test_deserialize("1,\n1.0,\n1.1,\n10.0,\n", ArrowDataType::Decimal(1, 0))?;
    let expected =
        Int128Array::from(&[Some(1), Some(1), None, Some(10)]).to(ArrowDataType::Decimal(1, 0));
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn boolean() -> Result<()> {
    let input = vec!["true", "True", "False", "F", "t"];
    let input = input.join("\n");

    let expected = BooleanArray::from(&[Some(true), Some(true), Some(false), None, None]);

    let result = test_deserialize(&input, ArrowDataType::Boolean)?;

    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn float32() -> Result<()> {
    let input = vec!["12.34", "12", "0.0", "inf", "-inf", "dd"];
    let input = input.join("\n");

    let expected = Float32Array::from(&[
        Some(12.34),
        Some(12.0),
        Some(0.0),
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
        None,
    ]);

    let result = test_deserialize(&input, ArrowDataType::Float32)?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn deserialize_binary() -> Result<()> {
    let input = vec!["aa", "bb"];
    let input = input.join("\n");

    let expected = BinaryArray::<i32>::from([Some(b"aa"), Some(b"bb")]);

    let result = test_deserialize(&input, ArrowDataType::Binary)?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

#[test]
fn deserialize_timestamp() -> Result<()> {
    let input = vec!["1996-12-19T16:34:57-02:00", "1996-12-19T16:34:58-02:00"];
    let input = input.join("\n");

    let data_type = ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("-01:00".to_string()));

    let expected = Int64Array::from([Some(851020497000), Some(851020498000)]).to(data_type.clone());

    let result = test_deserialize(&input, data_type)?;
    assert_eq!(expected, result.as_ref());
    Ok(())
}

proptest! {
    #[test]
    #[cfg_attr(miri, ignore)] // miri and proptest do not work well :(
    fn i64_proptest(v in any::<i64>()) {
        assert_eq!(infer(v.to_string().as_bytes()), ArrowDataType::Int64);
    }
}

proptest! {
    #[test]
    #[cfg_attr(miri, ignore)] // miri and proptest do not work well :(
    fn utf8_proptest(v in "a.*") {
        assert_eq!(infer(v.as_bytes()), ArrowDataType::Utf8);
    }
}

proptest! {
    #[test]
    #[cfg_attr(miri, ignore)] // miri and proptest do not work well :(
    fn dates(v in "1996-12-19T16:3[0-9]:57-02:00") {
        assert_eq!(infer(v.as_bytes()), ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("-02:00".to_string())));
    }
}
