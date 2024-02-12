use std::io::Cursor;

use polars_arrow::array::*;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::*;
use polars_arrow::error::Result;
use polars_arrow::io::csv::write::*;

fn data() -> Chunk<Box<dyn Array>> {
    let c1 = Utf8Array::<i32>::from_slice(["a b", "c", "d"]);
    let c2 = Float64Array::from([Some(123.564532), None, Some(-556132.25)]);
    let c3 = UInt32Array::from_slice([3, 2, 1]);
    let c4 = BooleanArray::from(&[Some(true), Some(false), None]);
    let c5 = PrimitiveArray::<i64>::from([None, Some(1555584887378), Some(1555555555555)])
        .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None));
    let c6 = PrimitiveArray::<i32>::from_vec(vec![1234, 24680, 85563])
        .to(ArrowDataType::Time32(TimeUnit::Second));
    let keys = UInt32Array::from_slice([2, 0, 1]);
    let c7 = DictionaryArray::try_from_keys(keys, Box::new(c1.clone())).unwrap();

    Chunk::new(vec![
        Box::new(c1) as Box<dyn Array>,
        Box::new(c2),
        Box::new(c3),
        Box::new(c4),
        Box::new(c5),
        Box::new(c6),
        Box::new(c7),
    ])
}

#[test]
fn write_csv() -> Result<()> {
    let columns = data();

    let mut writer = Cursor::new(Vec::<u8>::new());
    let options = SerializeOptions::default();

    write_header(
        &mut writer,
        &["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
        &options,
    )?;
    write_chunk(&mut writer, &columns, &options)?;

    // check
    let buffer = writer.into_inner();
    assert_eq!(
        r#"c1,c2,c3,c4,c5,c6,c7
a b,123.564532,3,true,,00:20:34,d
c,,2,false,2019-04-18 10:54:47.378,06:51:20,a b
d,-556132.25,1,,2019-04-18 02:45:55.555,23:46:03,c
"#
        .to_string(),
        String::from_utf8(buffer).unwrap(),
    );
    Ok(())
}

#[test]
fn write_csv_custom_options() -> Result<()> {
    let batch = data();

    let mut writer = Cursor::new(Vec::<u8>::new());

    let options = SerializeOptions {
        time32_format: Some("%r".to_string()),
        time64_format: Some("%r".to_string()),
        delimiter: b'|',
        ..Default::default()
    };
    write_chunk(&mut writer, &batch, &options)?;

    // check
    let buffer = writer.into_inner();
    assert_eq!(
        r#"a b|123.564532|3|true||12:20:34 AM|d
c||2|false|2019-04-18 10:54:47.378|06:51:20 AM|a b
d|-556132.25|1||2019-04-18 02:45:55.555|11:46:03 PM|c
"#
        .to_string(),
        String::from_utf8(buffer).unwrap(),
    );
    Ok(())
}

fn data_array(column: &str) -> (Chunk<Box<dyn Array>>, Vec<&'static str>) {
    let (array, expected) = match column {
        "utf8" => (
            Utf8Array::<i32>::from_slice(["a b", "c", "d"]).boxed(),
            vec!["a b", "c", "d"],
        ),
        "large_utf8" => (
            Utf8Array::<i64>::from_slice(["a b", "c", "d"]).boxed(),
            vec!["a b", "c", "d"],
        ),
        "binary" => (
            BinaryArray::<i32>::from_slice(["a b", "c", "d"]).boxed(),
            vec!["a b", "c", "d"],
        ),
        "large_binary" => (
            BinaryArray::<i64>::from_slice(["a b", "c", "d"]).boxed(),
            vec!["a b", "c", "d"],
        ),
        "i8" => (
            Int8Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "i16" => (
            Int16Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "i32" => (
            Int32Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "i64" => (
            Int64Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "u8" => (
            UInt8Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "u16" => (
            UInt16Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "u32" => (
            UInt32Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "u64" => (
            UInt64Array::from_slice([3, 2, 1]).boxed(),
            vec!["3", "2", "1"],
        ),
        "f32" => (Float32Array::from_slice([3.1]).boxed(), vec!["3.1"]),
        "f64" => (Float64Array::from_slice([3.1]).boxed(), vec!["3.1"]),
        "date32" => {
            let array = PrimitiveArray::<i32>::from_vec(vec![1]).to(ArrowDataType::Date32);
            (array.boxed(), vec!["1970-01-02 00:00:00"])
        },
        "date64" => {
            let array = PrimitiveArray::<i64>::from_vec(vec![1_000]).to(ArrowDataType::Date64);
            (array.boxed(), vec!["1970-01-01 00:00:01"])
        },
        "time32[ms]" => {
            let array = PrimitiveArray::<i32>::from_vec(vec![1_234_001, 24_680_001, 85_563_001])
                .to(ArrowDataType::Time32(TimeUnit::Millisecond));
            (
                array.boxed(),
                vec!["00:20:34.001", "06:51:20.001", "23:46:03.001"],
            )
        },
        "time64[us]" => {
            let array = PrimitiveArray::<i64>::from_vec(vec![
                1_234_000_001,
                24_680_000_001,
                85_563_000_001,
            ])
            .to(ArrowDataType::Time64(TimeUnit::Microsecond));
            (
                array.boxed(),
                vec!["00:20:34.000001", "06:51:20.000001", "23:46:03.000001"],
            )
        },
        "time64[ns]" => {
            let array = PrimitiveArray::<i64>::from_vec(vec![
                1_234_000_000_001,
                24_680_000_000_001,
                85_563_000_000_001,
            ])
            .to(ArrowDataType::Time64(TimeUnit::Nanosecond));
            (
                array.boxed(),
                vec![
                    "00:20:34.000000001",
                    "06:51:20.000000001",
                    "23:46:03.000000001",
                ],
            )
        },
        "ts[s]" => {
            let array = PrimitiveArray::<i64>::from_slice([1_555_584_887, 1_555_555_555])
                .to(ArrowDataType::Timestamp(TimeUnit::Second, None));
            (
                array.boxed(),
                vec!["2019-04-18 10:54:47", "2019-04-18 02:45:55"],
            )
        },
        "ts[ms]" => {
            let array = PrimitiveArray::<i64>::from_slice([1_555_584_887_378, 1_555_555_555_555])
                .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None));
            (
                array.boxed(),
                vec!["2019-04-18 10:54:47.378", "2019-04-18 02:45:55.555"],
            )
        },
        "ts[us]" => {
            let array =
                PrimitiveArray::<i64>::from_slice([1_555_584_887_378_001, 1_555_555_555_555_001])
                    .to(ArrowDataType::Timestamp(TimeUnit::Microsecond, None));
            (
                array.boxed(),
                vec!["2019-04-18 10:54:47.378001", "2019-04-18 02:45:55.555001"],
            )
        },
        "ts[ns]" => {
            let array = PrimitiveArray::<i64>::from_slice([
                1_555_584_887_378_000_001,
                1_555_555_555_555_000_001,
            ])
            .to(ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));
            (
                array.boxed(),
                vec![
                    "2019-04-18 10:54:47.378000001",
                    "2019-04-18 02:45:55.555000001",
                ],
            )
        },
        "ts[ns,offset]" => {
            let array = PrimitiveArray::<i64>::from_slice([
                1_555_584_887_378_000_001,
                1_555_555_555_555_000_001,
            ])
            .to(ArrowDataType::Timestamp(
                TimeUnit::Nanosecond,
                Some("+01:00".to_string()),
            ));
            (
                array.boxed(),
                vec![
                    "2019-04-18 11:54:47.378000001 +01:00",
                    "2019-04-18 03:45:55.555000001 +01:00",
                ],
            )
        },
        "ts[ns,tz]" => {
            let array = PrimitiveArray::<i64>::from_slice([
                1_555_584_887_378_000_001,
                1_555_555_555_555_000_001,
            ])
            .to(ArrowDataType::Timestamp(
                TimeUnit::Nanosecond,
                Some("Europe/Lisbon".to_string()),
            ));
            (
                array.boxed(),
                vec![
                    "2019-04-18 11:54:47.378000001 WEST",
                    "2019-04-18 03:45:55.555000001 WEST",
                ],
            )
        },
        "dictionary[u32]" => {
            let keys = UInt32Array::from_slice([2, 1, 0]);
            let values = Utf8Array::<i64>::from_slice(["a b", "c", "d"]).boxed();
            let array = DictionaryArray::try_from_keys(keys, values).unwrap();
            (array.boxed(), vec!["d", "c", "a b"])
        },
        "dictionary[u64]" => {
            let keys = UInt64Array::from_slice([2, 1, 0]);
            let values = Utf8Array::<i64>::from_slice(["a b", "c", "d"]).boxed();
            let array = DictionaryArray::try_from_keys(keys, values).unwrap();
            (array.boxed(), vec!["d", "c", "a b"])
        },
        _ => todo!(),
    };

    (Chunk::new(vec![array]), expected)
}

fn test_array(
    chunk: Chunk<Box<dyn Array>>,
    data: Vec<&'static str>,
    options: SerializeOptions,
) -> Result<()> {
    let mut writer = Cursor::new(Vec::<u8>::new());

    write_header(&mut writer, &["c1"], &options)?;
    write_chunk(&mut writer, &chunk, &options)?;

    // check
    let buffer = writer.into_inner();

    let mut expected = "c1\n".to_owned();
    expected.push_str(&data.join("\n"));
    expected.push('\n');
    assert_eq!(expected, String::from_utf8(buffer).unwrap());
    Ok(())
}

fn write_single(column: &str) -> Result<()> {
    let (chunk, data) = data_array(column);

    test_array(chunk, data, SerializeOptions::default())
}

#[test]
fn write_each() -> Result<()> {
    for i in [
        "utf8",
        "large_utf8",
        "binary",
        "large_binary",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "f32",
        "f64",
        "date32",
        "date64",
        "time32[ms]",
        "time64[us]",
        "time64[ns]",
        "ts[s]",
        "ts[ms]",
        "ts[us]",
        "ts[ns]",
        "ts[ns,offset]",
        "dictionary[u32]",
        "dictionary[u64]",
    ] {
        write_single(i)?;
    }
    Ok(())
}

#[test]
#[cfg(feature = "chrono-tz")]
fn write_tz_timezone() -> Result<()> {
    write_single("ts[ns,tz]")
}

#[test]
fn write_tz_timezone_formatted_offset() -> Result<()> {
    let array =
        PrimitiveArray::<i64>::from_slice([1_555_584_887_378_000_001, 1_555_555_555_555_000_001])
            .to(ArrowDataType::Timestamp(
                TimeUnit::Nanosecond,
                Some("+01:00".to_string()),
            ));

    let columns = Chunk::new(vec![array.boxed()]);
    let expected = vec![
        "2019-04-18T11:54:47.378000001+01:00",
        "2019-04-18T03:45:55.555000001+01:00",
    ];
    test_array(
        columns,
        expected,
        SerializeOptions {
            timestamp_format: Some("%Y-%m-%dT%H:%M:%S%.f%:z".to_string()),
            ..Default::default()
        },
    )
}

#[test]
#[cfg(feature = "chrono-tz")]
fn write_tz_timezone_formatted_tz() -> Result<()> {
    let array =
        PrimitiveArray::<i64>::from_slice([1_555_584_887_378_000_001, 1_555_555_555_555_000_001])
            .to(ArrowDataType::Timestamp(
                TimeUnit::Nanosecond,
                Some("Europe/Lisbon".to_string()),
            ));

    let columns = Chunk::new(vec![array.boxed()]);
    let expected = vec![
        "2019-04-18T11:54:47.378000001+01:00",
        "2019-04-18T03:45:55.555000001+01:00",
    ];
    test_array(
        columns,
        expected,
        SerializeOptions {
            timestamp_format: Some("%Y-%m-%dT%H:%M:%S%.f%:z".to_string()),
            ..Default::default()
        },
    )
}

fn test_generic(chunk: Chunk<Box<dyn Array>>, expected: &str) {
    let mut writer = vec![];
    let options = SerializeOptions::default();
    write_chunk(&mut writer, &chunk, &options).unwrap();
    let csv = std::str::from_utf8(&writer).unwrap();

    assert_eq!(csv, expected);
}

#[test]
fn write_empty_and_missing() {
    let a = Utf8Array::<i32>::from([Some(""), None]);
    let b = Utf8Array::<i32>::from([None, Some("")]);
    let chunk = Chunk::new(vec![a.boxed(), b.boxed()]);
    test_generic(chunk, "\"\",\n,\"\"\n");
}

#[test]
fn write_escaping_resize_local_buf() {
    // tests if local buffer reallocates properly
    for payload in [
        "Acme co., Ltd.",
        "bar,1234567890123456789012345678901234567890123456789012345678900293480293847",
        "This is the mail system at host smtp.sciprofiles.com.I'm sorry to have to inform you that your message could notbe delivered to one or more recipients. It's attached below.For further assistance,bar",
    ] {
        let a = Utf8Array::<i32>::from_slice([payload]);
        let chunk = Chunk::new(vec![a.boxed()]);

        test_generic(chunk,  &format!("\"{payload}\"\n"));

        let a = Utf8Array::<i64>::from_slice([payload]);
        let chunk = Chunk::new(vec![a.boxed()]);

        test_generic(chunk,  &format!("\"{payload}\"\n"));
    }
}

#[test]
fn serialize_vec() -> Result<()> {
    let columns = data();

    let options = SerializeOptions::default();

    let data = serialize(&columns, &options)?;

    // check
    assert_eq!(
        vec![
            b"a b,123.564532,3,true,,00:20:34,d\n".to_vec(),
            b"c,,2,false,2019-04-18 10:54:47.378,06:51:20,a b\n".to_vec(),
            b"d,-556132.25,1,,2019-04-18 02:45:55.555,23:46:03,c\n".to_vec(),
        ],
        data
    );
    Ok(())
}
