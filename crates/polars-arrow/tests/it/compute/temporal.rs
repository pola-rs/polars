use polars_arrow::array::*;
use polars_arrow::compute::temporal::*;
use polars_arrow::datatypes::*;

macro_rules! temporal_test {
    ($func:ident, $extract:ident, $data_types:path) => {
        #[test]
        fn $func() {
            for data_type in $data_types() {
                let data = TestData::data(&data_type);
                let result = $extract(&*data.input).unwrap();

                assert_eq!(
                    result,
                    data.$extract.unwrap(),
                    "\"{}\" failed on type: {:?}",
                    stringify!($extract),
                    data_type
                );
            }
        }
    };
}

temporal_test!(temporal_hour, hour, TestData::available_time_like_types);
temporal_test!(temporal_minute, minute, TestData::available_time_like_types);
temporal_test!(temporal_second, second, TestData::available_time_like_types);
temporal_test!(
    temporal_nanosecond,
    nanosecond,
    TestData::available_time_like_types
);
temporal_test!(temporal_year, year, TestData::available_date_like_types);
temporal_test!(temporal_month, month, TestData::available_date_like_types);
temporal_test!(temporal_day, day, TestData::available_date_like_types);
temporal_test!(
    temporal_weekday,
    weekday,
    TestData::available_date_like_types
);
temporal_test!(
    temporal_iso_week,
    iso_week,
    TestData::available_date_like_types
);

struct TestData {
    input: Box<dyn Array>,
    year: Option<Int32Array>,
    month: Option<UInt32Array>,
    day: Option<UInt32Array>,
    weekday: Option<UInt32Array>,
    iso_week: Option<UInt32Array>,
    hour: Option<UInt32Array>,
    minute: Option<UInt32Array>,
    second: Option<UInt32Array>,
    nanosecond: Option<UInt32Array>,
}

impl TestData {
    fn data(data_type: &DataType) -> TestData {
        match data_type {
            DataType::Date64 => TestData {
                input: Box::new(
                    Int64Array::from(&[Some(1514764800000), None, Some(1550636625000)])
                        .to(data_type.clone()),
                ),
                year: Some(Int32Array::from(&[Some(2018), None, Some(2019)])),
                month: Some(UInt32Array::from(&[Some(1), None, Some(2)])),
                day: Some(UInt32Array::from(&[Some(1), None, Some(20)])),
                weekday: Some(UInt32Array::from(&[Some(1), None, Some(3)])),
                iso_week: Some(UInt32Array::from(&[Some(1), None, Some(8)])),
                hour: Some(UInt32Array::from(&[Some(0), None, Some(4)])),
                minute: Some(UInt32Array::from(&[Some(0), None, Some(23)])),
                second: Some(UInt32Array::from(&[Some(0), None, Some(45)])),
                nanosecond: Some(UInt32Array::from(&[Some(0), None, Some(0)])),
            },
            DataType::Date32 => TestData {
                input: Box::new(Int32Array::from(&[Some(15147), None]).to(data_type.clone())),
                year: Some(Int32Array::from(&[Some(2011), None])),
                month: Some(UInt32Array::from(&[Some(6), None])),
                day: Some(UInt32Array::from(&[Some(22), None])),
                weekday: Some(UInt32Array::from(&[Some(3), None])),
                iso_week: Some(UInt32Array::from(&[Some(25), None])),
                hour: Some(UInt32Array::from(&[Some(0), None])),
                minute: Some(UInt32Array::from(&[Some(0), None])),
                second: Some(UInt32Array::from(&[Some(0), None])),
                nanosecond: Some(UInt32Array::from(&[Some(0), None])),
            },
            DataType::Time32(TimeUnit::Second) => TestData {
                input: Box::new(Int32Array::from(&[Some(37800), None]).to(data_type.clone())),
                year: None,
                month: None,
                day: None,
                weekday: None,
                iso_week: None,
                hour: Some(UInt32Array::from(&[Some(10), None])),
                minute: Some(UInt32Array::from(&[Some(30), None])),
                second: Some(UInt32Array::from(&[Some(0), None])),
                nanosecond: Some(UInt32Array::from(&[Some(0), None])),
            },
            DataType::Time64(TimeUnit::Microsecond) => TestData {
                input: Box::new(Int64Array::from(&[Some(378000000), None]).to(data_type.clone())),
                year: None,
                month: None,
                day: None,
                weekday: None,
                iso_week: None,
                hour: Some(UInt32Array::from(&[Some(0), None])),
                minute: Some(UInt32Array::from(&[Some(6), None])),
                second: Some(UInt32Array::from(&[Some(18), None])),
                nanosecond: Some(UInt32Array::from(&[Some(0), None])),
            },
            DataType::Time64(TimeUnit::Nanosecond) => TestData {
                input: Box::new(
                    Int64Array::from(&[Some(378000000100), None]).to(data_type.clone()),
                ),
                year: None,
                month: None,
                day: None,
                weekday: None,
                iso_week: None,
                hour: Some(UInt32Array::from(&[Some(0), None])),
                minute: Some(UInt32Array::from(&[Some(6), None])),
                second: Some(UInt32Array::from(&[Some(18), None])),
                nanosecond: Some(UInt32Array::from(&[Some(100), None])),
            },
            DataType::Timestamp(TimeUnit::Microsecond, None) => TestData {
                // 68216970000000 (Epoch Microsecond) has 29th Feb (leap year)
                input: Box::new(
                    Int64Array::from(&[Some(1612025847000000), None, Some(68216970000000)])
                        .to(data_type.clone()),
                ),
                year: Some(Int32Array::from(&[Some(2021), None, Some(1972)])),
                month: Some(UInt32Array::from(&[Some(1), None, Some(2)])),
                day: Some(UInt32Array::from(&[Some(30), None, Some(29)])),
                weekday: Some(UInt32Array::from(&[Some(6), None, Some(2)])),
                iso_week: Some(UInt32Array::from(&[Some(4), None, Some(9)])),
                hour: Some(UInt32Array::from(&[Some(16), None, Some(13)])),
                minute: Some(UInt32Array::from(&[Some(57), None, Some(9)])),
                second: Some(UInt32Array::from(&[Some(27), None, Some(30)])),
                nanosecond: Some(UInt32Array::from(&[Some(0), None, Some(0)])),
            },
            _ => unreachable!(),
        }
    }

    fn available_time_like_types() -> Vec<DataType> {
        vec![
            DataType::Date32,
            DataType::Date64,
            DataType::Time32(TimeUnit::Second),
            DataType::Time64(TimeUnit::Microsecond),
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ]
    }

    fn available_date_like_types() -> Vec<DataType> {
        vec![
            DataType::Date32,
            DataType::Date64,
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ]
    }
}

macro_rules! temporal_tz_test {
    ($func:ident, $extract:ident) => {
        #[cfg(feature = "chrono-tz")]
        #[test]
        fn $func() {
            let test_data = test_data_tz();

            for data in test_data {
                let result = $extract(&*data.input).unwrap();

                assert_eq!(result, data.$extract.unwrap());
            }
        }
    };
}

temporal_tz_test!(temporal_tz_hour, hour);
temporal_tz_test!(temporal_tz_minute, minute);
temporal_tz_test!(temporal_tz_second, second);
temporal_tz_test!(temporal_tz_nanosecond, nanosecond);
temporal_tz_test!(temporal_tz_year, year);
temporal_tz_test!(temporal_tz_month, month);
temporal_tz_test!(temporal_tz_day, day);
temporal_tz_test!(temporal_tz_weekday, weekday);
temporal_tz_test!(temporal_tz_iso_week, iso_week);

fn test_data_tz() -> Vec<TestData> {
    vec![
        TestData {
            input: Box::new(
                // Mon May 24 2021 17:25:30 GMT+0000
                Int64Array::from(&[Some(1621877130000000), None]).to(DataType::Timestamp(
                    TimeUnit::Microsecond,
                    Some("GMT".to_string()),
                )),
            ),
            year: Some(Int32Array::from(&[Some(2021), None])),
            month: Some(UInt32Array::from(&[Some(5), None])),
            day: Some(UInt32Array::from(&[Some(24), None])),
            weekday: Some(UInt32Array::from(&[Some(1), None])),
            iso_week: Some(UInt32Array::from(&[Some(21), None])),
            hour: Some(UInt32Array::from(&[Some(17), None])),
            minute: Some(UInt32Array::from(&[Some(25), None])),
            second: Some(UInt32Array::from(&[Some(30), None])),
            nanosecond: Some(UInt32Array::from(&[Some(0), None])),
        },
        TestData {
            input: Box::new(Int64Array::from(&[Some(1621877130000000), None]).to(
                DataType::Timestamp(TimeUnit::Microsecond, Some("+01:00".to_string())),
            )),
            year: Some(Int32Array::from(&[Some(2021), None])),
            month: Some(UInt32Array::from(&[Some(5), None])),
            day: Some(UInt32Array::from(&[Some(24), None])),
            weekday: Some(UInt32Array::from(&[Some(1), None])),
            iso_week: Some(UInt32Array::from(&[Some(21), None])),
            hour: Some(UInt32Array::from(&[Some(18), None])),
            minute: Some(UInt32Array::from(&[Some(25), None])),
            second: Some(UInt32Array::from(&[Some(30), None])),
            nanosecond: Some(UInt32Array::from(&[Some(0), None])),
        },
        TestData {
            input: Box::new(Int64Array::from(&[Some(1621877130000000), None]).to(
                DataType::Timestamp(TimeUnit::Microsecond, Some("Europe/Lisbon".to_string())),
            )),
            year: Some(Int32Array::from(&[Some(2021), None])),
            month: Some(UInt32Array::from(&[Some(5), None])),
            day: Some(UInt32Array::from(&[Some(24), None])),
            weekday: Some(UInt32Array::from(&[Some(1), None])),
            iso_week: Some(UInt32Array::from(&[Some(21), None])),
            hour: Some(UInt32Array::from(&[Some(18), None])),
            minute: Some(UInt32Array::from(&[Some(25), None])),
            second: Some(UInt32Array::from(&[Some(30), None])),
            nanosecond: Some(UInt32Array::from(&[Some(0), None])),
        },
        TestData {
            input: Box::new(
                // Sun Mar 29 2020 00:00:00 GMT+0000 (Western European Standard Time)
                Int64Array::from(&[Some(1585440000), None]).to(DataType::Timestamp(
                    TimeUnit::Second,
                    Some("Europe/Lisbon".to_string()),
                )),
            ),
            year: Some(Int32Array::from(&[Some(2020), None])),
            month: Some(UInt32Array::from(&[Some(3), None])),
            day: Some(UInt32Array::from(&[Some(29), None])),
            weekday: Some(UInt32Array::from(&[Some(7), None])),
            iso_week: Some(UInt32Array::from(&[Some(13), None])),
            hour: Some(UInt32Array::from(&[Some(0), None])),
            minute: Some(UInt32Array::from(&[Some(0), None])),
            second: Some(UInt32Array::from(&[Some(0), None])),
            nanosecond: Some(UInt32Array::from(&[Some(0), None])),
        },
        TestData {
            input: Box::new(
                // Sun Mar 29 2020 02:00:00 GMT+0100 (Western European Summer Time)
                Int64Array::from(&[Some(1585443600), None]).to(DataType::Timestamp(
                    TimeUnit::Second,
                    Some("Europe/Lisbon".to_string()),
                )),
            ),
            year: Some(Int32Array::from(&[Some(2020), None])),
            month: Some(UInt32Array::from(&[Some(3), None])),
            day: Some(UInt32Array::from(&[Some(29), None])),
            weekday: Some(UInt32Array::from(&[Some(7), None])),
            iso_week: Some(UInt32Array::from(&[Some(13), None])),
            hour: Some(UInt32Array::from(&[Some(2), None])),
            minute: Some(UInt32Array::from(&[Some(0), None])),
            second: Some(UInt32Array::from(&[Some(0), None])),
            nanosecond: Some(UInt32Array::from(&[Some(0), None])),
        },
    ]
}

#[test]
fn consistency_hour() {
    consistency_check(can_hour, hour);
}

#[test]
fn consistency_minute() {
    consistency_check(can_minute, minute);
}

#[test]
fn consistency_second() {
    consistency_check(can_second, second);
}

#[test]
fn consistency_nanosecond() {
    consistency_check(can_nanosecond, nanosecond);
}

#[test]
fn consistency_year() {
    consistency_check(can_year, year);
}

#[test]
fn consistency_month() {
    consistency_check(can_month, month);
}

#[test]
fn consistency_day() {
    consistency_check(can_day, day);
}

#[test]
fn consistency_weekday() {
    consistency_check(can_weekday, weekday);
}

#[test]
fn consistency_iso_week() {
    consistency_check(can_iso_week, iso_week);
}

fn consistency_check<O: polars_arrow::types::NativeType>(
    can_extract: fn(&DataType) -> bool,
    extract: fn(&dyn Array) -> polars_arrow::error::Result<PrimitiveArray<O>>,
) {
    use polars_arrow::datatypes::DataType::*;

    let datatypes = vec![
        Null,
        Boolean,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float32,
        Float64,
        Timestamp(TimeUnit::Second, None),
        Timestamp(TimeUnit::Millisecond, None),
        Timestamp(TimeUnit::Microsecond, None),
        Timestamp(TimeUnit::Nanosecond, None),
        Timestamp(TimeUnit::Nanosecond, Some("+00:00".to_string())),
        Time64(TimeUnit::Microsecond),
        Time64(TimeUnit::Nanosecond),
        Date32,
        Time32(TimeUnit::Second),
        Time32(TimeUnit::Millisecond),
        Date64,
        Utf8,
        LargeUtf8,
        Binary,
        LargeBinary,
        Duration(TimeUnit::Second),
        Duration(TimeUnit::Millisecond),
        Duration(TimeUnit::Microsecond),
        Duration(TimeUnit::Nanosecond),
    ];

    datatypes.into_iter().for_each(|d1| {
        let array = new_null_array(d1.clone(), 10);
        if can_extract(&d1) {
            assert!(extract(array.as_ref()).is_ok());
        } else {
            assert!(extract(array.as_ref()).is_err());
        }
    });
}
