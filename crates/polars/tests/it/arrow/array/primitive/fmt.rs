use arrow::array::*;
use arrow::datatypes::*;
use arrow::types::{days_ms, months_days_ns};

#[test]
fn debug_int32() {
    let array = Int32Array::from(&[Some(1), None, Some(2)]);
    assert_eq!(format!("{array:?}"), "Int32[1, None, 2]");
}

#[test]
fn debug_date32() {
    let array = Int32Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Date32);
    assert_eq!(format!("{array:?}"), "Date32[1970-01-02, None, 1970-01-03]");
}

#[test]
fn debug_time32s() {
    let array =
        Int32Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Time32(TimeUnit::Second));
    assert_eq!(
        format!("{array:?}"),
        "Time32(Second)[00:00:01, None, 00:00:02]"
    );
}

#[test]
fn debug_time32ms() {
    let array = Int32Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Time32(TimeUnit::Millisecond));
    assert_eq!(
        format!("{array:?}"),
        "Time32(Millisecond)[00:00:00.001, None, 00:00:00.002]"
    );
}

#[test]
fn debug_interval_d() {
    let array = Int32Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Interval(IntervalUnit::YearMonth));
    assert_eq!(format!("{array:?}"), "Interval(YearMonth)[1m, None, 2m]");
}

#[test]
fn debug_int64() {
    let array = Int64Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Int64);
    assert_eq!(format!("{array:?}"), "Int64[1, None, 2]");
}

#[test]
fn debug_date64() {
    let array = Int64Array::from(&[Some(1), None, Some(86400000)]).to(ArrowDataType::Date64);
    assert_eq!(format!("{array:?}"), "Date64[1970-01-01, None, 1970-01-02]");
}

#[test]
fn debug_time64us() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Time64(TimeUnit::Microsecond));
    assert_eq!(
        format!("{array:?}"),
        "Time64(Microsecond)[00:00:00.000001, None, 00:00:00.000002]"
    );
}

#[test]
fn debug_time64ns() {
    let array =
        Int64Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Time64(TimeUnit::Nanosecond));
    assert_eq!(
        format!("{array:?}"),
        "Time64(Nanosecond)[00:00:00.000000001, None, 00:00:00.000000002]"
    );
}

#[test]
fn debug_timestamp_s() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Timestamp(TimeUnit::Second, None));
    assert_eq!(
        format!("{array:?}"),
        "Timestamp(Second, None)[1970-01-01 00:00:01, None, 1970-01-01 00:00:02]"
    );
}

#[test]
fn debug_timestamp_ms() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None));
    assert_eq!(
        format!("{array:?}"),
        "Timestamp(Millisecond, None)[1970-01-01 00:00:00.001, None, 1970-01-01 00:00:00.002]"
    );
}

#[test]
fn debug_timestamp_us() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Timestamp(TimeUnit::Microsecond, None));
    assert_eq!(
        format!("{array:?}"),
        "Timestamp(Microsecond, None)[1970-01-01 00:00:00.000001, None, 1970-01-01 00:00:00.000002]"
    );
}

#[test]
fn debug_timestamp_ns() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));
    assert_eq!(
        format!("{array:?}"),
        "Timestamp(Nanosecond, None)[1970-01-01 00:00:00.000000001, None, 1970-01-01 00:00:00.000000002]"
    );
}

#[test]
fn debug_timestamp_tz_ns() {
    let array = Int64Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Timestamp(
        TimeUnit::Nanosecond,
        Some("+02:00".to_string()),
    ));
    assert_eq!(
        format!("{array:?}"),
        "Timestamp(Nanosecond, Some(\"+02:00\"))[1970-01-01 02:00:00.000000001 +02:00, None, 1970-01-01 02:00:00.000000002 +02:00]"
    );
}

#[test]
fn debug_timestamp_tz_not_parsable() {
    let array = Int64Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Timestamp(
        TimeUnit::Nanosecond,
        Some("aa".to_string()),
    ));
    assert_eq!(
        format!("{array:?}"),
        "Timestamp(Nanosecond, Some(\"aa\"))[1 (aa), None, 2 (aa)]"
    );
}

#[cfg(feature = "timezones")]
#[test]
fn debug_timestamp_tz1_ns() {
    let array = Int64Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Timestamp(
        TimeUnit::Nanosecond,
        Some("Europe/Lisbon".to_string()),
    ));
    assert_eq!(
        format!("{array:?}"),
        "Timestamp(Nanosecond, Some(\"Europe/Lisbon\"))[1970-01-01 01:00:00.000000001 CET, None, 1970-01-01 01:00:00.000000002 CET]"
    );
}

#[test]
fn debug_duration_ms() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Duration(TimeUnit::Millisecond));
    assert_eq!(
        format!("{array:?}"),
        "Duration(Millisecond)[1ms, None, 2ms]"
    );
}

#[test]
fn debug_duration_s() {
    let array =
        Int64Array::from(&[Some(1), None, Some(2)]).to(ArrowDataType::Duration(TimeUnit::Second));
    assert_eq!(format!("{array:?}"), "Duration(Second)[1s, None, 2s]");
}

#[test]
fn debug_duration_us() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Duration(TimeUnit::Microsecond));
    assert_eq!(
        format!("{array:?}"),
        "Duration(Microsecond)[1us, None, 2us]"
    );
}

#[test]
fn debug_duration_ns() {
    let array = Int64Array::from(&[Some(1), None, Some(2)])
        .to(ArrowDataType::Duration(TimeUnit::Nanosecond));
    assert_eq!(format!("{array:?}"), "Duration(Nanosecond)[1ns, None, 2ns]");
}

#[test]
fn debug_decimal() {
    let array =
        Int128Array::from(&[Some(12345), None, Some(23456)]).to(ArrowDataType::Decimal(5, 2));
    assert_eq!(format!("{array:?}"), "Decimal(5, 2)[123.45, None, 234.56]");
}

#[test]
fn debug_decimal1() {
    let array =
        Int128Array::from(&[Some(12345), None, Some(23456)]).to(ArrowDataType::Decimal(5, 1));
    assert_eq!(format!("{array:?}"), "Decimal(5, 1)[1234.5, None, 2345.6]");
}

#[test]
fn debug_interval_days_ms() {
    let array = DaysMsArray::from(&[Some(days_ms::new(1, 1)), None, Some(days_ms::new(2, 2))]);
    assert_eq!(
        format!("{array:?}"),
        "Interval(DayTime)[1d1ms, None, 2d2ms]"
    );
}

#[test]
fn debug_months_days_ns() {
    let data = &[
        Some(months_days_ns::new(1, 1, 2)),
        None,
        Some(months_days_ns::new(2, 3, 3)),
    ];

    let array = MonthsDaysNsArray::from(&data);

    assert_eq!(
        format!("{array:?}"),
        "Interval(MonthDayNano)[1m1d2ns, None, 2m3d3ns]"
    );
}
