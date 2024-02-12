use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::time::*;
use polars_arrow::datatypes::{DataType, TimeUnit};
use polars_arrow::scalar::*;
use polars_arrow::types::months_days_ns;

#[test]
fn test_adding_timestamp() {
    let timestamp =
        PrimitiveArray::from([Some(100000i64), Some(200000i64), None, Some(300000i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );

    let duration = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)])
        .to(DataType::Duration(TimeUnit::Second));

    let result = add_duration(&timestamp, &duration);
    let expected =
        PrimitiveArray::from([Some(100010i64), Some(200020i64), None, Some(300030i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );

    assert_eq!(result, expected);

    let duration = PrimitiveScalar::from(Some(10i64)).to(DataType::Duration(TimeUnit::Second));

    let result = add_duration_scalar(&timestamp, &duration);
    let expected =
        PrimitiveArray::from([Some(100010i64), Some(200010i64), None, Some(300010i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );
    assert_eq!(result, expected);
}

#[test]
fn test_adding_duration_different_scale() {
    let timestamp =
        PrimitiveArray::from([Some(100000i64), Some(200000i64), None, Some(300000i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );
    let expected =
        PrimitiveArray::from([Some(100010i64), Some(200020i64), None, Some(300030i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );

    // Testing duration in milliseconds
    let duration = PrimitiveArray::from([Some(10_000i64), Some(20_000i64), None, Some(30_000i64)])
        .to(DataType::Duration(TimeUnit::Millisecond));

    let result = add_duration(&timestamp, &duration);
    assert_eq!(result, expected);

    // Testing duration in nanoseconds.
    // The last digits in the nanosecond are not significant enough to
    // be added to the timestamp which is in seconds and doesn't have a
    // fractional value
    let duration = PrimitiveArray::from([
        Some(10_000_000_999i64),
        Some(20_000_000_000i64),
        None,
        Some(30_000_000_000i64),
    ])
    .to(DataType::Duration(TimeUnit::Nanosecond));

    let result = add_duration(&timestamp, &duration);
    assert_eq!(result, expected);
}

#[test]
fn test_adding_subtract_timestamps_scale() {
    let timestamp = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)]).to(
        DataType::Timestamp(TimeUnit::Millisecond, Some("America/New_York".to_string())),
    );
    let duration = PrimitiveArray::from([Some(1i64), Some(2i64), None, Some(3i64)])
        .to(DataType::Duration(TimeUnit::Second));

    let expected = PrimitiveArray::from([Some(1_010i64), Some(2_020i64), None, Some(3_030i64)]).to(
        DataType::Timestamp(TimeUnit::Millisecond, Some("America/New_York".to_string())),
    );

    let result = add_duration(&timestamp, &duration);
    assert_eq!(result, expected);

    let timestamp = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)]).to(
        DataType::Timestamp(TimeUnit::Nanosecond, Some("America/New_York".to_string())),
    );
    let duration = PrimitiveArray::from([Some(1i64), Some(2i64), None, Some(3i64)])
        .to(DataType::Duration(TimeUnit::Second));

    let expected = PrimitiveArray::from([
        Some(1_000_000_010i64),
        Some(2_000_000_020i64),
        None,
        Some(3_000_000_030i64),
    ])
    .to(DataType::Timestamp(
        TimeUnit::Nanosecond,
        Some("America/New_York".to_string()),
    ));

    let result = add_duration(&timestamp, &duration);
    assert_eq!(result, expected);
}

#[test]
fn test_subtract_timestamp() {
    let timestamp =
        PrimitiveArray::from([Some(100000i64), Some(200000i64), None, Some(300000i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );

    let duration = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)])
        .to(DataType::Duration(TimeUnit::Second));

    let result = subtract_duration(&timestamp, &duration);
    let expected =
        PrimitiveArray::from([Some(99990i64), Some(199980i64), None, Some(299970i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );

    assert_eq!(result, expected);
}

#[test]
fn test_subtracting_duration_different_scale() {
    let timestamp =
        PrimitiveArray::from([Some(100000i64), Some(200000i64), None, Some(300000i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );
    let expected =
        PrimitiveArray::from([Some(99990i64), Some(199980i64), None, Some(299970i64)]).to(
            DataType::Timestamp(TimeUnit::Second, Some("America/New_York".to_string())),
        );

    // Testing duration in milliseconds
    let duration = PrimitiveArray::from([Some(10_000i64), Some(20_000i64), None, Some(30_000i64)])
        .to(DataType::Duration(TimeUnit::Millisecond));

    let result = subtract_duration(&timestamp, &duration);
    assert_eq!(result, expected);

    // Testing duration in nanoseconds.
    // The last digits in the nanosecond are not significant enough to
    // be added to the timestamp which is in seconds and doesn't have a
    // fractional value
    let duration = PrimitiveArray::from([
        Some(10_000_000_999i64),
        Some(20_000_000_000i64),
        None,
        Some(30_000_000_000i64),
    ])
    .to(DataType::Duration(TimeUnit::Nanosecond));

    let result = subtract_duration(&timestamp, &duration);
    assert_eq!(result, expected);
}

#[test]
fn test_subtracting_subtract_timestamps_scale() {
    let timestamp = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)]).to(
        DataType::Timestamp(TimeUnit::Millisecond, Some("America/New_York".to_string())),
    );
    let duration = PrimitiveArray::from([Some(1i64), Some(2i64), None, Some(3i64)])
        .to(DataType::Duration(TimeUnit::Second));

    let expected =
        PrimitiveArray::from([Some(-990i64), Some(-1_980i64), None, Some(-2_970i64)]).to(
            DataType::Timestamp(TimeUnit::Millisecond, Some("America/New_York".to_string())),
        );

    let result = subtract_duration(&timestamp, &duration);
    assert_eq!(result, expected);

    let timestamp = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)]).to(
        DataType::Timestamp(TimeUnit::Nanosecond, Some("America/New_York".to_string())),
    );
    let duration = PrimitiveArray::from([Some(1i64), Some(2i64), None, Some(3i64)])
        .to(DataType::Duration(TimeUnit::Second));

    let expected = PrimitiveArray::from([
        Some(-999_999_990i64),
        Some(-1_999_999_980i64),
        None,
        Some(-2_999_999_970i64),
    ])
    .to(DataType::Timestamp(
        TimeUnit::Nanosecond,
        Some("America/New_York".to_string()),
    ));

    let result = subtract_duration(&timestamp, &duration);
    assert_eq!(result, expected);
}

#[test]
fn test_subtract_timestamps() {
    let timestamp_a =
        PrimitiveArray::from([Some(100_010i64), Some(200_020i64), None, Some(300_030i64)])
            .to(DataType::Timestamp(TimeUnit::Second, None));

    let timestamp_b =
        PrimitiveArray::from([Some(100_000i64), Some(200_000i64), None, Some(300_000i64)])
            .to(DataType::Timestamp(TimeUnit::Second, None));

    let expected = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)])
        .to(DataType::Duration(TimeUnit::Second));

    let result = subtract_timestamps(&timestamp_a, &timestamp_b).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_subtract_timestamps_scale() {
    let timestamp_a = PrimitiveArray::from([
        Some(100_000_000i64),
        Some(200_000_000i64),
        None,
        Some(300_000_000i64),
    ])
    .to(DataType::Timestamp(TimeUnit::Millisecond, None));

    let timestamp_b =
        PrimitiveArray::from([Some(100_010i64), Some(200_020i64), None, Some(300_030i64)])
            .to(DataType::Timestamp(TimeUnit::Second, None));

    let expected =
        PrimitiveArray::from([Some(-10_000i64), Some(-20_000i64), None, Some(-30_000i64)])
            .to(DataType::Duration(TimeUnit::Millisecond));

    let result = subtract_timestamps(&timestamp_a, &timestamp_b).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_adding_to_time() {
    let duration = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)])
        .to(DataType::Duration(TimeUnit::Second));

    // Testing Time32
    let time_32 = PrimitiveArray::from([Some(100000i32), Some(200000i32), None, Some(300000i32)])
        .to(DataType::Time32(TimeUnit::Second));

    let result = add_duration(&time_32, &duration);
    let expected = PrimitiveArray::from([Some(100010i32), Some(200020i32), None, Some(300030i32)])
        .to(DataType::Time32(TimeUnit::Second));

    assert_eq!(result, expected);
}

#[test]
fn test_subtract_to_time() {
    let duration = PrimitiveArray::from([Some(10i64), Some(20i64), None, Some(30i64)])
        .to(DataType::Duration(TimeUnit::Second));

    // Testing Time32
    let time_32 = PrimitiveArray::from([Some(100000i32), Some(200000i32), None, Some(300000i32)])
        .to(DataType::Time32(TimeUnit::Second));

    let result = subtract_duration(&time_32, &duration);
    let expected = PrimitiveArray::from([Some(99990i32), Some(199980i32), None, Some(299970i32)])
        .to(DataType::Time32(TimeUnit::Second));

    assert_eq!(result, expected);
}

#[test]
fn test_date32() {
    let duration = PrimitiveArray::from([
        Some(86_400),     // 1 day
        Some(864_000i64), // 10 days
        None,
        Some(8_640_000i64), // 100 days
    ])
    .to(DataType::Duration(TimeUnit::Second));

    let date_32 =
        PrimitiveArray::from([Some(100_000i32), Some(100_000i32), None, Some(100_000i32)])
            .to(DataType::Date32);

    let result = add_duration(&date_32, &duration);
    let expected =
        PrimitiveArray::from([Some(100_001i32), Some(100_010i32), None, Some(100_100i32)])
            .to(DataType::Date32);

    assert_eq!(result, expected);

    let result = subtract_duration(&date_32, &duration);
    let expected = PrimitiveArray::from([Some(99_999i32), Some(99_990i32), None, Some(99_900i32)])
        .to(DataType::Date32);

    assert_eq!(result, expected);
}

#[test]
fn test_date64() {
    let duration = PrimitiveArray::from([
        Some(10i64),  // 10 milliseconds
        Some(100i64), // 100 milliseconds
        None,
        Some(1_000i64), // 1000 milliseconds
    ])
    .to(DataType::Duration(TimeUnit::Millisecond));

    let date_64 =
        PrimitiveArray::from([Some(100_000i64), Some(100_000i64), None, Some(100_000i64)])
            .to(DataType::Date64);

    let result = add_duration(&date_64, &duration);
    let expected =
        PrimitiveArray::from([Some(100_010i64), Some(100_100i64), None, Some(101_000i64)])
            .to(DataType::Date64);

    assert_eq!(result, expected);

    let result = subtract_duration(&date_64, &duration);
    let expected = PrimitiveArray::from([Some(99_990i64), Some(99_900i64), None, Some(99_000i64)])
        .to(DataType::Date64);

    assert_eq!(result, expected);
}

#[test]
fn test_add_interval() {
    let timestamp =
        PrimitiveArray::from_slice([1i64]).to(DataType::Timestamp(TimeUnit::Second, None));

    let interval = months_days_ns::new(0, 1, 0);

    let intervals = PrimitiveArray::from_slice([interval]);

    let expected = PrimitiveArray::from_slice([1i64 + 24 * 60 * 60])
        .to(DataType::Timestamp(TimeUnit::Second, None));

    let result = add_interval(&timestamp, &intervals).unwrap();
    assert_eq!(result, expected);

    let result = add_interval_scalar(&timestamp, &Some(interval).into()).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_add_interval_offset() {
    let timestamp = PrimitiveArray::from_slice([1i64]).to(DataType::Timestamp(
        TimeUnit::Second,
        Some("+01:00".to_string()),
    ));

    let interval = months_days_ns::new(0, 1, 0);

    let intervals = PrimitiveArray::from_slice([interval]);

    let expected = PrimitiveArray::from_slice([1i64 + 24 * 60 * 60]).to(DataType::Timestamp(
        TimeUnit::Second,
        Some("+01:00".to_string()),
    ));

    let result = add_interval(&timestamp, &intervals).unwrap();
    assert_eq!(result, expected);

    let result = add_interval_scalar(&timestamp, &Some(interval).into()).unwrap();
    assert_eq!(result, expected);
}

#[cfg(feature = "chrono-tz")]
#[test]
fn test_add_interval_tz() {
    let timestamp = PrimitiveArray::from_slice([1i64]).to(DataType::Timestamp(
        TimeUnit::Second,
        Some("GMT".to_string()),
    ));

    let interval = months_days_ns::new(0, 1, 0);
    let intervals = PrimitiveArray::from_slice([interval]);

    let expected = PrimitiveArray::from_slice([1i64 + 24 * 60 * 60]).to(DataType::Timestamp(
        TimeUnit::Second,
        Some("GMT".to_string()),
    ));

    let result = add_interval(&timestamp, &intervals).unwrap();
    assert_eq!(result, expected);

    let result = add_interval_scalar(&timestamp, &Some(interval).into()).unwrap();
    assert_eq!(result, expected);
}
