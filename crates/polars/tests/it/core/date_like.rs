use super::*;

#[test]
#[cfg(feature = "dtype-datetime")]
#[cfg_attr(miri, ignore)]
fn test_datelike_join() -> PolarsResult<()> {
    let s = Series::new("foo", &[1, 2, 3]);
    let mut s1 = s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;
    s1.rename("bar");

    let df = DataFrame::new(vec![s, s1])?;

    let out = df.left_join(&df.clone(), ["bar"], ["bar"])?;
    assert!(matches!(
        out.column("bar")?.dtype(),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    ));

    let out = df.inner_join(&df.clone(), ["bar"], ["bar"])?;
    assert!(matches!(
        out.column("bar")?.dtype(),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    ));

    let out = df.full_join(&df.clone(), ["bar"], ["bar"])?;
    assert!(matches!(
        out.column("bar")?.dtype(),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    ));
    Ok(())
}

#[test]
#[cfg(all(feature = "dtype-datetime", feature = "dtype-duration"))]
fn test_datelike_methods() -> PolarsResult<()> {
    let s = Series::new("foo", &[1, 2, 3]);
    let s = s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;

    let out = s.subtract(&s)?;
    assert!(matches!(
        out.dtype(),
        DataType::Duration(TimeUnit::Nanoseconds)
    ));

    let mut a = s.clone();
    a.append(&s).unwrap();
    assert_eq!(a.len(), 6);

    Ok(())
}

#[test]
#[cfg(all(feature = "dtype-datetime", feature = "dtype-duration"))]
fn test_arithmetic_dispatch() {
    let s = Int64Chunked::new("", &[1, 2, 3])
        .into_datetime(TimeUnit::Nanoseconds, None)
        .into_series();

    // check if we don't panic.
    let out = &s * 100;
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = &s / 100;
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = &s + 100;
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = &s - 100;
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = &s % 100;
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );

    let out = 100.mul(&s);
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = 100.div(&s);
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = 100.sub(&s);
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = 100.add(&s);
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let out = 100.rem(&s);
    assert_eq!(
        out.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
}

#[test]
#[cfg(feature = "dtype-duration")]
fn test_duration() -> PolarsResult<()> {
    let a = Int64Chunked::new("", &[1, 2, 3])
        .into_datetime(TimeUnit::Nanoseconds, None)
        .into_series();
    let b = Int64Chunked::new("", &[2, 3, 4])
        .into_datetime(TimeUnit::Nanoseconds, None)
        .into_series();
    let c = Int64Chunked::new("", &[1, 1, 1])
        .into_duration(TimeUnit::Nanoseconds)
        .into_series();
    assert_eq!(
        *b.subtract(&a)?.dtype(),
        DataType::Duration(TimeUnit::Nanoseconds)
    );
    assert_eq!(
        *a.add_to(&c)?.dtype(),
        DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    assert_eq!(
        b.subtract(&a)?,
        Int64Chunked::full("", 1, a.len())
            .into_duration(TimeUnit::Nanoseconds)
            .into_series()
    );
    Ok(())
}

#[test]
#[cfg(feature = "dtype-duration")]
fn test_duration_date_arithmetic() -> PolarsResult<()> {
    let date1 = Int32Chunked::new("", &[1, 1, 1]).into_date().into_series();
    let date2 = Int32Chunked::new("", &[2, 3, 4]).into_date().into_series();

    let diff_ms = &date2 - &date1;
    let diff_ms = diff_ms?;
    let diff_us = diff_ms
        .cast(&DataType::Duration(TimeUnit::Microseconds))
        .unwrap();
    let diff_ns = diff_ms
        .cast(&DataType::Duration(TimeUnit::Nanoseconds))
        .unwrap();

    // `+` is commutative for date and duration
    assert_series_eq(&(&diff_ms + &date1)?, &(&date1 + &diff_ms)?);
    assert_series_eq(&(&diff_us + &date1)?, &(&date1 + &diff_us)?);
    assert_series_eq(&(&diff_ns + &date1)?, &(&date1 + &diff_ns)?);

    // `+` is correct date and duration
    assert_series_eq(&(&diff_ms + &date1)?, &date2);
    assert_series_eq(&(&diff_us + &date1)?, &date2);
    assert_series_eq(&(&diff_ns + &date1)?, &date2);

    Ok(())
}

fn assert_series_eq(s1: &Series, s2: &Series) {
    assert!(s1.equals(s2))
}
