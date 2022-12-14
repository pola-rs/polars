use super::*;

#[test]
#[cfg(feature = "dtype-datetime")]
fn test_agg_list_type() -> PolarsResult<()> {
    let s = Series::new("foo", &[1, 2, 3]);
    let s = s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;

    let l = unsafe { s.agg_list(&GroupsProxy::Idx(vec![(0, vec![0, 1, 2])].into())) };

    match l.dtype() {
        DataType::List(inner) => {
            assert!(matches!(
                &**inner,
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            ))
        }
        _ => assert!(false),
    }

    Ok(())
}

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

    let out = df.outer_join(&df.clone(), ["bar"], ["bar"])?;
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
