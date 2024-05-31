use super::*;

#[test]
fn test_rolling() {
    let s = Int32Chunked::new("foo", &[1, 2, 3, 2, 1]).into_series();
    let a = s
        .rolling_sum(RollingOptionsFixedWindow {
            window_size: 2,
            min_periods: 1,
            ..Default::default()
        })
        .unwrap();
    let a = a.i32().unwrap();
    assert_eq!(
        Vec::from(a),
        [1, 3, 5, 5, 3]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );
    let a = s
        .rolling_min(RollingOptionsFixedWindow {
            window_size: 2,
            min_periods: 1,
            ..Default::default()
        })
        .unwrap();
    let a = a.i32().unwrap();
    assert_eq!(
        Vec::from(a),
        [1, 1, 2, 2, 1]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );
    let a = s
        .rolling_max(RollingOptionsFixedWindow {
            window_size: 2,
            weights: Some(vec![1., 1.]),
            min_periods: 1,
            ..Default::default()
        })
        .unwrap();

    let a = a.f64().unwrap();
    assert_eq!(
        Vec::from(a),
        [1., 2., 3., 3., 2.]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_rolling_min_periods() {
    let s = Int32Chunked::new("foo", &[1, 2, 3, 2, 1]).into_series();
    let a = s
        .rolling_max(RollingOptionsFixedWindow {
            window_size: 2,
            min_periods: 2,
            ..Default::default()
        })
        .unwrap();
    let a = a.i32().unwrap();
    assert_eq!(Vec::from(a), &[None, Some(2), Some(3), Some(3), Some(2)]);
}

#[test]
fn test_rolling_mean() {
    let s = Float64Chunked::new(
        "foo",
        &[
            Some(0.0),
            Some(1.0),
            Some(2.0),
            None,
            None,
            Some(5.0),
            Some(6.0),
        ],
    )
    .into_series();

    // check err on wrong input
    assert!(s
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: 1,
            min_periods: 2,
            ..Default::default()
        })
        .is_err());

    // validate that we divide by the proper window length. (same as pandas)
    let a = s
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            center: false,
            ..Default::default()
        })
        .unwrap();
    let a = a.f64().unwrap();
    assert_eq!(
        Vec::from(a),
        &[
            Some(0.0),
            Some(0.5),
            Some(1.0),
            Some(1.5),
            Some(2.0),
            Some(5.0),
            Some(5.5)
        ]
    );

    // check centered rolling window
    let a = s
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            center: true,
            ..Default::default()
        })
        .unwrap();
    let a = a.f64().unwrap();
    assert_eq!(
        Vec::from(a),
        &[
            Some(0.5),
            Some(1.0),
            Some(1.5),
            Some(2.0),
            Some(5.0),
            Some(5.5),
            Some(5.5)
        ]
    );

    // integers
    let ca = Int32Chunked::from_slice("", &[1, 8, 6, 2, 16, 10]);
    let out = ca
        .into_series()
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: 2,
            weights: None,
            min_periods: 2,
            center: false,
            ..Default::default()
        })
        .unwrap();

    let out = out.f64().unwrap();
    assert_eq!(
        Vec::from(out),
        &[None, Some(4.5), Some(7.0), Some(4.0), Some(9.0), Some(13.0),]
    );
}

#[test]
fn test_rolling_map() {
    let ca = Float64Chunked::new(
        "foo",
        &[
            Some(0.0),
            Some(1.0),
            Some(2.0),
            None,
            None,
            Some(5.0),
            Some(6.0),
        ],
    );

    let out = ca
        .rolling_map(
            &|s| s.sum_reduce().unwrap().into_series(s.name()),
            RollingOptionsFixedWindow {
                window_size: 3,
                min_periods: 3,
                ..Default::default()
            },
        )
        .unwrap();

    let out = out.f64().unwrap();

    assert_eq!(
        Vec::from(out),
        &[None, None, Some(3.0), None, None, None, None,]
    );
}

#[test]
fn test_rolling_var() {
    let s = Float64Chunked::new(
        "foo",
        &[
            Some(0.0),
            Some(1.0),
            Some(2.0),
            None,
            None,
            Some(5.0),
            Some(6.0),
        ],
    )
    .into_series();
    // window larger than array
    assert_eq!(
        s.rolling_var(RollingOptionsFixedWindow {
            window_size: 10,
            min_periods: 10,
            ..Default::default()
        })
        .unwrap()
        .null_count(),
        s.len()
    );

    let options = RollingOptionsFixedWindow {
        window_size: 3,
        min_periods: 3,
        ..Default::default()
    };
    let out = s
        .rolling_var(options.clone())
        .unwrap()
        .cast(&DataType::Int32)
        .unwrap();
    let out = out.i32().unwrap();
    assert_eq!(
        Vec::from(out),
        &[None, None, Some(1), None, None, None, None,]
    );

    let s = Float64Chunked::from_slice("", &[0.0, 2.0, 8.0, 3.0, 12.0, 1.0]).into_series();
    let out = s
        .rolling_var(options)
        .unwrap()
        .cast(&DataType::Int32)
        .unwrap();
    let out = out.i32().unwrap();

    assert_eq!(
        Vec::from(out),
        &[None, None, Some(17), Some(10), Some(20), Some(34),]
    );

    // check centered rolling window
    let out = s
        .rolling_var(RollingOptionsFixedWindow {
            window_size: 4,
            min_periods: 3,
            center: true,
            ..Default::default()
        })
        .unwrap();
    let out = out.f64().unwrap().to_vec();

    let exp_res = &[
        None,
        Some(17.333333333333332),
        Some(11.583333333333334),
        Some(21.583333333333332),
        Some(24.666666666666668),
        Some(34.33333333333334),
    ];
    let test_res = out.iter().zip(exp_res.iter()).all(|(&a, &b)| match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => (a - b).abs() < 1e-12,
        (_, _) => false,
    });
    assert!(
        test_res,
        "{:?} is not approximately equal to {:?}",
        out, exp_res
    );
}
