use polars_parquet::parquet::indexes::Interval;

#[test]
fn bitmap_incomplete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![HybridEncoded::Bitmap(&[0b01000011], 7)].into_iter(),
        vec![Interval::new(1, 2)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Skipped(1),
            FilteredHybridEncoded::Bitmap {
                values: &[0b01000011],
                offset: 1,
                length: 2,
            }
        ]
    );
}

#[test]
fn bitmap_complete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![HybridEncoded::Bitmap(&[0b01000011], 8)].into_iter(),
        vec![Interval::new(0, 8)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![FilteredHybridEncoded::Bitmap {
            values: &[0b01000011],
            offset: 0,
            length: 8,
        }]
    );
}

#[test]
fn bitmap_interval_incomplete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Bitmap(&[0b01000011], 8),
            HybridEncoded::Bitmap(&[0b11111111], 8),
        ]
        .into_iter(),
        vec![Interval::new(0, 10)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Bitmap {
                values: &[0b01000011],
                offset: 0,
                length: 8,
            },
            FilteredHybridEncoded::Bitmap {
                values: &[0b11111111],
                offset: 0,
                length: 2,
            }
        ]
    );
}

#[test]
fn bitmap_interval_run_incomplete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Bitmap(&[0b01100011], 8),
            HybridEncoded::Bitmap(&[0b11111111], 8),
        ]
        .into_iter(),
        vec![Interval::new(0, 5), Interval::new(7, 4)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Bitmap {
                values: &[0b01100011],
                offset: 0,
                length: 5,
            },
            FilteredHybridEncoded::Skipped(2),
            FilteredHybridEncoded::Bitmap {
                values: &[0b01100011],
                offset: 7,
                length: 1,
            },
            FilteredHybridEncoded::Bitmap {
                values: &[0b11111111],
                offset: 0,
                length: 3,
            }
        ]
    );
}

#[test]
fn bitmap_interval_run_skipped() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Bitmap(&[0b01100011], 8),
            HybridEncoded::Bitmap(&[0b11111111], 8),
        ]
        .into_iter(),
        vec![Interval::new(9, 2)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Skipped(4),
            FilteredHybridEncoded::Skipped(1),
            FilteredHybridEncoded::Bitmap {
                values: &[0b11111111],
                offset: 1,
                length: 2,
            },
        ]
    );
}

#[test]
fn bitmap_interval_run_offset_skipped() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Bitmap(&[0b01100011], 8),
            HybridEncoded::Bitmap(&[0b11111111], 8),
        ]
        .into_iter(),
        vec![Interval::new(0, 1), Interval::new(9, 2)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Bitmap {
                values: &[0b01100011],
                offset: 0,
                length: 1,
            },
            FilteredHybridEncoded::Skipped(3),
            FilteredHybridEncoded::Skipped(1),
            FilteredHybridEncoded::Bitmap {
                values: &[0b11111111],
                offset: 1,
                length: 2,
            },
        ]
    );
}

#[test]
fn repeated_incomplete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![HybridEncoded::Repeated(true, 7)].into_iter(),
        vec![Interval::new(1, 2)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Skipped(1),
            FilteredHybridEncoded::Repeated {
                is_set: true,
                length: 2,
            }
        ]
    );
}

#[test]
fn repeated_complete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![HybridEncoded::Repeated(true, 8)].into_iter(),
        vec![Interval::new(0, 8)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![FilteredHybridEncoded::Repeated {
            is_set: true,
            length: 8,
        }]
    );
}

#[test]
fn repeated_interval_incomplete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Repeated(true, 8),
            HybridEncoded::Repeated(false, 8),
        ]
        .into_iter(),
        vec![Interval::new(0, 10)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Repeated {
                is_set: true,
                length: 8,
            },
            FilteredHybridEncoded::Repeated {
                is_set: false,
                length: 2,
            }
        ]
    );
}

#[test]
fn repeated_interval_run_incomplete() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Repeated(true, 8),
            HybridEncoded::Repeated(false, 8),
        ]
        .into_iter(),
        vec![Interval::new(0, 5), Interval::new(7, 4)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Repeated {
                is_set: true,
                length: 5,
            },
            FilteredHybridEncoded::Skipped(2),
            FilteredHybridEncoded::Repeated {
                is_set: true,
                length: 1,
            },
            FilteredHybridEncoded::Repeated {
                is_set: false,
                length: 3,
            }
        ]
    );
}

#[test]
fn repeated_interval_run_skipped() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Repeated(true, 8),
            HybridEncoded::Repeated(false, 8),
        ]
        .into_iter(),
        vec![Interval::new(9, 2)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Skipped(8),
            FilteredHybridEncoded::Skipped(0),
            FilteredHybridEncoded::Repeated {
                is_set: false,
                length: 2,
            },
        ]
    );
}

#[test]
fn repeated_interval_run_offset_skipped() {
    let mut iter = FilteredHybridBitmapIter::new(
        vec![
            HybridEncoded::Repeated(true, 8),
            HybridEncoded::Repeated(false, 8),
        ]
        .into_iter(),
        vec![Interval::new(0, 1), Interval::new(9, 2)].into(),
    );
    let a = iter.by_ref().collect::<Vec<_>>();
    assert_eq!(iter.len(), 0);
    assert_eq!(
        a,
        vec![
            FilteredHybridEncoded::Repeated {
                is_set: true,
                length: 1,
            },
            FilteredHybridEncoded::Skipped(7),
            FilteredHybridEncoded::Skipped(0),
            FilteredHybridEncoded::Repeated {
                is_set: false,
                length: 2,
            },
        ]
    );
}
