use arrow::bitmap::utils::BitmapIter;

#[test]
fn basic() {
    let values = &[0b01011011u8];
    let iter = BitmapIter::new(values, 0, 6);
    let result = iter.collect::<Vec<_>>();
    assert_eq!(result, vec![true, true, false, true, true, false])
}

#[test]
fn large() {
    let values = &[0b01011011u8];
    let values = std::iter::repeat(values)
        .take(63)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let len = 63 * 8;
    let iter = BitmapIter::new(&values, 0, len);
    assert_eq!(iter.count(), len);
}

#[test]
fn offset() {
    let values = &[0b01011011u8];
    let iter = BitmapIter::new(values, 2, 4);
    let result = iter.collect::<Vec<_>>();
    assert_eq!(result, vec![false, true, true, false])
}

#[test]
fn rev() {
    let values = &[0b01011011u8, 0b01011011u8];
    let iter = BitmapIter::new(values, 2, 13);
    let result = iter.rev().collect::<Vec<_>>();
    assert_eq!(
        result,
        vec![false, true, true, false, true, false, true, true, false, true, true, false, true]
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
    )
}
