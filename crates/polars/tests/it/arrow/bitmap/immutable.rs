use arrow::array::Splitable;
use arrow::bitmap::Bitmap;

#[test]
fn as_slice() {
    let b = Bitmap::from([true, true, true, true, true, true, true, true, true]);

    let (slice, offset, length) = b.as_slice();
    assert_eq!(slice, &[0b11111111, 0b1]);
    assert_eq!(offset, 0);
    assert_eq!(length, 9);
}

#[test]
fn as_slice_offset() {
    let b = Bitmap::from([true, true, true, true, true, true, true, true, true]);
    let b = b.sliced(8, 1);

    let (slice, offset, length) = b.as_slice();
    assert_eq!(slice, &[0b1]);
    assert_eq!(offset, 0);
    assert_eq!(length, 1);
}

#[test]
fn as_slice_offset_middle() {
    let b = Bitmap::from_u8_slice([0, 0, 0, 0b00010101], 27);
    let b = b.sliced(22, 5);

    let (slice, offset, length) = b.as_slice();
    assert_eq!(slice, &[0, 0b00010101]);
    assert_eq!(offset, 6);
    assert_eq!(length, 5);
}

#[test]
fn split_at_unset_bits() {
    let bm = Bitmap::from_u8_slice([0b01101010, 0, 0, 0b100], 27);

    assert_eq!(bm.unset_bits(), 22);

    let (lhs, rhs) = bm.split_at(5);
    assert_eq!(lhs.lazy_unset_bits(), Some(3));
    assert_eq!(rhs.lazy_unset_bits(), Some(19));

    let (lhs, rhs) = bm.split_at(22);
    assert_eq!(lhs.lazy_unset_bits(), Some(18));
    assert_eq!(rhs.lazy_unset_bits(), Some(4));

    let (lhs, rhs) = bm.split_at(0);
    assert_eq!(lhs.lazy_unset_bits(), Some(0));
    assert_eq!(rhs.lazy_unset_bits(), Some(22));

    let (lhs, rhs) = bm.split_at(27);
    assert_eq!(lhs.lazy_unset_bits(), Some(22));
    assert_eq!(rhs.lazy_unset_bits(), Some(0));

    let bm = Bitmap::new_zeroed(1024);
    let (lhs, rhs) = bm.split_at(512);
    assert_eq!(lhs.lazy_unset_bits(), Some(512));
    assert_eq!(rhs.lazy_unset_bits(), Some(512));

    let bm = Bitmap::new_with_value(true, 1024);
    let (lhs, rhs) = bm.split_at(512);
    assert_eq!(lhs.lazy_unset_bits(), Some(0));
    assert_eq!(rhs.lazy_unset_bits(), Some(0));
}

#[test]
fn debug() {
    let b = Bitmap::from([true, true, false, true, true, true, true, true, true]);
    let b = b.sliced(2, 7);

    assert_eq!(
        format!("{b:?}"),
        "Bitmap { len: 7, offset: 2, bytes: [0b111110__, 0b_______1] }"
    );
}
