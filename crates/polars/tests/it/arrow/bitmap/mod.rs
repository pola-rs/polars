mod assign_ops;
mod bitmap_ops;
mod immutable;
mod mutable;
mod utils;

use arrow::array::Splitable;
use arrow::bitmap::Bitmap;
use proptest::prelude::*;

/// Returns a strategy of an arbitrary sliced [`Bitmap`] of size up to 1000
pub(crate) fn bitmap_strategy() -> impl Strategy<Value = Bitmap> {
    prop::collection::vec(any::<bool>(), 1..1000)
        .prop_flat_map(|vec| {
            let len = vec.len();
            (Just(vec), 0..len)
        })
        .prop_flat_map(|(vec, index)| {
            let len = vec.len();
            (Just(vec), Just(index), 0..len - index)
        })
        .prop_flat_map(|(vec, index, len)| {
            let bitmap = Bitmap::from(&vec);
            let bitmap = bitmap.sliced(index, len);
            Just(bitmap)
        })
}

fn create_bitmap<P: AsRef<[u8]>>(bytes: P, len: usize) -> Bitmap {
    let buffer = Vec::<u8>::from(bytes.as_ref());
    Bitmap::from_u8_vec(buffer, len)
}

#[test]
fn eq() {
    let lhs = create_bitmap([0b01101010], 8);
    let rhs = create_bitmap([0b01001110], 8);
    assert!(lhs != rhs);
}

#[test]
fn eq_len() {
    let lhs = create_bitmap([0b01101010], 6);
    let rhs = create_bitmap([0b00101010], 6);
    assert!(lhs == rhs);
    let rhs = create_bitmap([0b00001010], 6);
    assert!(lhs != rhs);
}

#[test]
fn eq_slice() {
    let lhs = create_bitmap([0b10101010], 8).sliced(1, 7);
    let rhs = create_bitmap([0b10101011], 8).sliced(1, 7);
    assert!(lhs == rhs);

    let lhs = create_bitmap([0b10101010], 8).sliced(2, 6);
    let rhs = create_bitmap([0b10101110], 8).sliced(2, 6);
    assert!(lhs != rhs);
}

#[test]
fn and() {
    let lhs = create_bitmap([0b01101010], 8);
    let rhs = create_bitmap([0b01001110], 8);
    let expected = create_bitmap([0b01001010], 8);
    assert_eq!(&lhs & &rhs, expected);
}

#[test]
fn or_large() {
    let input: &[u8] = &[
        0b00000000, 0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
        0b01000010, 0b11111111,
    ];
    let input1: &[u8] = &[
        0b00000000, 0b00000001, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000,
        0b10000000, 0b11111111,
    ];
    let expected: &[u8] = &[
        0b00000000, 0b00000001, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000,
        0b11000010, 0b11111111,
    ];

    let lhs = create_bitmap(input, 62);
    let rhs = create_bitmap(input1, 62);
    let expected = create_bitmap(expected, 62);
    assert_eq!(&lhs | &rhs, expected);
}

#[test]
fn and_offset() {
    let lhs = create_bitmap([0b01101011], 8).sliced(1, 7);
    let rhs = create_bitmap([0b01001111], 8).sliced(1, 7);
    let expected = create_bitmap([0b01001010], 8).sliced(1, 7);
    assert_eq!(&lhs & &rhs, expected);
}

#[test]
fn or() {
    let lhs = create_bitmap([0b01101010], 8);
    let rhs = create_bitmap([0b01001110], 8);
    let expected = create_bitmap([0b01101110], 8);
    assert_eq!(&lhs | &rhs, expected);
}

#[test]
fn not() {
    let lhs = create_bitmap([0b01101010], 6);
    let expected = create_bitmap([0b00010101], 6);
    assert_eq!(!&lhs, expected);
}

#[test]
fn subslicing_gives_correct_null_count() {
    let base = Bitmap::from([false, true, true, false, false, true, true, true]);
    assert_eq!(base.unset_bits(), 3);

    let view1 = base.clone().sliced(0, 1);
    let view2 = base.sliced(1, 7);
    assert_eq!(view1.unset_bits(), 1);
    assert_eq!(view2.unset_bits(), 2);

    let view3 = view2.sliced(0, 1);
    assert_eq!(view3.unset_bits(), 0);
}

#[test]
fn split_at() {
    let bm = create_bitmap([0b01101010], 8);

    let (lhs, rhs) = bm.split_at(5);
    assert_eq!(
        &lhs.iter().collect::<Vec<bool>>(),
        &[false, true, false, true, false]
    );
    assert_eq!(&rhs.iter().collect::<Vec<bool>>(), &[true, true, false]);
}
