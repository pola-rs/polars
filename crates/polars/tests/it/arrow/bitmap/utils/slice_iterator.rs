use arrow::bitmap::utils::SlicesIterator;
use arrow::bitmap::Bitmap;
use proptest::prelude::*;

use super::bitmap_strategy;

proptest! {
    /// Asserts that:
    /// * `slots` is the number of set bits in the bitmap
    /// * the sum of the lens of the slices equals `slots`
    /// * each item on each slice is set
    #[test]
    #[cfg_attr(miri, ignore)] // miri and proptest do not work well :(
    fn check_invariants(bitmap in bitmap_strategy()) {
        let iter = SlicesIterator::new(&bitmap);

        let slots = iter.slots();

        assert_eq!(bitmap.len() - bitmap.unset_bits(), slots);

        let slices = iter.collect::<Vec<_>>();
        let mut sum = 0;
        for (start, len) in slices {
            sum += len;
            for i in start..(start+len) {
                assert!(bitmap.get_bit(i));
            }
        }
        assert_eq!(sum, slots);
    }
}

#[test]
fn single_set() {
    let values = (0..16).map(|i| i == 1).collect::<Bitmap>();

    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    let chunks = iter.collect::<Vec<_>>();

    assert_eq!(chunks, vec![(1, 1)]);
    assert_eq!(count, 1);
}

#[test]
fn single_unset() {
    let values = (0..64).map(|i| i != 1).collect::<Bitmap>();

    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    let chunks = iter.collect::<Vec<_>>();

    assert_eq!(chunks, vec![(0, 1), (2, 62)]);
    assert_eq!(count, 64 - 1);
}

#[test]
fn generic() {
    let values = (0..130).map(|i| i % 62 != 0).collect::<Bitmap>();

    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    let chunks = iter.collect::<Vec<_>>();

    assert_eq!(chunks, vec![(1, 61), (63, 61), (125, 5)]);
    assert_eq!(count, 61 + 61 + 5);
}

#[test]
fn incomplete_byte() {
    let values = (0..6).map(|i| i == 1).collect::<Bitmap>();

    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    let chunks = iter.collect::<Vec<_>>();

    assert_eq!(chunks, vec![(1, 1)]);
    assert_eq!(count, 1);
}

#[test]
fn incomplete_byte1() {
    let values = (0..12).map(|i| i == 9).collect::<Bitmap>();

    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    let chunks = iter.collect::<Vec<_>>();

    assert_eq!(chunks, vec![(9, 1)]);
    assert_eq!(count, 1);
}

#[test]
fn end_of_byte() {
    let values = (0..16).map(|i| i != 7).collect::<Bitmap>();

    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    let chunks = iter.collect::<Vec<_>>();

    assert_eq!(chunks, vec![(0, 7), (8, 8)]);
    assert_eq!(count, 15);
}

#[test]
fn bla() {
    let values = vec![true, true, true, true, true, true, true, false]
        .into_iter()
        .collect::<Bitmap>();
    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    assert_eq!(values.unset_bits() + iter.slots(), values.len());

    let total = iter.into_iter().fold(0, |acc, x| acc + x.1);

    assert_eq!(count, total);
}

#[test]
fn past_end_should_not_be_returned() {
    let values = Bitmap::from_u8_slice([0b11111010], 3);
    let iter = SlicesIterator::new(&values);
    let count = iter.slots();
    assert_eq!(values.unset_bits() + iter.slots(), values.len());

    let total = iter.into_iter().fold(0, |acc, x| acc + x.1);

    assert_eq!(count, total);
}

#[test]
fn sliced() {
    let values = Bitmap::from_u8_slice([0b11111010, 0b11111011], 16);
    let values = values.sliced(8, 2);
    let iter = SlicesIterator::new(&values);

    let chunks = iter.collect::<Vec<_>>();

    // the first "11" in the second byte
    assert_eq!(chunks, vec![(0, 2)]);
}

#[test]
fn remainder_1() {
    let values = Bitmap::from_u8_slice([0, 0, 0b00000000, 0b00010101], 27);
    let values = values.sliced(22, 5);
    let iter = SlicesIterator::new(&values);
    let chunks = iter.collect::<Vec<_>>();
    assert_eq!(chunks, vec![(2, 1), (4, 1)]);
}
