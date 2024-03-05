use arrow::bitmap::{and, or, xor, Bitmap};
use proptest::prelude::*;

use super::bitmap_strategy;

proptest! {
    /// Asserts that !bitmap equals all bits flipped
    #[test]
    #[cfg_attr(miri, ignore)] // miri and proptest do not work well :(
    fn not(bitmap in bitmap_strategy()) {
        let not_bitmap: Bitmap = bitmap.iter().map(|x| !x).collect();

        assert_eq!(!&bitmap, not_bitmap);
    }
}

#[test]
fn test_fast_paths() {
    let all_true = Bitmap::from(&[true, true]);
    let all_false = Bitmap::from(&[false, false]);
    let toggled = Bitmap::from(&[true, false]);

    assert_eq!(and(&all_true, &all_true), all_true);
    assert_eq!(and(&all_false, &all_true), all_false);
    assert_eq!(and(&all_true, &all_false), all_false);
    assert_eq!(and(&toggled, &all_false), all_false);
    assert_eq!(and(&toggled, &all_true), toggled);

    assert_eq!(or(&all_true, &all_true), all_true);
    assert_eq!(or(&all_true, &all_false), all_true);
    assert_eq!(or(&all_false, &all_true), all_true);
    assert_eq!(or(&all_false, &all_false), all_false);
    assert_eq!(or(&toggled, &all_false), toggled);

    assert_eq!(xor(&all_true, &all_true), all_false);
    assert_eq!(xor(&all_true, &all_false), all_true);
    assert_eq!(xor(&all_false, &all_true), all_true);
    assert_eq!(xor(&all_false, &all_false), all_false);
    assert_eq!(xor(&toggled, &toggled), all_false);
}
