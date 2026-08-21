use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and_many;

#[test]
// Four Some bitmaps are needed to hit the relevant match branch in combine_validities_and_many().
fn all_unset_bits_are_preserved() {
    let validities = [
        Some(Bitmap::from([false, true])),
        Some(Bitmap::from([true, false])),
        Some(Bitmap::from([true, true])),
        Some(Bitmap::from([true, true])),
    ];

    // The result contains two unset bits and must remain Some,
    // because None represents an all-valid bitmap.
    assert_eq!(
        combine_validities_and_many(&validities),
        Some(Bitmap::from([false, false]))
    );
}
