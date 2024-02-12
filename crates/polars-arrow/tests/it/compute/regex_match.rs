use polars_arrow::array::{BooleanArray, Utf8Array};
use polars_arrow::compute::regex_match::*;
use polars_arrow::error::Result;
use polars_arrow::offset::Offset;

fn test_generic<O: Offset, F: Fn(&Utf8Array<O>, &Utf8Array<O>) -> Result<BooleanArray>>(
    lhs: Vec<&str>,
    pattern: Vec<&str>,
    op: F,
    expected: Vec<bool>,
) {
    let lhs = Utf8Array::<O>::from_slice(lhs);
    let pattern = Utf8Array::<O>::from_slice(pattern);
    let expected = BooleanArray::from_slice(expected);
    let result = op(&lhs, &pattern).unwrap();
    assert_eq!(result, expected);
}

fn test_generic_scalar<O: Offset, F: Fn(&Utf8Array<O>, &str) -> Result<BooleanArray>>(
    lhs: Vec<&str>,
    pattern: &str,
    op: F,
    expected: Vec<bool>,
) {
    let lhs = Utf8Array::<O>::from_slice(lhs);
    let expected = BooleanArray::from_slice(expected);
    let result = op(&lhs, pattern).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_like() {
    test_generic::<i32, _>(
        vec![
            "arrow", "arrow", "arrow", "arrow", "arrow", "arrows", "arrow",
        ],
        vec!["arrow", "^ar", "ro", "foo", "arr$", "arrow.", "arrow."],
        regex_match,
        vec![true, true, true, false, false, true, false],
    )
}

#[test]
fn test_like_scalar() {
    test_generic_scalar::<i32, _>(
        vec!["arrow", "parquet", "datafusion", "flight"],
        "ar",
        regex_match_scalar,
        vec![true, true, false, false],
    );

    test_generic_scalar::<i32, _>(
        vec!["arrow", "parquet", "datafusion", "flight"],
        "^ar",
        regex_match_scalar,
        vec![true, false, false, false],
    )
}
