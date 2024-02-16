use arrow::array::*;
use arrow::offset::Offset;

use super::{binary_cases, test_equal};

fn test_generic_string_equal<O: Offset>() {
    let cases = binary_cases();

    for (lhs, rhs, expected) in cases {
        let lhs = lhs.iter().map(|x| x.as_deref());
        let rhs = rhs.iter().map(|x| x.as_deref());
        let lhs = Utf8Array::<O>::from_trusted_len_iter(lhs);
        let rhs = Utf8Array::<O>::from_trusted_len_iter(rhs);
        test_equal(&lhs, &rhs, expected);
    }
}

#[test]
fn utf8_equal() {
    test_generic_string_equal::<i32>()
}

#[test]
fn large_utf8_equal() {
    test_generic_string_equal::<i64>()
}
