use arrow::array::*;

use super::test_equal;

#[test]
fn test_primitive() {
    let cases = vec![
        (
            vec![Some(1), Some(2), Some(3)],
            vec![Some(1), Some(2), Some(3)],
            true,
        ),
        (
            vec![Some(1), Some(2), Some(3)],
            vec![Some(1), Some(2), Some(4)],
            false,
        ),
        (
            vec![Some(1), Some(2), None],
            vec![Some(1), Some(2), None],
            true,
        ),
        (
            vec![Some(1), None, Some(3)],
            vec![Some(1), Some(2), None],
            false,
        ),
        (
            vec![Some(1), None, None],
            vec![Some(1), Some(2), None],
            false,
        ),
    ];

    for (lhs, rhs, expected) in cases {
        let lhs = Int32Array::from(&lhs);
        let rhs = Int32Array::from(&rhs);
        test_equal(&lhs, &rhs, expected);
    }
}

#[test]
fn test_primitive_slice() {
    let cases = vec![
        (
            vec![Some(1), Some(2), Some(3)],
            (0, 1),
            vec![Some(1), Some(2), Some(3)],
            (0, 1),
            true,
        ),
        (
            vec![Some(1), Some(2), Some(3)],
            (1, 1),
            vec![Some(1), Some(2), Some(3)],
            (2, 1),
            false,
        ),
        (
            vec![Some(1), Some(2), None],
            (1, 1),
            vec![Some(1), None, Some(2)],
            (2, 1),
            true,
        ),
        (
            vec![None, Some(2), None],
            (1, 1),
            vec![None, None, Some(2)],
            (2, 1),
            true,
        ),
        (
            vec![Some(1), None, Some(2), None, Some(3)],
            (2, 2),
            vec![None, Some(2), None, Some(3)],
            (1, 2),
            true,
        ),
    ];

    for (lhs, slice_lhs, rhs, slice_rhs, expected) in cases {
        let lhs = Int32Array::from(&lhs);
        let lhs = lhs.sliced(slice_lhs.0, slice_lhs.1);
        let rhs = Int32Array::from(&rhs);
        let rhs = rhs.sliced(slice_rhs.0, slice_rhs.1);

        test_equal(&lhs, &rhs, expected);
    }
}
