use arrow::array::Int32Array;
use arrow::compute::arity_assign::{binary, unary};

#[test]
fn test_unary_assign() {
    let mut a = Int32Array::from([Some(5), Some(6), None, Some(10)]);

    unary(&mut a, |x| x + 10);

    assert_eq!(a, Int32Array::from([Some(15), Some(16), None, Some(20)]))
}

#[test]
fn test_binary_assign() {
    let mut a = Int32Array::from([Some(5), Some(6), None, Some(10)]);
    let b = Int32Array::from([Some(1), Some(2), Some(1), None]);

    binary(&mut a, &b, |x, y| x + y);

    assert_eq!(a, Int32Array::from([Some(6), Some(8), None, None]))
}
