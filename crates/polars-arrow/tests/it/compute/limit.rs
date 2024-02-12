use polars_arrow::array::*;
use polars_arrow::compute::limit::limit;

#[test]
fn limit_array() {
    let a = Int32Array::from_slice([5, 6, 7, 8, 9]);
    let b = limit(&a, 3);
    let c = b.as_ref().as_any().downcast_ref::<Int32Array>().unwrap();
    let expected = Int32Array::from_slice([5, 6, 7]);
    assert_eq!(&expected, c);
}
