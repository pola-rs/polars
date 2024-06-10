use arrow::array::Int32Array;
use polars_core::prelude::Int32Chunked;
use polars_core::utils::Container;

#[test]
fn test_auto_rechunk_many_small() {
    let arr = Int32Array::from_vec(vec![1]);

    let ca = Int32Chunked::from_chunk_iter("", std::iter::repeat(arr).take(5));
    assert_eq!(ca.n_chunks(), 1);
}
