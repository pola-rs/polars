use polars_arrow::export::arrow::array::BinaryArray;
use polars_core::export::ahash::RandomState;
use polars_core::hashing::_hash_binary_array;
use polars_core::prelude::*;

pub(super) fn hash_series(columns: &[Series], buf: &mut Vec<u64>, hb: &RandomState) {
    debug_assert!(buf.is_empty());
    let mut col_iter = columns.iter();
    let first_key = col_iter.next().unwrap();
    first_key.vec_hash(hb.clone(), buf).unwrap();
    for other_key in col_iter {
        other_key.vec_hash_combine(hb.clone(), buf).unwrap();
    }
}

pub(super) fn hash_rows(columns: &BinaryArray<i64>, buf: &mut Vec<u64>, hb: &RandomState) {
    debug_assert!(buf.is_empty());
    _hash_binary_array(columns, hb.clone(), buf);
}

pub(super) fn load_vec<T, F: Fn() -> T>(partitions: usize, item: F) -> Vec<T> {
    let mut buf = Vec::with_capacity(partitions);
    for _ in 0..partitions {
        buf.push(item());
    }
    buf
}
