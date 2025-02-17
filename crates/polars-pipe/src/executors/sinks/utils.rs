use arrow::array::BinaryArray;
use polars_core::hashing::_hash_binary_array;
use polars_utils::aliases::PlRandomState;

pub(super) fn hash_rows(columns: &BinaryArray<i64>, buf: &mut Vec<u64>, hb: &PlRandomState) {
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
