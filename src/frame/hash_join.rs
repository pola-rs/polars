use crate::{
    datatypes::UInt32Chunked,
    series::chunked_array::ChunkedArray,
    prelude::*,
};
use fnv::FnvHashMap;

// If you know one of the tables is smaller, it is best to make it the second parameter.
// fn _hash_join<A, B, K>(first: &[(K, A)], second: &[(K, B)]) -> Vec<(A, K, B)>
//     where
//         K: Hash + Eq + Copy,
//         A: Copy,
//         B: Copy,
// {
//     let mut hash_map = HashMap::new();
//
//     // hash phase
//     for &(key, val_a) in second {
//         // collect all values by their keys, appending new ones to each existing entry
//         hash_map.entry(key).or_insert_with(Vec::new).push(val_a);
//     }
//
//     let mut result = Vec::new();
//     // join phase
//     for &(key, val_b) in first {
//         if let Some(vals) = hash_map.get(&key) {
//             let tuples = vals.iter().map(|&val_a| (val_b, key, val_a));
//             result.extend(tuples);
//         }
//     }
//
//     result
// }


fn hash_join<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> UInt32Chunked {
    // b.iter().for_each(||)


    unimplemented!()
}
