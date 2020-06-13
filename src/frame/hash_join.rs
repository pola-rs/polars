use crate::datatypes::UInt32Type;
use crate::series::chunked_array::PrimitiveChunkedBuilder;
use crate::{datatypes::UInt32Chunked, prelude::*, series::chunked_array::ChunkedArray};
use arrow::datatypes::ArrowPrimitiveType;
use fnv::FnvHashMap;
use std::hash::Hash;

/// Hash join a and b.
///     b should be the shorter relation.
fn hash_join<T>(
    a: impl Iterator<Item = Option<T>>,
    b: impl Iterator<Item = Option<T>>,
) -> Vec<(usize, usize)>
where
    T: Hash + Eq + Copy,
{
    let mut hashmap = FnvHashMap::default();

    b.enumerate().for_each(|(idx, o)| {
        if let Some(key) = o {
            hashmap.entry(key).or_insert_with(Vec::new).push(idx)
        }
    });

    let mut results = Vec::new();
    a.enumerate().for_each(|(idx_a, o)| {
        if let Some(key) = o {
            if let Some(indexes_b) = hashmap.get(&key) {
                let tuples = indexes_b.iter().map(|&idx_b| (idx_a, idx_b));
                results.extend(tuples)
            }
        }
    });
    results
}

pub trait HashJoin<T> {
    fn hash_join(&self, other: &ChunkedArray<T>) -> (UInt32Chunked, UInt32Chunked);
}

impl<T> HashJoin<T> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Eq + Hash,
{
    fn hash_join(&self, other: &ChunkedArray<T>) -> (UInt32Chunked, UInt32Chunked) {
        // The shortest relation will be used to create a hash table.
        let left_first = self.len() > other.len();
        let a;
        let b;
        if left_first {
            a = self;
            b = other;
        } else {
            b = self;
            a = other;
        }

        // Resort the relation tuple to match the input, (left, right)
        let srt_tuples = |(a, b)| {
            if left_first {
                (a, b)
            } else {
                (b, a)
            }
        };

        // Create the join tuples
        let join_tuples = hash_join(a.iter(), b.iter());

        // Create the UInt32Chunked arrays. These can be used to take values from both the dataframes.
        let mut left =
            PrimitiveChunkedBuilder::<UInt32Type>::new("left_take_idx", join_tuples.len());
        let mut right =
            PrimitiveChunkedBuilder::<UInt32Type>::new("right_take_idx", join_tuples.len());
        join_tuples.into_iter().map(srt_tuples).for_each(|(a, b)| {
            left.append_value(a as u32);
            right.append_value(b as u32);
        });
        (left.finish(), right.finish())
    }
}
