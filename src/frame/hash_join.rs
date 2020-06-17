use crate::datatypes::UInt32Type;
use crate::frame::DataFrame;
use crate::series::chunked_array::PrimitiveChunkedBuilder;
use crate::{datatypes::UInt32Chunked, prelude::*, series::chunked_array::ChunkedArray};
use arrow::compute::TakeOptions;
use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};
use fnv::{FnvBuildHasher, FnvHashMap};
use std::collections::HashSet;
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

impl DataFrame {
    pub fn join(&self, other: &DataFrame, left_on: &str, right_on: &str) -> Result<DataFrame> {
        let s_left = self.select(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.select(right_on).ok_or(PolarsError::NotFound)?;

        macro_rules! hash_join {
            ($s_right:ident, $ca_left:ident, $type_:ident) => {{
                let ca_right = $s_right.$type_()?;
                $ca_left.hash_join(ca_right)
            }};
        }

        let (take_left, take_right) = match s_left {
            Series::UInt32(ca_left) => hash_join!(s_right, ca_left, u32),
            Series::Int32(ca_left) => hash_join!(s_right, ca_left, i32),
            Series::Int64(ca_left) => hash_join!(s_right, ca_left, i64),
            Series::Bool(ca_left) => hash_join!(s_right, ca_left, bool),
            _ => unimplemented!(),
        };

        let mut df_left = self.take(&take_left, Some(TakeOptions::default()))?;
        let mut df_right = other.take(&take_right, Some(TakeOptions::default()))?;
        df_right.drop(right_on);

        let mut left_names =
            HashSet::with_capacity_and_hasher(df_left.width(), FnvBuildHasher::default());
        for field in df_left.schema.fields() {
            left_names.insert(field.name());
        }

        let mut rename_strs = Vec::with_capacity(df_right.width());

        for field in df_right.schema.fields() {
            if left_names.contains(field.name()) {
                rename_strs.push(field.name().to_owned())
            }
        }

        for name in rename_strs {
            df_right.rename(&name, &format!("{}_right", name))?
        }

        df_left.hstack(&df_right.columns);
        Ok(df_left)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hash_join() {
        let s0 = Series::init("days", [0, 1, 2].as_ref());
        let s1 = Series::init("temp", [22.1, 19.9, 7.].as_ref());
        let s2 = Series::init("rain", [0.2, 0.1, 0.3].as_ref());
        let temp = DataFrame::new_from_columns(vec![s0, s1, s2]).unwrap();

        let s0 = Series::init("days", [1, 2, 3, 1].as_ref());
        let s1 = Series::init("rain", [0.1, 0.2, 0.3, 0.4].as_ref());
        let rain = DataFrame::new_from_columns(vec![s0, s1]).unwrap();

        let joined = temp.join(&rain, "days", "days");
        println!("{}", joined.unwrap())
    }
}
