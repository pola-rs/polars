use crate::datatypes::UInt32Type;
use crate::frame::DataFrame;
use crate::series::chunked_array::builder::PrimitiveChunkedBuilder;
use crate::{datatypes::UInt32Chunked, prelude::*, series::chunked_array::ChunkedArray};
use arrow::compute::TakeOptions;
use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};
use fnv::{FnvBuildHasher, FnvHashMap};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

// TODO: join reuse code in functions/ macros

pub(crate) fn prepare_hashed_relation<T>(
    b: impl Iterator<Item = Option<T>>,
) -> HashMap<T, Vec<usize>, FnvBuildHasher>
where
    T: Hash + Eq + Copy,
{
    let mut hash_tbl = FnvHashMap::default();

    b.enumerate().for_each(|(idx, o)| {
        if let Some(key) = o {
            hash_tbl.entry(key).or_insert_with(Vec::new).push(idx)
        }
    });
    hash_tbl
}

/// Hash join a and b.
///     b should be the shorter relation.
fn hash_join<T>(
    a: impl Iterator<Item = Option<T>>,
    b: impl Iterator<Item = Option<T>>,
) -> Vec<(usize, usize)>
where
    T: Hash + Eq + Copy,
{
    let hash_tbl = prepare_hashed_relation(b);

    let mut results = Vec::new();
    a.enumerate().for_each(|(idx_a, o)| {
        if let Some(key) = o {
            if let Some(indexes_b) = hash_tbl.get(&key) {
                let tuples = indexes_b.iter().map(|&idx_b| (idx_a, idx_b));
                results.extend(tuples)
            }
        }
    });
    results
}

fn hash_join_left<T>(
    a: impl Iterator<Item = Option<T>>,
    b: impl Iterator<Item = Option<T>>,
) -> Vec<(usize, Option<usize>)>
where
    T: Hash + Eq + Copy,
{
    let hash_tbl = prepare_hashed_relation(b);
    let mut results = Vec::new();

    a.enumerate().for_each(|(idx_a, o)| {
        match o {
            // left value is null, so right is automatically null
            None => results.push((idx_a, None)),
            Some(key) => {
                match hash_tbl.get(&key) {
                    // left and right matches
                    Some(indexes_b) => {
                        results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b))))
                    }
                    // only left values, right = null
                    None => results.push((idx_a, None)),
                }
            }
        }
    });
    results
}

pub trait HashJoin<T> {
    fn hash_join(&self, other: &ChunkedArray<T>) -> (UInt32Chunked, UInt32Chunked);
    fn hash_join_left(&self, other: &ChunkedArray<T>) -> (UInt32Chunked, UInt32Chunked);
}

macro_rules! create_join_tuples {
    // wrap option makes the iterator add an Option, needed for utf-8
    ($self:expr, $other:expr) => {{
        // The shortest relation will be used to create a hash table.
        let left_first = $self.len() > $other.len();
        let a;
        let b;
        if left_first {
            a = $self;
            b = $other;
        } else {
            b = $self;
            a = $other;
        }

        // Resort the relation tuple to match the input, (left, right)
        let srt_tuples = move |(a, b)| {
            if left_first {
                (a, b)
            } else {
                (b, a)
            }
        };

        (srt_tuples, a, b)
    }};
}

impl<T> HashJoin<T> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Eq + Hash,
{
    fn hash_join(&self, other: &ChunkedArray<T>) -> (UInt32Chunked, UInt32Chunked) {
        let (srt_tuples, a, b) = create_join_tuples!(self, other);
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

    fn hash_join_left(&self, other: &ChunkedArray<T>) -> (UInt32Chunked, UInt32Chunked) {
        let join_tuples = hash_join_left(self.iter(), other.iter());
        // Create the UInt32Chunked arrays. These can be used to take values from both the dataframes.
        let mut left =
            PrimitiveChunkedBuilder::<UInt32Type>::new("left_take_idx", join_tuples.len());
        let mut right =
            PrimitiveChunkedBuilder::<UInt32Type>::new("right_take_idx", join_tuples.len());
        join_tuples
            .into_iter()
            .for_each(|(idx_left, opt_idx_right)| {
                left.append_value(idx_left as u32);

                match opt_idx_right {
                    Some(idx) => right.append_value(idx as u32).expect("could not append"),
                    None => right.append_null().expect("could not append"),
                };
            });
        (left.finish(), right.finish())
    }
}

impl HashJoin<Utf8Type> for Utf8Chunked {
    fn hash_join(&self, other: &Utf8Chunked) -> (UInt32Chunked, UInt32Chunked) {
        let (srt_tuples, a, b) = create_join_tuples!(self, other);
        // Create the join tuples
        let join_tuples = hash_join(a.iter().map(|v| Some(v)), b.iter().map(|v| Some(v)));

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

    fn hash_join_left(&self, other: &Utf8Chunked) -> (UInt32Chunked, UInt32Chunked) {
        let join_tuples =
            hash_join_left(self.iter().map(|v| Some(v)), other.iter().map(|v| Some(v)));
        // Create the UInt32Chunked arrays. These can be used to take values from both the dataframes.
        let mut left =
            PrimitiveChunkedBuilder::<UInt32Type>::new("left_take_idx", join_tuples.len());
        let mut right =
            PrimitiveChunkedBuilder::<UInt32Type>::new("right_take_idx", join_tuples.len());
        join_tuples
            .into_iter()
            .for_each(|(idx_left, opt_idx_right)| {
                left.append_value(idx_left as u32);

                match opt_idx_right {
                    Some(idx) => right.append_value(idx as u32).expect("could not append"),
                    None => right.append_null().expect("could not append"),
                };
            });
        (left.finish(), right.finish())
    }
}

impl DataFrame {
    fn finish_join(
        &self,
        other: &DataFrame,
        take_left: &UInt32Chunked,
        take_right: &UInt32Chunked,
        right_on: &str,
    ) -> Result<DataFrame> {
        let mut df_left = self.take(take_left, Some(TakeOptions::default()))?;
        let mut df_right = other.take(take_right, Some(TakeOptions::default()))?;
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

    pub fn inner_join(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
    ) -> Result<DataFrame> {
        let s_left = self.select(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.select(right_on).ok_or(PolarsError::NotFound)?;

        macro_rules! hash_join {
            ($s_right:ident, $ca_left:ident, $type_:ident) => {{
                // call the type method series.i32()
                let ca_right = $s_right.$type_()?;
                $ca_left.hash_join(ca_right)
            }};
        }

        let (take_left, take_right) = match s_left {
            Series::UInt32(ca_left) => hash_join!(s_right, ca_left, u32),
            Series::Int32(ca_left) => hash_join!(s_right, ca_left, i32),
            Series::Int64(ca_left) => hash_join!(s_right, ca_left, i64),
            Series::Bool(ca_left) => hash_join!(s_right, ca_left, bool),
            Series::Utf8(ca_left) => hash_join!(s_right, ca_left, utf8),
            _ => unimplemented!(),
        };
        self.finish_join(other, &take_left, &take_right, right_on)
    }

    pub fn left_join(&self, other: &DataFrame, left_on: &str, right_on: &str) -> Result<DataFrame> {
        let s_left = self.select(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.select(right_on).ok_or(PolarsError::NotFound)?;

        macro_rules! hash_join {
            ($s_right:ident, $ca_left:ident, $type_:ident) => {{
                let ca_right = $s_right.$type_()?;
                $ca_left.hash_join_left(ca_right)
            }};
        }

        let (take_left, take_right) = match s_left {
            Series::UInt32(ca_left) => hash_join!(s_right, ca_left, u32),
            Series::Int32(ca_left) => hash_join!(s_right, ca_left, i32),
            Series::Int64(ca_left) => hash_join!(s_right, ca_left, i64),
            Series::Bool(ca_left) => hash_join!(s_right, ca_left, bool),
            Series::Utf8(ca_left) => hash_join!(s_right, ca_left, utf8),
            _ => unimplemented!(),
        };
        self.finish_join(other, &take_left, &take_right, right_on)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_inner_join() {
        let s0 = Series::init("days", [0, 1, 2].as_ref());
        let s1 = Series::init("temp", [22.1, 19.9, 7.].as_ref());
        let s2 = Series::init("rain", [0.2, 0.1, 0.3].as_ref());
        let temp = DataFrame::new_from_columns(vec![s0, s1, s2]).unwrap();

        let s0 = Series::init("days", [1, 2, 3, 1].as_ref());
        let s1 = Series::init("rain", [0.1, 0.2, 0.3, 0.4].as_ref());
        let rain = DataFrame::new_from_columns(vec![s0, s1]).unwrap();

        let joined = temp.inner_join(&rain, "days", "days").unwrap();

        let join_col_days = Series::init("days", [1, 2, 1].as_ref());
        let join_col_temp = Series::init("temp", [19.9, 7., 19.9].as_ref());
        let join_col_rain = Series::init("rain", [0.1, 0.3, 0.1].as_ref());
        let join_col_rain_right = Series::init("rain_right", [0.1, 0.2, 0.4].as_ref());
        let true_df = DataFrame::new_from_columns(vec![
            join_col_days,
            join_col_temp,
            join_col_rain,
            join_col_rain_right,
        ])
        .unwrap();

        assert!(joined.frame_equal(&true_df));
        println!("{}", joined)
    }

    #[test]
    fn test_left_join() {
        let s0 = Series::init("days", [0, 1, 2, 3, 4].as_ref());
        let s1 = Series::init("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
        let temp = DataFrame::new_from_columns(vec![s0, s1]).unwrap();

        let s0 = Series::init("days", [1, 2].as_ref());
        let s1 = Series::init("rain", [0.1, 0.2].as_ref());
        let rain = DataFrame::new_from_columns(vec![s0, s1]).unwrap();
        let joined = temp.left_join(&rain, "days", "days");
        println!("{}", joined.unwrap())
    }
}
