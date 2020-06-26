use crate::prelude::*;
use arrow::compute::TakeOptions;
use arrow::datatypes::ArrowPrimitiveType;
use fnv::{FnvBuildHasher, FnvHashMap};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

macro_rules! hash_join_inner {
    ($s_right:ident, $ca_left:ident, $type_:ident) => {{
        // call the type method series.i32()
        let ca_right = $s_right.$type_()?;
        $ca_left.hash_join_inner(ca_right)
    }};
}

macro_rules! hash_join_left {
    ($s_right:ident, $ca_left:ident, $type_:ident) => {{
        // call the type method series.i32()
        let ca_right = $s_right.$type_()?;
        $ca_left.hash_join_left(ca_right)
    }};
}

macro_rules! apply_hash_join_on_series {
    ($s_left:ident, $s_right:ident, $join_macro:ident) => {{
        match $s_left {
            Series::UInt32(ca_left) => $join_macro!($s_right, ca_left, u32),
            Series::Int32(ca_left) => $join_macro!($s_right, ca_left, i32),
            Series::Int64(ca_left) => $join_macro!($s_right, ca_left, i64),
            Series::Bool(ca_left) => $join_macro!($s_right, ca_left, bool),
            Series::Utf8(ca_left) => $join_macro!($s_right, ca_left, utf8),
            _ => unimplemented!(),
        }
    }};
}

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

pub(crate) fn prepare_hashed_relation_non_null<T>(
    b: impl Iterator<Item = T>,
) -> HashMap<T, Vec<usize>, FnvBuildHasher>
where
    T: Hash + Eq + Copy,
{
    let mut hash_tbl = FnvHashMap::default();

    b.enumerate()
        .for_each(|(idx, key)| hash_tbl.entry(key).or_insert_with(Vec::new).push(idx));
    hash_tbl
}

/// Hash join a and b.
///     b should be the shorter relation.
fn hash_join_tuples_inner<T>(
    a: impl Iterator<Item = Option<T>>,
    b: impl Iterator<Item = Option<T>>,
    // Because b should be the shorter relation we could need to swap to keep left left and right right.
    swap: bool,
) -> Vec<(usize, usize)>
where
    T: Hash + Eq + Copy,
{
    let hash_tbl = prepare_hashed_relation(b);

    let mut results = Vec::new();
    a.enumerate().for_each(|(idx_a, o)| {
        if let Some(key) = o {
            if let Some(indexes_b) = hash_tbl.get(&key) {
                let tuples =
                    indexes_b.iter().map(
                        |&idx_b| {
                            if swap {
                                (idx_b, idx_a)
                            } else {
                                (idx_a, idx_b)
                            }
                        },
                    );
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
    fn hash_join_inner(&self, other: &ChunkedArray<T>) -> Vec<(usize, usize)>;
    fn hash_join_left(&self, other: &ChunkedArray<T>) -> Vec<(usize, Option<usize>)>;
}

macro_rules! create_join_tuples {
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

        (a, b, !left_first)
    }};
}

impl<T> HashJoin<T> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Eq + Hash,
{
    fn hash_join_inner(&self, other: &ChunkedArray<T>) -> Vec<(usize, usize)> {
        let (a, b, swap) = create_join_tuples!(self, other);

        let a = if let Ok(slice) = a.cont_slice() {
            slice.iter().map(|v| Some(v))
        } else {
            a.iter()
        };
        // Create the join tuples
        hash_join_tuples_inner(a.iter(), b.iter(), swap)
    }

    fn hash_join_left(&self, other: &ChunkedArray<T>) -> Vec<(usize, Option<usize>)> {
        hash_join_left(self.iter(), other.iter())
    }
}

impl HashJoin<Utf8Type> for Utf8Chunked {
    fn hash_join_inner(&self, other: &Utf8Chunked) -> Vec<(usize, usize)> {
        let (a, b, swap) = create_join_tuples!(self, other);
        // Create the join tuples
        hash_join_tuples_inner(a.iter().map(|v| Some(v)), b.iter().map(|v| Some(v)), swap)
    }

    fn hash_join_left(&self, other: &Utf8Chunked) -> Vec<(usize, Option<usize>)> {
        hash_join_left(self.iter().map(|v| Some(v)), other.iter().map(|v| Some(v)))
    }
}

impl DataFrame {
    /// Utility method to finish a join.
    fn finish_join(
        &self,
        mut df_left: DataFrame,
        mut df_right: DataFrame,
        right_on: &str,
    ) -> Result<DataFrame> {
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

        df_left.hstack(&df_right.columns)?;
        Ok(df_left)
    }

    fn create_left_df<B>(&self, join_tuples: &[(usize, B)]) -> Result<DataFrame> {
        self.take_iter(
            join_tuples.iter().map(|(left, _right)| Some(*left)),
            Some(TakeOptions::default()),
            Some(join_tuples.len()),
        )
    }

    /// Perform an inner join on two DataFrames.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> Result<DataFrame> {
    ///     left.inner_join(right, "join_column_left", "join_column_right")
    /// }
    /// ```
    pub fn inner_join(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
    ) -> Result<DataFrame> {
        let s_left = self.select(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.select(right_on).ok_or(PolarsError::NotFound)?;
        let join_tuples = apply_hash_join_on_series!(s_left, s_right, hash_join_inner);

        let df_left = self.create_left_df(&join_tuples)?;
        let df_right = other.take_iter(
            join_tuples.iter().map(|(_left, right)| Some(*right)),
            Some(TakeOptions::default()),
            Some(join_tuples.len()),
        )?;

        self.finish_join(df_left, df_right, right_on)
    }

    /// Perform a left join on two DataFrames
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> Result<DataFrame> {
    ///     left.left_join(right, "join_column_left", "join_column_right")
    /// }
    /// ```
    pub fn left_join(&self, other: &DataFrame, left_on: &str, right_on: &str) -> Result<DataFrame> {
        let s_left = self.select(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.select(right_on).ok_or(PolarsError::NotFound)?;

        let opt_join_tuples: Vec<(usize, Option<usize>)> =
            apply_hash_join_on_series!(s_left, s_right, hash_join_left);
        let df_left = self.create_left_df(&opt_join_tuples)?;
        let df_right = other.take_iter(
            opt_join_tuples.iter().map(|(_left, right)| *right),
            Some(TakeOptions::default()),
            Some(opt_join_tuples.len()),
        )?;
        self.finish_join(df_left, df_right, right_on)
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
        let joined = temp.left_join(&rain, "days", "days").unwrap();
        println!("{}", &joined);
        assert_eq!(
            (joined.f_select("rain").sum::<f32>().unwrap() * 10.).round(),
            3.
        );
        assert_eq!(joined.f_select("rain").null_count(), 3)
    }
}
