use crate::prelude::*;
use crossbeam::thread;
use fnv::{FnvBuildHasher, FnvHashMap};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use unsafe_unwrap::UnsafeUnwrap;

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

macro_rules! hash_join_outer {
    ($s_right:ident, $ca_left:ident, $type_:ident) => {{
        // call the type method series.i32()
        let ca_right = $s_right.$type_()?;
        $ca_left.hash_join_outer(ca_right)
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
/// NOTE that T also can be an Option<T>. Nulls are seen as equal.
fn hash_join_tuples_inner<T>(
    a: impl Iterator<Item = T>,
    b: impl Iterator<Item = T>,
    // Because b should be the shorter relation we could need to swap to keep left left and right right.
    swap: bool,
) -> Vec<(usize, usize)>
where
    T: Hash + Eq + Copy,
{
    let mut results = Vec::new();
    // First we hash one relation
    let hash_tbl = prepare_hashed_relation(b);

    // Next we probe the other relation in the hash table
    a.enumerate().for_each(|(idx_a, key)| {
        if let Some(indexes_b) = hash_tbl.get(&key) {
            let tuples = indexes_b
                .iter()
                .map(|&idx_b| if swap { (idx_b, idx_a) } else { (idx_a, idx_b) });
            results.extend(tuples)
        }
    });
    results
}

/// Hash join left. None/ Nulls are regarded as Equal
/// All left values are joined so no Option<usize> there.
fn hash_join_tuples_left<T>(
    a: impl Iterator<Item = T>,
    b: impl Iterator<Item = T>,
) -> Vec<(usize, Option<usize>)>
where
    T: Hash + Eq + Copy,
{
    let mut results = Vec::new();
    // First we hash one relation
    let hash_tbl = prepare_hashed_relation(b);

    // Next we probe the other relation in the hash table
    a.enumerate().for_each(|(idx_a, key)| {
        match hash_tbl.get(&key) {
            // left and right matches
            Some(indexes_b) => results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b)))),
            // only left values, right = null
            None => results.push((idx_a, None)),
        }
    });
    results
}

/// Hash join outer. Both left and right can have no match so Options
/// We accept a closure as we need to do two passes over the same iterators.
fn hash_join_tuples_outer<'a, T, I, J>(
    a: I,
    b: J,
    swap: bool,
) -> HashSet<(Option<usize>, Option<usize>), FnvBuildHasher>
where
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
    T: Hash + Eq + Copy + Sync,
{
    let mut results = HashSet::with_capacity_and_hasher(
        a.size_hint().0 + b.size_hint().0,
        FnvBuildHasher::default(),
    );

    // prepare hash table
    let mut hash_tbl = prepare_hashed_relation(b);

    // probe the hash table.
    // Note: indexes from b that are not matched will be None, Some(idx_b)
    // Therefore we remove the matches and the remaining will be joined from the right
    a.enumerate().for_each(|(idx_a, key)| {
        match hash_tbl.remove(&key) {
            // left and right matches
            Some(indexes_b) => results.extend(indexes_b.iter().map(|&idx_b| {
                if swap {
                    (Some(idx_b), Some(idx_a))
                } else {
                    (Some(idx_a), Some(idx_b))
                }
            })),
            // only left values, right = null
            None => {
                results.insert(if swap {
                    (None, Some(idx_a))
                } else {
                    (Some(idx_a), None)
                });
            }
        }
    });
    hash_tbl.iter().for_each(|(_k, indexes_b)| {
        // remaining joined values from the right table
        results.extend(indexes_b.iter().map(|&idx_b| {
            if swap {
                (Some(idx_b), None)
            } else {
                (None, Some(idx_b))
            }
        }))
    });

    results
}

pub trait HashJoin<T> {
    fn hash_join_inner(&self, other: &ChunkedArray<T>) -> Vec<(usize, usize)>;
    fn hash_join_left(&self, other: &ChunkedArray<T>) -> Vec<(usize, Option<usize>)>;
    fn hash_join_outer(
        &self,
        other: &ChunkedArray<T>,
    ) -> HashSet<(Option<usize>, Option<usize>), FnvBuildHasher>;
}

macro_rules! det_hash_prone_order {
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
    T: PolarsNumericType + Sync,
    T::Native: Eq + Hash,
{
    fn hash_join_inner(&self, other: &ChunkedArray<T>) -> Vec<(usize, usize)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        match (a.cont_slice(), b.cont_slice()) {
            (Ok(a_slice), Ok(b_slice)) => {
                hash_join_tuples_inner(a_slice.iter(), b_slice.iter(), swap)
            }
            (Ok(a_slice), Err(_)) => {
                hash_join_tuples_inner(
                    a_slice.iter().map(|v| Some(*v)), // take ownership
                    b.into_iter(),
                    swap,
                )
            }
            (Err(_), Ok(b_slice)) => {
                hash_join_tuples_inner(a.into_iter(), b_slice.iter().map(|v| Some(*v)), swap)
            }
            (Err(_), Err(_)) => hash_join_tuples_inner(a.into_iter(), b.into_iter(), swap),
        }
    }

    fn hash_join_left(&self, other: &ChunkedArray<T>) -> Vec<(usize, Option<usize>)> {
        match (self.cont_slice(), other.cont_slice()) {
            (Ok(a_slice), Ok(b_slice)) => hash_join_tuples_left(a_slice.iter(), b_slice.iter()),
            (Ok(a_slice), Err(_)) => {
                hash_join_tuples_left(
                    a_slice.iter().map(|v| Some(*v)), // take ownership
                    other.into_iter(),
                )
            }
            (Err(_), Ok(b_slice)) => {
                hash_join_tuples_left(self.into_iter(), b_slice.iter().map(|v| Some(*v)))
            }
            (Err(_), Err(_)) => hash_join_tuples_left(self.into_iter(), other.into_iter()),
        }
    }

    fn hash_join_outer(
        &self,
        other: &ChunkedArray<T>,
    ) -> HashSet<(Option<usize>, Option<usize>), FnvBuildHasher> {
        let (a, b, swap) = det_hash_prone_order!(self, other);
        match (a.cont_slice(), b.cont_slice()) {
            (Ok(a_slice), Ok(b_slice)) => {
                hash_join_tuples_outer(a_slice.iter(), b_slice.iter(), swap)
            }
            (Ok(a_slice), Err(_)) => {
                hash_join_tuples_outer(
                    a_slice.iter().map(|v| Some(*v)), // take ownership
                    b.into_iter(),
                    swap,
                )
            }
            (Err(_), Ok(b_slice)) => hash_join_tuples_outer(
                a.into_iter(),
                b_slice.iter().map(|v: &T::Native| Some(*v)),
                swap,
            ),
            (Err(_), Err(_)) => hash_join_tuples_outer(a.into_iter(), b.into_iter(), swap),
        }
    }
}

impl HashJoin<BooleanType> for BooleanChunked {
    fn hash_join_inner(&self, other: &BooleanChunked) -> Vec<(usize, usize)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);
        // Create the join tuples
        hash_join_tuples_inner(a.into_iter(), b.into_iter(), swap)
    }

    fn hash_join_left(&self, other: &BooleanChunked) -> Vec<(usize, Option<usize>)> {
        hash_join_tuples_left(self.into_iter(), other.into_iter())
    }

    fn hash_join_outer(
        &self,
        other: &BooleanChunked,
    ) -> HashSet<(Option<usize>, Option<usize>), FnvBuildHasher> {
        let (a, b, swap) = det_hash_prone_order!(self, other);
        hash_join_tuples_outer(a.into_iter(), b.into_iter(), swap)
    }
}

impl HashJoin<Utf8Type> for Utf8Chunked {
    fn hash_join_inner(&self, other: &Utf8Chunked) -> Vec<(usize, usize)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);
        // Create the join tuples
        hash_join_tuples_inner(a.into_iter(), b.into_iter(), swap)
    }

    fn hash_join_left(&self, other: &Utf8Chunked) -> Vec<(usize, Option<usize>)> {
        hash_join_tuples_left(self.into_iter(), other.into_iter())
    }

    fn hash_join_outer(
        &self,
        other: &Utf8Chunked,
    ) -> HashSet<(Option<usize>, Option<usize>), FnvBuildHasher> {
        let (a, b, swap) = det_hash_prone_order!(self, other);
        hash_join_tuples_outer(a.into_iter(), b.into_iter(), swap)
    }
}

impl DataFrame {
    /// Utility method to finish a join.
    fn finish_join(&self, mut df_left: DataFrame, mut df_right: DataFrame) -> Result<DataFrame> {
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

    fn create_left_df<B: Sync>(&self, join_tuples: &[(usize, B)]) -> Result<DataFrame> {
        self.take_iter(
            join_tuples.iter().map(|(left, _right)| Some(*left)),
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
        let s_left = self.column(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.column(right_on).ok_or(PolarsError::NotFound)?;
        let join_tuples = apply_hash_join_on_series!(s_left, s_right, hash_join_inner);

        let (df_left, df_right) = exec_concurrent!(
            { self.create_left_df(&join_tuples).expect("could not take") },
            {
                other
                    .drop_pure(right_on)
                    .unwrap()
                    .take_iter(
                        join_tuples.iter().map(|(_left, right)| Some(*right)),
                        Some(join_tuples.len()),
                    )
                    .expect("could not take")
            }
        );
        self.finish_join(df_left, df_right)
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
        let s_left = self.column(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.column(right_on).ok_or(PolarsError::NotFound)?;
        let opt_join_tuples: Vec<(usize, Option<usize>)> =
            apply_hash_join_on_series!(s_left, s_right, hash_join_left);

        let (df_left, df_right) = exec_concurrent!(
            {
                self.create_left_df(&opt_join_tuples)
                    .expect("could not take")
            },
            {
                other
                    .drop_pure(right_on)
                    .unwrap()
                    .take_iter(
                        opt_join_tuples.iter().map(|(_left, right)| *right),
                        Some(opt_join_tuples.len()),
                    )
                    .expect("could not take")
            }
        );
        self.finish_join(df_left, df_right)
    }

    /// Perform an outer join on two DataFrames
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> Result<DataFrame> {
    ///     left.outer_join(right, "join_column_left", "join_column_right")
    /// }
    /// ```
    pub fn outer_join(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
    ) -> Result<DataFrame> {
        let s_left = self.column(left_on).ok_or(PolarsError::NotFound)?;
        let s_right = other.column(right_on).ok_or(PolarsError::NotFound)?;

        // Get the indexes of the joined relations
        let opt_join_tuples: HashSet<(Option<usize>, Option<usize>), FnvBuildHasher> =
            apply_hash_join_on_series!(s_left, s_right, hash_join_outer);

        // Take the left and right dataframes by join tuples
        let (mut df_left, df_right) = exec_concurrent!(
            {
                self.drop_pure(left_on)
                    .unwrap()
                    .take_iter(
                        opt_join_tuples.iter().map(|(left, _right)| *left),
                        Some(opt_join_tuples.len()),
                    )
                    .expect("could not take")
            },
            {
                other
                    .drop_pure(right_on)
                    .unwrap()
                    .take_iter(
                        opt_join_tuples.iter().map(|(_left, right)| *right),
                        Some(opt_join_tuples.len()),
                    )
                    .expect("could not take")
            }
        );

        // Create the column used to join. This column has values from both left and right.
        macro_rules! downcast_and_replace_joined_column {
            ($type:ident) => {{
                let left_join_col = s_left.$type().unwrap();
                let right_join_col = s_right.$type().unwrap();

                let left_rand_access = left_join_col.take_rand();
                let right_rand_access = right_join_col.take_rand();

                let mut s: Series = opt_join_tuples
                    .iter()
                    .map(|(opt_left_idx, opt_right_idx)| {
                        if let Some(left_idx) = opt_left_idx {
                            unsafe { left_rand_access.get_unchecked(*left_idx) }
                        } else {
                            unsafe {
                                let right_idx = opt_right_idx.unsafe_unwrap();
                                right_rand_access.get_unchecked(right_idx)
                            }
                        }
                    })
                    .collect();
                s.rename(left_on);
                df_left.hstack(&[s])?;
            }};
        }

        match s_left.dtype() {
            ArrowDataType::UInt32 => downcast_and_replace_joined_column!(u32),
            ArrowDataType::Int32 => downcast_and_replace_joined_column!(i32),
            ArrowDataType::Int64 => downcast_and_replace_joined_column!(i64),
            ArrowDataType::Date32(DateUnit::Millisecond) => {
                downcast_and_replace_joined_column!(i32)
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                downcast_and_replace_joined_column!(i64)
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                downcast_and_replace_joined_column!(i64)
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => downcast_and_replace_joined_column!(i64),
            ArrowDataType::Boolean => downcast_and_replace_joined_column!(bool),
            ArrowDataType::Utf8 => downcast_and_replace_joined_column!(utf8),
            _ => unimplemented!(),
        }
        self.finish_join(df_left, df_right)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    fn create_frames() -> (DataFrame, DataFrame) {
        let s0 = Series::new("days", [0, 1, 2].as_ref());
        let s1 = Series::new("temp", [22.1, 19.9, 7.].as_ref());
        let s2 = Series::new("rain", [0.2, 0.1, 0.3].as_ref());
        let temp = DataFrame::new(vec![s0, s1, s2]).unwrap();

        let s0 = Series::new("days", [1, 2, 3, 1].as_ref());
        let s1 = Series::new("rain", [0.1, 0.2, 0.3, 0.4].as_ref());
        let rain = DataFrame::new(vec![s0, s1]).unwrap();
        (temp, rain)
    }

    #[test]
    fn test_inner_join() {
        let (temp, rain) = create_frames();
        let joined = temp.inner_join(&rain, "days", "days").unwrap();

        let join_col_days = Series::new("days", [1, 2, 1].as_ref());
        let join_col_temp = Series::new("temp", [19.9, 7., 19.9].as_ref());
        let join_col_rain = Series::new("rain", [0.1, 0.3, 0.1].as_ref());
        let join_col_rain_right = Series::new("rain_right", [0.1, 0.2, 0.4].as_ref());
        let true_df = DataFrame::new(vec![
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
        let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
        let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
        let temp = DataFrame::new(vec![s0, s1]).unwrap();

        let s0 = Series::new("days", [1, 2].as_ref());
        let s1 = Series::new("rain", [0.1, 0.2].as_ref());
        let rain = DataFrame::new(vec![s0, s1]).unwrap();
        let joined = temp.left_join(&rain, "days", "days").unwrap();
        println!("{}", &joined);
        assert_eq!(
            (joined.f_column("rain").sum::<f32>().unwrap() * 10.).round(),
            3.
        );
        assert_eq!(joined.f_column("rain").null_count(), 3)
    }

    #[test]
    fn test_outer_join() {
        let (temp, rain) = create_frames();
        let joined = temp.outer_join(&rain, "days", "days").unwrap();
        println!("{:?}", &joined);
        assert_eq!(joined.height(), 5);
        assert_eq!(joined.column("days").unwrap().sum::<i32>(), Some(7));
    }
}
