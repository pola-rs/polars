use crate::prelude::*;
use crate::utils::Xob;
use ahash::RandomState;
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
            Series::UInt8(ca_left) => $join_macro!($s_right, ca_left, u8),
            Series::UInt16(ca_left) => $join_macro!($s_right, ca_left, u16),
            Series::UInt32(ca_left) => $join_macro!($s_right, ca_left, u32),
            Series::UInt64(ca_left) => $join_macro!($s_right, ca_left, u64),
            Series::Int8(ca_left) => $join_macro!($s_right, ca_left, i8),
            Series::Int16(ca_left) => $join_macro!($s_right, ca_left, i16),
            Series::Int32(ca_left) => $join_macro!($s_right, ca_left, i32),
            Series::Int64(ca_left) => $join_macro!($s_right, ca_left, i64),
            Series::Bool(ca_left) => $join_macro!($s_right, ca_left, bool),
            Series::Utf8(ca_left) => $join_macro!($s_right, ca_left, utf8),
            Series::Date32(ca_left) => $join_macro!($s_right, ca_left, date32),
            Series::Date64(ca_left) => $join_macro!($s_right, ca_left, date64),
            Series::Time32Millisecond(ca_left) => {
                $join_macro!($s_right, ca_left, time32_millisecond)
            }
            Series::Time32Second(ca_left) => $join_macro!($s_right, ca_left, time32_second),
            Series::Time64Nanosecond(ca_left) => $join_macro!($s_right, ca_left, time64_nanosecond),
            Series::Time64Microsecond(ca_left) => {
                $join_macro!($s_right, ca_left, time64_microsecond)
            }
            Series::DurationMillisecond(ca_left) => {
                $join_macro!($s_right, ca_left, duration_millisecond)
            }
            Series::DurationSecond(ca_left) => $join_macro!($s_right, ca_left, duration_second),
            Series::DurationNanosecond(ca_left) => {
                $join_macro!($s_right, ca_left, duration_nanosecond)
            }
            Series::DurationMicrosecond(ca_left) => {
                $join_macro!($s_right, ca_left, duration_microsecond)
            }
            Series::TimestampMillisecond(ca_left) => {
                $join_macro!($s_right, ca_left, timestamp_millisecond)
            }
            Series::TimestampSecond(ca_left) => $join_macro!($s_right, ca_left, timestamp_second),
            Series::TimestampNanosecond(ca_left) => {
                $join_macro!($s_right, ca_left, timestamp_nanosecond)
            }
            Series::TimestampMicrosecond(ca_left) => {
                $join_macro!($s_right, ca_left, timestamp_microsecond)
            }
            Series::IntervalDayTime(ca_left) => $join_macro!($s_right, ca_left, interval_daytime),
            Series::IntervalYearMonth(ca_left) => {
                $join_macro!($s_right, ca_left, interval_year_month)
            }
            _ => unimplemented!(),
        }
    }};
}

pub(crate) fn prepare_hashed_relation<T>(
    b: impl Iterator<Item = T>,
) -> HashMap<T, Vec<usize>, RandomState>
where
    T: Hash + Eq,
{
    let mut hash_tbl: HashMap<T, Vec<usize>, ahash::RandomState> =
        HashMap::with_capacity_and_hasher(b.size_hint().0 / 10, RandomState::new());

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
    // code duplication is because we want to only do the swap check once
    if swap {
        a.enumerate().for_each(|(idx_a, key)| {
            if let Some(indexes_b) = hash_tbl.get(&key) {
                let tuples = indexes_b.iter().map(|&idx_b| (idx_b, idx_a));
                results.extend(tuples)
            }
        });
    } else {
        a.enumerate().for_each(|(idx_a, key)| {
            if let Some(indexes_b) = hash_tbl.get(&key) {
                let tuples = indexes_b.iter().map(|&idx_b| (idx_a, idx_b));
                results.extend(tuples)
            }
        });
    }
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
) -> Vec<(Option<usize>, Option<usize>)>
where
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
    T: Hash + Eq + Copy + Sync,
{
    let mut results = Vec::with_capacity(a.size_hint().0 + b.size_hint().0);

    // prepare hash table
    let mut hash_tbl = prepare_hashed_relation(b);

    // probe the hash table.
    // Note: indexes from b that are not matched will be None, Some(idx_b)
    // Therefore we remove the matches and the remaining will be joined from the right

    // code duplication is because we want to only do the swap check once
    if swap {
        a.enumerate().for_each(|(idx_a, key)| {
            match hash_tbl.remove(&key) {
                // left and right matches
                Some(indexes_b) => {
                    results.extend(indexes_b.iter().map(|&idx_b| (Some(idx_b), Some(idx_a))))
                }
                // only left values, right = null
                None => {
                    results.push((None, Some(idx_a)));
                }
            }
        });
        hash_tbl.iter().for_each(|(_k, indexes_b)| {
            // remaining joined values from the right table
            results.extend(indexes_b.iter().map(|&idx_b| (Some(idx_b), None)))
        });
    } else {
        a.enumerate().for_each(|(idx_a, key)| {
            match hash_tbl.remove(&key) {
                // left and right matches
                Some(indexes_b) => {
                    results.extend(indexes_b.iter().map(|&idx_b| (Some(idx_a), Some(idx_b))))
                }
                // only left values, right = null
                None => {
                    results.push((Some(idx_a), None));
                }
            }
        });
        hash_tbl.iter().for_each(|(_k, indexes_b)| {
            // remaining joined values from the right table
            results.extend(indexes_b.iter().map(|&idx_b| (None, Some(idx_b))))
        });
    };

    results
}

pub trait HashJoin<T> {
    fn hash_join_inner(&self, other: &ChunkedArray<T>) -> Vec<(usize, usize)>;
    fn hash_join_left(&self, other: &ChunkedArray<T>) -> Vec<(usize, Option<usize>)>;
    fn hash_join_outer(&self, other: &ChunkedArray<T>) -> Vec<(Option<usize>, Option<usize>)>;
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

    fn hash_join_outer(&self, other: &ChunkedArray<T>) -> Vec<(Option<usize>, Option<usize>)> {
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
        match (a.is_optimal_aligned(), b.is_optimal_aligned()) {
            (true, true) => {
                hash_join_tuples_inner(a.into_no_null_iter(), b.into_no_null_iter(), swap)
            }
            _ => hash_join_tuples_inner(a.into_iter(), b.into_iter(), swap),
        }
    }

    fn hash_join_left(&self, other: &BooleanChunked) -> Vec<(usize, Option<usize>)> {
        match (self.is_optimal_aligned(), other.is_optimal_aligned()) {
            (true, true) => {
                hash_join_tuples_left(self.into_no_null_iter(), other.into_no_null_iter())
            }
            _ => hash_join_tuples_left(self.into_iter(), other.into_iter()),
        }
    }

    fn hash_join_outer(&self, other: &BooleanChunked) -> Vec<(Option<usize>, Option<usize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);
        match (a.is_optimal_aligned(), b.is_optimal_aligned()) {
            (true, true) => {
                hash_join_tuples_outer(a.into_no_null_iter(), b.into_no_null_iter(), swap)
            }
            _ => hash_join_tuples_outer(a.into_iter(), b.into_iter(), swap),
        }
    }
}

impl HashJoin<Utf8Type> for Utf8Chunked {
    fn hash_join_inner(&self, other: &Utf8Chunked) -> Vec<(usize, usize)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);

        // Create the join tuples
        match (a.is_optimal_aligned(), b.is_optimal_aligned()) {
            (true, true) => {
                hash_join_tuples_inner(a.into_no_null_iter(), b.into_no_null_iter(), swap)
            }
            _ => hash_join_tuples_inner(a.into_iter(), b.into_iter(), swap),
        }
    }

    fn hash_join_left(&self, other: &Utf8Chunked) -> Vec<(usize, Option<usize>)> {
        match (self.is_optimal_aligned(), other.is_optimal_aligned()) {
            (true, true) => {
                hash_join_tuples_left(self.into_no_null_iter(), other.into_no_null_iter())
            }
            _ => hash_join_tuples_left(self.into_iter(), other.into_iter()),
        }
    }

    fn hash_join_outer(&self, other: &Utf8Chunked) -> Vec<(Option<usize>, Option<usize>)> {
        let (a, b, swap) = det_hash_prone_order!(self, other);
        match (a.is_optimal_aligned(), b.is_optimal_aligned()) {
            (true, true) => {
                hash_join_tuples_outer(a.into_no_null_iter(), b.into_no_null_iter(), swap)
            }
            _ => hash_join_tuples_outer(a.into_iter(), b.into_iter(), swap),
        }
    }
}

trait ZipOuterJoinColumn {
    fn zip_outer_join_column(
        &self,
        _right_column: &Series,
        _opt_join_tuples: &[(Option<usize>, Option<usize>)],
    ) -> Series {
        unimplemented!()
    }
}

impl<T> ZipOuterJoinColumn for ChunkedArray<T>
where
    T: PolarsIntegerType,
{
    fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<usize>, Option<usize>)],
    ) -> Series {
        let right_ca = self.unpack_series_matching_type(right_column).unwrap();

        let left_rand_access = self.take_rand();
        let right_rand_access = right_ca.take_rand();

        opt_join_tuples
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
            .collect::<Xob<ChunkedArray<T>>>()
            .into_inner()
            .into_series()
    }
}

impl ZipOuterJoinColumn for Float32Chunked {}
impl ZipOuterJoinColumn for Float64Chunked {}
impl ZipOuterJoinColumn for ListChunked {}

macro_rules! impl_zip_outer_join {
    ($chunkedtype:ident) => {
        impl ZipOuterJoinColumn for $chunkedtype {
            fn zip_outer_join_column(
                &self,
                right_column: &Series,
                opt_join_tuples: &[(Option<usize>, Option<usize>)],
            ) -> Series {
                let right_ca = self.unpack_series_matching_type(right_column).unwrap();

                let left_rand_access = self.take_rand();
                let right_rand_access = right_ca.take_rand();

                opt_join_tuples
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
                    .collect::<$chunkedtype>()
                    .into_series()
            }
        }
    };
}
impl_zip_outer_join!(BooleanChunked);
impl_zip_outer_join!(Utf8Chunked);

impl DataFrame {
    /// Utility method to finish a join.
    fn finish_join(&self, mut df_left: DataFrame, mut df_right: DataFrame) -> Result<DataFrame> {
        let mut left_names = HashSet::with_capacity_and_hasher(df_left.width(), RandomState::new());

        df_left.columns.iter().for_each(|series| {
            left_names.insert(series.name());
        });

        let mut rename_strs = Vec::with_capacity(df_right.width());

        df_right.columns.iter().for_each(|series| {
            if left_names.contains(series.name()) {
                rename_strs.push(series.name().to_owned())
            }
        });

        for name in rename_strs {
            df_right.rename(&name, &format!("{}_right", name))?;
        }

        df_left.hstack_mut(&df_right.columns)?;
        Ok(df_left)
    }

    fn create_left_df<B: Sync>(&self, join_tuples: &[(usize, B)]) -> DataFrame {
        unsafe {
            self.take_iter_unchecked(
                join_tuples.iter().map(|(left, _right)| *left),
                Some(join_tuples.len()),
            )
        }
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
        let s_left = self.column(left_on)?;
        let s_right = other.column(right_on)?;
        self.inner_join_from_series(other, s_left, s_right)
    }

    pub(crate) fn inner_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
    ) -> Result<DataFrame> {
        let join_tuples = apply_hash_join_on_series!(s_left, s_right, hash_join_inner);

        let (df_left, df_right) = rayon::join(
            || self.create_left_df(&join_tuples),
            || unsafe {
                other.drop(s_right.name()).unwrap().take_iter_unchecked(
                    join_tuples.iter().map(|(_left, right)| *right),
                    Some(join_tuples.len()),
                )
            },
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
        let s_left = self.column(left_on)?;
        let s_right = other.column(right_on)?;
        self.left_join_from_series(other, s_left, s_right)
    }

    pub(crate) fn left_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
    ) -> Result<DataFrame> {
        let opt_join_tuples = apply_hash_join_on_series!(s_left, s_right, hash_join_left);

        let (df_left, df_right) = rayon::join(
            || self.create_left_df(&opt_join_tuples),
            || unsafe {
                other.drop(s_right.name()).unwrap().take_opt_iter_unchecked(
                    opt_join_tuples.iter().map(|(_left, right)| *right),
                    Some(opt_join_tuples.len()),
                )
            },
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
        let s_left = self.column(left_on)?;
        let s_right = other.column(right_on)?;
        self.outer_join_from_series(other, s_left, s_right)
    }
    pub(crate) fn outer_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
    ) -> Result<DataFrame> {
        // Get the indexes of the joined relations
        let opt_join_tuples: Vec<(Option<usize>, Option<usize>)> =
            apply_hash_join_on_series!(s_left, s_right, hash_join_outer);

        // Take the left and right dataframes by join tuples
        let (mut df_left, df_right) = rayon::join(
            || unsafe {
                self.drop(s_left.name()).unwrap().take_opt_iter_unchecked(
                    opt_join_tuples.iter().map(|(left, _right)| *left),
                    Some(opt_join_tuples.len()),
                )
            },
            || unsafe {
                other.drop(s_right.name()).unwrap().take_opt_iter_unchecked(
                    opt_join_tuples.iter().map(|(_left, right)| *right),
                    Some(opt_join_tuples.len()),
                )
            },
        );
        let mut s =
            apply_method_all_series!(s_left, zip_outer_join_column, s_right, &opt_join_tuples);
        s.rename(s_left.name());
        df_left.hstack_mut(&[s])?;
        self.finish_join(df_left, df_right)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    fn create_frames() -> (DataFrame, DataFrame) {
        let s0 = Series::new("days", &[0, 1, 2]);
        let s1 = Series::new("temp", &[22.1, 19.9, 7.]);
        let s2 = Series::new("rain", &[0.2, 0.1, 0.3]);
        let temp = DataFrame::new(vec![s0, s1, s2]).unwrap();

        let s0 = Series::new("days", &[1, 2, 3, 1]);
        let s1 = Series::new("rain", &[0.1, 0.2, 0.3, 0.4]);
        let rain = DataFrame::new(vec![s0, s1]).unwrap();
        (temp, rain)
    }

    #[test]
    fn test_inner_join() {
        let (temp, rain) = create_frames();
        let joined = temp.inner_join(&rain, "days", "days").unwrap();

        let join_col_days = Series::new("days", &[1, 2, 1]);
        let join_col_temp = Series::new("temp", &[19.9, 7., 19.9]);
        let join_col_rain = Series::new("rain", &[0.1, 0.3, 0.1]);
        let join_col_rain_right = Series::new("rain_right", [0.1, 0.2, 0.4].as_ref());
        let true_df = DataFrame::new(vec![
            join_col_days,
            join_col_temp,
            join_col_rain,
            join_col_rain_right,
        ])
        .unwrap();

        println!("{}", joined);
        assert!(joined.frame_equal(&true_df));
    }

    #[test]
    fn test_left_join() {
        let s0 = Series::new("days", &[0, 1, 2, 3, 4]);
        let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
        let temp = DataFrame::new(vec![s0, s1]).unwrap();

        let s0 = Series::new("days", &[1, 2]);
        let s1 = Series::new("rain", &[0.1, 0.2]);
        let rain = DataFrame::new(vec![s0, s1]).unwrap();
        let joined = temp.left_join(&rain, "days", "days").unwrap();
        println!("{}", &joined);
        assert_eq!(
            (joined.column("rain").unwrap().sum::<f32>().unwrap() * 10.).round(),
            3.
        );
        assert_eq!(joined.column("rain").unwrap().null_count(), 3);

        // test join on utf8
        let s0 = Series::new("days", &["mo", "tue", "wed", "thu", "fri"]);
        let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
        let temp = DataFrame::new(vec![s0, s1]).unwrap();

        let s0 = Series::new("days", &["tue", "wed"]);
        let s1 = Series::new("rain", &[0.1, 0.2]);
        let rain = DataFrame::new(vec![s0, s1]).unwrap();
        let joined = temp.left_join(&rain, "days", "days").unwrap();
        println!("{}", &joined);
        assert_eq!(
            (joined.column("rain").unwrap().sum::<f32>().unwrap() * 10.).round(),
            3.
        );
        assert_eq!(joined.column("rain").unwrap().null_count(), 3);
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
