#[cfg(feature = "merge_sorted")]
mod merge_sorted;
#[cfg(feature = "chunked_ids")]
use std::borrow::Cow;

#[cfg(feature = "merge_sorted")]
pub use merge_sorted::_merge_sorted_dfs;
use polars_core::frame::hash_join::*;
use polars_core::prelude::*;
use polars_core::utils::{_to_physical_and_bit_repr, slice_slice};
use polars_core::POOL;

use super::*;

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

pub trait DataFrameJoinOps: IntoDf {
    /// Generic join method. Can be used to join on multiple columns.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Fruit" => &["Apple", "Banana", "Pear"],
    ///                          "Phosphorus (mg/100g)" => &[11, 22, 12])?;
    /// let df2: DataFrame = df!("Name" => &["Apple", "Banana", "Pear"],
    ///                          "Potassium (mg/100g)" => &[107, 358, 115])?;
    ///
    /// let df3: DataFrame = df1.join(&df2, ["Fruit"], ["Name"], JoinType::Inner, None)?;
    /// assert_eq!(df3.shape(), (3, 3));
    /// println!("{}", df3);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (3, 3)
    /// +--------+----------------------+---------------------+
    /// | Fruit  | Phosphorus (mg/100g) | Potassium (mg/100g) |
    /// | ---    | ---                  | ---                 |
    /// | str    | i32                  | i32                 |
    /// +========+======================+=====================+
    /// | Apple  | 11                   | 107                 |
    /// +--------+----------------------+---------------------+
    /// | Banana | 22                   | 358                 |
    /// +--------+----------------------+---------------------+
    /// | Pear   | 12                   | 115                 |
    /// +--------+----------------------+---------------------+
    /// ```
    fn join<I, S>(
        &self,
        other: &DataFrame,
        left_on: I,
        right_on: I,
        how: JoinType,
        suffix: Option<String>,
    ) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let df_left = self.to_df();
        #[cfg(feature = "cross_join")]
        if let JoinType::Cross = how {
            return df_left.cross_join(other, suffix.as_deref(), None);
        }
        let selected_left = df_left.select_series(left_on)?;
        let selected_right = other.select_series(right_on)?;
        self._join_impl(
            other,
            selected_left,
            selected_right,
            how,
            suffix,
            None,
            true,
            false,
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn _join_impl(
        &self,
        other: &DataFrame,
        selected_left: Vec<Series>,
        selected_right: Vec<Series>,
        how: JoinType,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
        _check_rechunk: bool,
        _verbose: bool,
    ) -> PolarsResult<DataFrame> {
        let left_df = self.to_df();

        #[cfg(feature = "cross_join")]
        if let JoinType::Cross = how {
            return left_df.cross_join(other, suffix.as_deref(), slice);
        }

        #[cfg(feature = "chunked_ids")]
        {
            // a left join create chunked-ids
            // the others not yet.
            // TODO! change this to other join types once they support chunked-id joins
            if _check_rechunk
                && !(matches!(how, JoinType::Left)
                    || std::env::var("POLARS_NO_CHUNKED_JOIN").is_ok())
            {
                let mut left = Cow::Borrowed(left_df);
                let mut right = Cow::Borrowed(other);
                if left_df.should_rechunk() {
                    if _verbose {
                        eprintln!("{:?} join triggered a rechunk of the left dataframe: {} columns are affected", how, left_df.width());
                    }

                    let mut tmp_left = left_df.clone();
                    tmp_left.as_single_chunk_par();
                    left = Cow::Owned(tmp_left);
                }
                if other.should_rechunk() {
                    if _verbose {
                        eprintln!("{:?} join triggered a rechunk of the right dataframe: {} columns are affected", how, other.width());
                    }
                    let mut tmp_right = other.clone();
                    tmp_right.as_single_chunk_par();
                    right = Cow::Owned(tmp_right);
                }
                return left._join_impl(
                    &right,
                    selected_left,
                    selected_right,
                    how,
                    suffix,
                    slice,
                    false,
                    _verbose,
                );
            }
        }

        if selected_right.len() != selected_left.len() {
            return Err(PolarsError::ComputeError(
                "the number of columns given as join key should be equal".into(),
            ));
        }
        if selected_left
            .iter()
            .zip(&selected_right)
            .any(|(l, r)| l.dtype() != r.dtype())
        {
            return Err(PolarsError::ComputeError("the dtype of the join keys don't match. first cast your columns to the correct dtype".into()));
        }

        #[cfg(feature = "dtype-categorical")]
        for (l, r) in selected_left.iter().zip(&selected_right) {
            _check_categorical_src(l.dtype(), r.dtype())?
        }

        // Single keys
        if selected_left.len() == 1 {
            let s_left = left_df.column(selected_left[0].name())?;
            let s_right = other.column(selected_right[0].name())?;
            return match how {
                JoinType::Inner => {
                    left_df._inner_join_from_series(other, s_left, s_right, suffix, slice, _verbose)
                }
                JoinType::Left => {
                    left_df._left_join_from_series(other, s_left, s_right, suffix, slice, _verbose)
                }
                JoinType::Outer => {
                    left_df._outer_join_from_series(other, s_left, s_right, suffix, slice)
                }
                #[cfg(feature = "semi_anti_join")]
                JoinType::Anti => left_df._semi_anti_join_from_series(s_left, s_right, slice, true),
                #[cfg(feature = "semi_anti_join")]
                JoinType::Semi => {
                    left_df._semi_anti_join_from_series(s_left, s_right, slice, false)
                }
                #[cfg(feature = "asof_join")]
                JoinType::AsOf(options) => {
                    let left_on = selected_left[0].name();
                    let right_on = selected_right[0].name();

                    match (options.left_by, options.right_by) {
                        (Some(left_by), Some(right_by)) => left_df._join_asof_by(
                            other,
                            left_on,
                            right_on,
                            left_by,
                            right_by,
                            options.strategy,
                            options.tolerance,
                            slice,
                        ),
                        (None, None) => left_df._join_asof(
                            other,
                            left_on,
                            right_on,
                            options.strategy,
                            options.tolerance,
                            suffix,
                            slice,
                        ),
                        _ => {
                            panic!("expected by arguments on both sides")
                        }
                    }
                }
                JoinType::Cross => {
                    unreachable!()
                }
            };
        }

        fn remove_selected(df: &DataFrame, selected: &[Series]) -> DataFrame {
            let mut new = None;
            for s in selected {
                new = match new {
                    None => Some(df.drop(s.name()).unwrap()),
                    Some(new) => Some(new.drop(s.name()).unwrap()),
                }
            }
            new.unwrap()
        }
        // make sure that we don't have logical types.
        // we don't overwrite the original selected as that might be used to create a column in the new df
        let selected_left_physical = _to_physical_and_bit_repr(&selected_left);
        let selected_right_physical = _to_physical_and_bit_repr(&selected_right);

        // multiple keys
        match how {
            JoinType::Inner => {
                let left = DataFrame::new_no_checks(selected_left_physical);
                let right = DataFrame::new_no_checks(selected_right_physical);
                let (mut left, mut right, swap) = det_hash_prone_order!(left, right);
                let (join_idx_left, join_idx_right) =
                    _inner_join_multiple_keys(&mut left, &mut right, swap);
                let mut join_idx_left = &*join_idx_left;
                let mut join_idx_right = &*join_idx_right;

                if let Some((offset, len)) = slice {
                    join_idx_left = slice_slice(join_idx_left, offset, len);
                    join_idx_right = slice_slice(join_idx_right, offset, len);
                }

                let (df_left, df_right) = POOL.join(
                    // safety: join indices are known to be in bounds
                    || unsafe { left_df._create_left_df_from_slice(join_idx_left, false, !swap) },
                    || unsafe {
                        // remove join columns
                        remove_selected(other, &selected_right)
                            ._take_unchecked_slice(join_idx_right, true)
                    },
                );
                _finish_join(df_left, df_right, suffix.as_deref())
            }
            JoinType::Left => {
                let mut left = DataFrame::new_no_checks(selected_left_physical);
                let mut right = DataFrame::new_no_checks(selected_right_physical);
                let ids = _left_join_multiple_keys(&mut left, &mut right, None, None);

                left_df._finish_left_join(
                    ids,
                    &remove_selected(other, &selected_right),
                    suffix,
                    slice,
                )
            }
            JoinType::Outer => {
                let left = DataFrame::new_no_checks(selected_left_physical);
                let right = DataFrame::new_no_checks(selected_right_physical);

                let (mut left, mut right, swap) = det_hash_prone_order!(left, right);
                let opt_join_tuples = _outer_join_multiple_keys(&mut left, &mut right, swap);

                let mut opt_join_tuples = &*opt_join_tuples;

                if let Some((offset, len)) = slice {
                    opt_join_tuples = slice_slice(opt_join_tuples, offset, len);
                }

                // Take the left and right dataframes by join tuples
                let (df_left, df_right) = POOL.join(
                    || unsafe {
                        remove_selected(left_df, &selected_left).take_opt_iter_unchecked(
                            opt_join_tuples
                                .iter()
                                .map(|(left, _right)| left.map(|i| i as usize)),
                        )
                    },
                    || unsafe {
                        remove_selected(other, &selected_right).take_opt_iter_unchecked(
                            opt_join_tuples
                                .iter()
                                .map(|(_left, right)| right.map(|i| i as usize)),
                        )
                    },
                );
                // Allocate a new vec for df_left so that the keys are left and then other values.
                let mut keys = Vec::with_capacity(selected_left.len() + df_left.width());
                for (s_left, s_right) in selected_left.iter().zip(&selected_right) {
                    let mut s = s_left.zip_outer_join_column(s_right, opt_join_tuples);
                    s.rename(s_left.name());
                    keys.push(s)
                }
                keys.extend_from_slice(df_left.get_columns());
                let df_left = DataFrame::new_no_checks(keys);
                _finish_join(df_left, df_right, suffix.as_deref())
            }
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(_) => Err(PolarsError::ComputeError(
                "asof join not supported for join on multiple keys".into(),
            )),
            #[cfg(feature = "semi_anti_join")]
            JoinType::Anti | JoinType::Semi => {
                let mut left = DataFrame::new_no_checks(selected_left_physical);
                let mut right = DataFrame::new_no_checks(selected_right_physical);

                let idx = if matches!(how, JoinType::Anti) {
                    _left_anti_multiple_keys(&mut left, &mut right)
                } else {
                    _left_semi_multiple_keys(&mut left, &mut right)
                };
                // Safety:
                // indices are in bounds
                Ok(unsafe { left_df._finish_anti_semi_join(&idx, slice) })
            }
            JoinType::Cross => {
                unreachable!()
            }
        }
    }

    /// Perform an inner join on two DataFrames.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> PolarsResult<DataFrame> {
    ///     left.inner_join(right, ["join_column_left"], ["join_column_right"])
    /// }
    /// ```
    fn inner_join<I, S>(
        &self,
        other: &DataFrame,
        left_on: I,
        right_on: I,
    ) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.join(other, left_on, right_on, JoinType::Inner, None)
    }

    /// Perform a left join on two DataFrames
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Wavelength (nm)" => &[480.0, 650.0, 577.0, 1201.0, 100.0])?;
    /// let df2: DataFrame = df!("Color" => &["Blue", "Yellow", "Red"],
    ///                          "Wavelength nm" => &[480.0, 577.0, 650.0])?;
    ///
    /// let df3: DataFrame = df1.left_join(&df2, ["Wavelength (nm)"], ["Wavelength nm"])?;
    /// println!("{:?}", df3);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (5, 2)
    /// +-----------------+--------+
    /// | Wavelength (nm) | Color  |
    /// | ---             | ---    |
    /// | f64             | str    |
    /// +=================+========+
    /// | 480             | Blue   |
    /// +-----------------+--------+
    /// | 650             | Red    |
    /// +-----------------+--------+
    /// | 577             | Yellow |
    /// +-----------------+--------+
    /// | 1201            | null   |
    /// +-----------------+--------+
    /// | 100             | null   |
    /// +-----------------+--------+
    /// ```
    fn left_join<I, S>(&self, other: &DataFrame, left_on: I, right_on: I) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.join(other, left_on, right_on, JoinType::Left, None)
    }

    /// Perform an outer join on two DataFrames
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> PolarsResult<DataFrame> {
    ///     left.outer_join(right, ["join_column_left"], ["join_column_right"])
    /// }
    /// ```
    fn outer_join<I, S>(
        &self,
        other: &DataFrame,
        left_on: I,
        right_on: I,
    ) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.join(other, left_on, right_on, JoinType::Outer, None)
    }
}

trait DataFrameJoinOpsPrivate: IntoDf {
    // hack for a macro
    fn len(&self) -> usize {
        self.to_df().height()
    }

    fn _inner_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
        verbose: bool,
    ) -> PolarsResult<DataFrame> {
        let left_df = self.to_df();
        #[cfg(feature = "dtype-categorical")]
        _check_categorical_src(s_left.dtype(), s_right.dtype())?;
        let ((join_tuples_left, join_tuples_right), sorted) =
            _sort_or_hash_inner(s_left, s_right, verbose);

        let mut join_tuples_left = &*join_tuples_left;
        let mut join_tuples_right = &*join_tuples_right;

        if let Some((offset, len)) = slice {
            join_tuples_left = slice_slice(join_tuples_left, offset, len);
            join_tuples_right = slice_slice(join_tuples_right, offset, len);
        }

        let (df_left, df_right) = POOL.join(
            // safety: join indices are known to be in bounds
            || unsafe { left_df._create_left_df_from_slice(join_tuples_left, false, sorted) },
            || unsafe {
                other
                    .drop(s_right.name())
                    .unwrap()
                    ._take_unchecked_slice(join_tuples_right, true)
            },
        );
        _finish_join(df_left, df_right, suffix.as_deref())
    }
}

impl DataFrameJoinOps for DataFrame {}
impl DataFrameJoinOpsPrivate for DataFrame {}
