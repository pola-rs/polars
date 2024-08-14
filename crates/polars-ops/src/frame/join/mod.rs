mod args;
#[cfg(feature = "asof_join")]
mod asof;
#[cfg(feature = "dtype-categorical")]
mod checks;
mod cross_join;
mod dispatch_left_right;
mod general;
mod hash_join;
#[cfg(feature = "merge_sorted")]
mod merge_sorted;

use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

pub use args::*;
use arrow::trusted_len::TrustedLen;
#[cfg(feature = "asof_join")]
pub use asof::{AsOfOptions, AsofJoin, AsofJoinBy, AsofStrategy};
#[cfg(feature = "dtype-categorical")]
pub(crate) use checks::*;
pub use cross_join::CrossJoin;
#[cfg(feature = "chunked_ids")]
use either::Either;
#[cfg(feature = "chunked_ids")]
use general::create_chunked_index_mapping;
pub use general::{_coalesce_full_join, _finish_join, _join_suffix_name};
pub use hash_join::*;
use hashbrown::hash_map::{Entry, RawEntryMut};
#[cfg(feature = "merge_sorted")]
pub use merge_sorted::_merge_sorted_dfs;
use polars_core::hashing::_HASHMAP_INIT_SIZE;
#[allow(unused_imports)]
use polars_core::prelude::sort::arg_sort_multiple::{
    encode_rows_vertical_par_unordered, encode_rows_vertical_par_unordered_broadcast_nulls,
};
use polars_core::prelude::*;
pub(super) use polars_core::series::IsSorted;
use polars_core::utils::slice_offsets;
#[allow(unused_imports)]
use polars_core::utils::slice_slice;
use polars_core::POOL;
use polars_utils::hashing::BytesHash;
use rayon::prelude::*;

use super::IntoDf;

pub trait DataFrameJoinOps: IntoDf {
    /// Generic join method. Can be used to join on multiple columns.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// # use polars_ops::prelude::*;
    /// let df1: DataFrame = df!("Fruit" => &["Apple", "Banana", "Pear"],
    ///                          "Phosphorus (mg/100g)" => &[11, 22, 12])?;
    /// let df2: DataFrame = df!("Name" => &["Apple", "Banana", "Pear"],
    ///                          "Potassium (mg/100g)" => &[107, 358, 115])?;
    ///
    /// let df3: DataFrame = df1.join(&df2, ["Fruit"], ["Name"], JoinArgs::new(JoinType::Inner))?;
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
        args: JoinArgs,
    ) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let df_left = self.to_df();
        let selected_left = df_left.select_series(left_on)?;
        let selected_right = other.select_series(right_on)?;
        self._join_impl(other, selected_left, selected_right, args, true, false)
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    #[allow(unused_mut)]
    fn _join_impl(
        &self,
        other: &DataFrame,
        mut selected_left: Vec<Series>,
        mut selected_right: Vec<Series>,
        mut args: JoinArgs,
        _check_rechunk: bool,
        _verbose: bool,
    ) -> PolarsResult<DataFrame> {
        let left_df = self.to_df();

        #[cfg(feature = "cross_join")]
        if let JoinType::Cross = args.how {
            return left_df.cross_join(other, args.suffix.as_deref(), args.slice);
        }

        // Clear literals if a frame is empty. Otherwise we could get an oob
        fn clear(s: &mut [Series]) {
            for s in s.iter_mut() {
                if s.len() == 1 {
                    *s = s.clear()
                }
            }
        }
        if left_df.is_empty() {
            clear(&mut selected_left);
        }
        if other.is_empty() {
            clear(&mut selected_right);
        }

        let should_coalesce = args.should_coalesce();
        assert_eq!(selected_left.len(), selected_right.len());

        #[cfg(feature = "chunked_ids")]
        {
            // a left join create chunked-ids
            // the others not yet.
            // TODO! change this to other join types once they support chunked-id joins
            if _check_rechunk
                && !(matches!(args.how, JoinType::Left)
                    || std::env::var("POLARS_NO_CHUNKED_JOIN").is_ok())
            {
                let mut left = Cow::Borrowed(left_df);
                let mut right = Cow::Borrowed(other);
                if left_df.should_rechunk() {
                    if _verbose {
                        eprintln!("{:?} join triggered a rechunk of the left DataFrame: {} columns are affected", args.how, left_df.width());
                    }

                    let mut tmp_left = left_df.clone();
                    tmp_left.as_single_chunk_par();
                    left = Cow::Owned(tmp_left);
                }
                if other.should_rechunk() {
                    if _verbose {
                        eprintln!("{:?} join triggered a rechunk of the right DataFrame: {} columns are affected", args.how, other.width());
                    }
                    let mut tmp_right = other.clone();
                    tmp_right.as_single_chunk_par();
                    right = Cow::Owned(tmp_right);
                }
                return left._join_impl(
                    &right,
                    selected_left,
                    selected_right,
                    args,
                    false,
                    _verbose,
                );
            }
        }

        if let Some((l, r)) = selected_left
            .iter()
            .zip(&selected_right)
            .find(|(l, r)| l.dtype() != r.dtype())
        {
            polars_bail!(
                ComputeError:
                    format!(
                        "datatypes of join keys don't match - `{}`: {} on left does not match `{}`: {} on right",
                        l.name(), l.dtype(), r.name(), r.dtype()
                    )
            );
        };

        #[cfg(feature = "dtype-categorical")]
        for (l, r) in selected_left.iter_mut().zip(selected_right.iter_mut()) {
            match _check_categorical_src(l.dtype(), r.dtype()) {
                Ok(_) => {},
                Err(_) => {
                    let (ca_left, ca_right) =
                        make_categoricals_compatible(l.categorical()?, r.categorical()?)?;
                    *l = ca_left.into_series().with_name(l.name());
                    *r = ca_right.into_series().with_name(r.name());
                },
            }
        }

        // Single keys.
        if selected_left.len() == 1 {
            let s_left = &selected_left[0];
            let s_right = &selected_right[0];
            let drop_names: Option<&[&str]> = if should_coalesce { None } else { Some(&[]) };
            return match args.how {
                JoinType::Inner => left_df
                    ._inner_join_from_series(other, s_left, s_right, args, _verbose, drop_names),
                JoinType::Left => dispatch_left_right::left_join_from_series(
                    self.to_df().clone(),
                    other,
                    s_left,
                    s_right,
                    args,
                    _verbose,
                    drop_names,
                ),
                JoinType::Right => dispatch_left_right::right_join_from_series(
                    self.to_df(),
                    other.clone(),
                    s_left,
                    s_right,
                    args,
                    _verbose,
                    drop_names,
                ),
                JoinType::Full => left_df._full_join_from_series(other, s_left, s_right, args),
                #[cfg(feature = "semi_anti_join")]
                JoinType::Anti => left_df._semi_anti_join_from_series(
                    s_left,
                    s_right,
                    args.slice,
                    true,
                    args.join_nulls,
                ),
                #[cfg(feature = "semi_anti_join")]
                JoinType::Semi => left_df._semi_anti_join_from_series(
                    s_left,
                    s_right,
                    args.slice,
                    false,
                    args.join_nulls,
                ),
                #[cfg(feature = "asof_join")]
                JoinType::AsOf(options) => match (options.left_by, options.right_by) {
                    (Some(left_by), Some(right_by)) => left_df._join_asof_by(
                        other,
                        s_left,
                        s_right,
                        left_by,
                        right_by,
                        options.strategy,
                        options.tolerance,
                        args.suffix.as_deref(),
                        args.slice,
                        should_coalesce,
                    ),
                    (None, None) => left_df._join_asof(
                        other,
                        s_left,
                        s_right,
                        options.strategy,
                        options.tolerance,
                        args.suffix,
                        args.slice,
                        should_coalesce,
                    ),
                    _ => {
                        panic!("expected by arguments on both sides")
                    },
                },
                JoinType::Cross => {
                    unreachable!()
                },
            };
        }

        let lhs_keys = prepare_keys_multiple(&selected_left, args.join_nulls)?.into_series();
        let rhs_keys = prepare_keys_multiple(&selected_right, args.join_nulls)?.into_series();

        let drop_names = if should_coalesce {
            Some(selected_right.iter().map(|s| s.name()).collect::<Vec<_>>())
        } else {
            Some(vec![])
        };

        // Multiple keys.
        match args.how {
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(_) => polars_bail!(
                ComputeError: "asof join not supported for join on multiple keys"
            ),
            JoinType::Cross => {
                unreachable!()
            },
            JoinType::Full => {
                let names_left = selected_left.iter().map(|s| s.name()).collect::<Vec<_>>();
                args.coalesce = JoinCoalesce::KeepColumns;
                let suffix = args.suffix.clone();
                let out = left_df._full_join_from_series(other, &lhs_keys, &rhs_keys, args);

                if should_coalesce {
                    Ok(_coalesce_full_join(
                        out?,
                        &names_left,
                        drop_names.as_ref().unwrap(),
                        suffix.as_deref(),
                        left_df,
                    ))
                } else {
                    out
                }
            },
            JoinType::Inner => left_df._inner_join_from_series(
                other,
                &lhs_keys,
                &rhs_keys,
                args,
                _verbose,
                drop_names.as_deref(),
            ),
            JoinType::Left => dispatch_left_right::left_join_from_series(
                left_df.clone(),
                other,
                &lhs_keys,
                &rhs_keys,
                args,
                _verbose,
                drop_names.as_deref(),
            ),
            JoinType::Right => dispatch_left_right::right_join_from_series(
                left_df,
                other.clone(),
                &lhs_keys,
                &rhs_keys,
                args,
                _verbose,
                drop_names.as_deref(),
            ),
            #[cfg(feature = "semi_anti_join")]
            JoinType::Anti | JoinType::Semi => self._join_impl(
                other,
                vec![lhs_keys],
                vec![rhs_keys],
                args,
                _check_rechunk,
                _verbose,
            ),
        }
    }

    /// Perform an inner join on two DataFrames.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// # use polars_ops::prelude::*;
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
        self.join(other, left_on, right_on, JoinArgs::new(JoinType::Inner))
    }

    /// Perform a left outer join on two DataFrames
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// # use polars_ops::prelude::*;
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
        self.join(other, left_on, right_on, JoinArgs::new(JoinType::Left))
    }

    /// Perform a full outer join on two DataFrames
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// # use polars_ops::prelude::*;
    /// fn join_dfs(left: &DataFrame, right: &DataFrame) -> PolarsResult<DataFrame> {
    ///     left.full_join(right, ["join_column_left"], ["join_column_right"])
    /// }
    /// ```
    fn full_join<I, S>(&self, other: &DataFrame, left_on: I, right_on: I) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.join(other, left_on, right_on, JoinArgs::new(JoinType::Full))
    }
}

trait DataFrameJoinOpsPrivate: IntoDf {
    fn _inner_join_from_series(
        &self,
        other: &DataFrame,
        s_left: &Series,
        s_right: &Series,
        args: JoinArgs,
        verbose: bool,
        drop_names: Option<&[&str]>,
    ) -> PolarsResult<DataFrame> {
        let left_df = self.to_df();
        #[cfg(feature = "dtype-categorical")]
        _check_categorical_src(s_left.dtype(), s_right.dtype())?;
        let ((join_tuples_left, join_tuples_right), sorted) =
            _sort_or_hash_inner(s_left, s_right, verbose, args.validation, args.join_nulls)?;

        let mut join_tuples_left = &*join_tuples_left;
        let mut join_tuples_right = &*join_tuples_right;

        if let Some((offset, len)) = args.slice {
            join_tuples_left = slice_slice(join_tuples_left, offset, len);
            join_tuples_right = slice_slice(join_tuples_right, offset, len);
        }

        let (df_left, df_right) = POOL.join(
            // SAFETY: join indices are known to be in bounds
            || unsafe { left_df._create_left_df_from_slice(join_tuples_left, false, sorted) },
            || unsafe {
                if let Some(drop_names) = drop_names {
                    other.drop_many(drop_names)
                } else {
                    other.drop(s_right.name()).unwrap()
                }
                ._take_unchecked_slice(join_tuples_right, true)
            },
        );
        _finish_join(df_left, df_right, args.suffix.as_deref())
    }
}

impl DataFrameJoinOps for DataFrame {}
impl DataFrameJoinOpsPrivate for DataFrame {}

fn prepare_keys_multiple(s: &[Series], join_nulls: bool) -> PolarsResult<BinaryOffsetChunked> {
    let keys = s
        .iter()
        .map(|s| {
            let phys = s.to_physical_repr();
            match phys.dtype() {
                DataType::Float32 => phys.f32().unwrap().to_canonical().into_series(),
                DataType::Float64 => phys.f64().unwrap().to_canonical().into_series(),
                _ => phys.into_owned(),
            }
        })
        .collect::<Vec<_>>();

    if join_nulls {
        encode_rows_vertical_par_unordered(&keys)
    } else {
        encode_rows_vertical_par_unordered_broadcast_nulls(&keys)
    }
}
pub fn private_left_join_multiple_keys(
    a: &DataFrame,
    b: &DataFrame,
    join_nulls: bool,
) -> PolarsResult<LeftJoinIds> {
    let a = prepare_keys_multiple(a.get_columns(), join_nulls)?.into_series();
    let b = prepare_keys_multiple(b.get_columns(), join_nulls)?.into_series();
    sort_or_hash_left(&a, &b, false, JoinValidation::ManyToMany, join_nulls)
}
