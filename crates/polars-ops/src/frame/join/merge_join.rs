use std::borrow::Cow;
use std::cmp::Ordering;

use arrow::array::Array;
use arrow::array::builder::ShareStrategy;
use arrow::bitmap::MutableBitmap;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::TotalOrd;
use polars_utils::{IdxSize, format_pl_smallstr};

use crate::frame::{JoinArgs, JoinType, MaintainOrderJoin};
use crate::series::coalesce_columns;

#[derive(Debug)]
pub struct MergeJoinSideParams {
    pub input_schema: SchemaRef,
    pub on: Vec<PlSmallStr>,
    pub key_col: PlSmallStr,
    pub emit_unmatched: bool,
}

#[derive(Debug)]
pub struct MergeJoinParams {
    pub left: MergeJoinSideParams,
    pub right: MergeJoinSideParams,
    pub output_schema: SchemaRef,
    pub key_descending: bool,
    pub key_nulls_last: bool,
    pub use_row_encoding: bool,
    pub args: JoinArgs,
}

impl MergeJoinParams {
    pub fn left_is_build(&self) -> bool {
        match self.args.maintain_order {
            MaintainOrderJoin::Right | MaintainOrderJoin::RightLeft => false,
            MaintainOrderJoin::Left | MaintainOrderJoin::LeftRight => true,
            MaintainOrderJoin::None if self.args.how == JoinType::Right => false,
            _ => true,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compute_join_dispatch(
    lk: &Series,
    rk: &Series,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    skip_build_rows: usize,
    limit_results: usize,
    left_sp: &MergeJoinSideParams,
    right_sp: &MergeJoinSideParams,
    params: &MergeJoinParams,
) -> (bool, usize) {
    macro_rules! dispatch {
        ($left_key_ca:expr) => {
            compute_join_impl(
                $left_key_ca,
                rk.as_ref().as_ref(),
                gather_left,
                gather_right,
                matched_right,
                skip_build_rows,
                limit_results,
                left_sp,
                right_sp,
                params,
            )
        };
    }

    assert_eq!(lk.dtype(), rk.dtype());
    match lk.dtype() {
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                type PhysCa = ChunkedArray<$T>;
                let lk_ca: &PhysCa  = lk.as_ref().as_ref();
                dispatch!(lk_ca)
            })
        },
        DataType::Boolean => dispatch!(lk.bool().unwrap()),
        DataType::String => dispatch!(lk.str().unwrap()),
        DataType::Binary => dispatch!(lk.binary().unwrap()),
        DataType::BinaryOffset => dispatch!(lk.binary_offset().unwrap()),
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(cats, _) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
            let lk_ca: &PhysCa = lk.as_ref().as_ref();
            dispatch!(lk_ca)
        }),
        DataType::Null => compute_join_impl_nullkeys(
            lk.len(),
            rk.len(),
            gather_left,
            gather_right,
            matched_right,
            skip_build_rows,
            limit_results,
            left_sp,
            right_sp,
            params,
        ),
        dt => unimplemented!("merge-join kernel not implemented for {:?}", dt),
    }
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn compute_join_impl<'a, T: PolarsDataType>(
    left_key: &'a ChunkedArray<T>,
    right_key: &'a ChunkedArray<T>,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    mut skip_build_rows: usize,
    limit_results: usize,
    build_sp: &MergeJoinSideParams,
    probe_sp: &MergeJoinSideParams,
    params: &MergeJoinParams,
) -> (bool, usize)
where
    T::Physical<'a>: TotalOrd,
{
    debug_assert!(gather_left.is_empty());
    debug_assert!(gather_right.is_empty());
    if probe_sp.emit_unmatched {
        debug_assert!(matched_right.len() == right_key.len());
    }

    let descending = params.key_descending;
    let left_key = left_key.downcast_as_array();
    let right_key = right_key.downcast_as_array();

    let mut iterator = left_key.iter().enumerate().skip(skip_build_rows).peekable();
    if iterator.peek().is_none() {
        return (true, skip_build_rows);
    }
    let mut skip_ahead_right = 0;
    for (idxl, left_keyval) in iterator {
        if gather_left.len() >= limit_results {
            return (false, skip_build_rows);
        }
        let left_keyval = left_keyval.as_ref();
        let mut matched = false;
        if params.args.nulls_equal || left_keyval.is_some() {
            for idxr in skip_ahead_right..right_key.len() {
                let right_keyval = unsafe { right_key.get_unchecked(idxr) };
                let right_keyval = right_keyval.as_ref();
                let mut ord: Option<Ordering> = match (&left_keyval, &right_keyval) {
                    (None, None) if params.args.nulls_equal => Some(Ordering::Equal),
                    (Some(l), Some(r)) => Some(TotalOrd::tot_cmp(*l, *r)),
                    _ => None,
                };
                if descending {
                    ord = ord.map(Ordering::reverse);
                }
                if ord == Some(Ordering::Equal) {
                    matched = true;
                    if probe_sp.emit_unmatched {
                        matched_right.set(idxr, true);
                    }
                    gather_left.push(idxl as IdxSize);
                    gather_right.push(idxr as IdxSize);
                } else if ord == Some(Ordering::Greater) {
                    skip_ahead_right = idxr;
                } else if ord == Some(Ordering::Less) {
                    break;
                }
            }
        }
        if build_sp.emit_unmatched && !matched {
            gather_left.push(idxl as IdxSize);
            gather_right.push(IdxSize::MAX);
        }
        skip_build_rows += 1;
    }
    (true, skip_build_rows)
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn compute_join_impl_nullkeys(
    left_n: usize,
    right_n: usize,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    mut skip_build_rows: usize,
    limit_results: usize,
    left_sp: &MergeJoinSideParams,
    right_sp: &MergeJoinSideParams,
    params: &MergeJoinParams,
) -> (bool, usize) {
    debug_assert!(gather_left.is_empty());
    debug_assert!(gather_right.is_empty());
    if right_sp.emit_unmatched {
        debug_assert!(matched_right.len() == right_n);
    }
    if !params.args.nulls_equal {
        return (true, skip_build_rows);
    }

    for idxl in skip_build_rows..left_n {
        gather_left.push(idxl as IdxSize);
        for idxr in 0..right_n {
            gather_right.push(idxr as IdxSize);
            if right_sp.emit_unmatched {
                matched_right.set(idxr, true);
            }
        }
        if left_sp.emit_unmatched && right_n == 0 {
            gather_right.push(IdxSize::MAX);
        }
        skip_build_rows += 1;
        if gather_left.len() >= limit_results {
            return (false, skip_build_rows);
        }
    }
    (true, skip_build_rows)
}

pub fn gather_and_postprocess(
    build: DataFrame,
    probe: DataFrame,
    gather_build: &[IdxSize],
    gather_probe: &[IdxSize],
    df_builders: &mut Option<(DataFrameBuilder, DataFrameBuilder)>,
    params: &MergeJoinParams,
) -> PolarsResult<DataFrame> {
    let should_coalesce = params.args.should_coalesce();

    let mut left;
    let gather_left;
    let mut right;
    let gather_right;
    if params.left_is_build() {
        left = build;
        gather_left = gather_build;
        right = probe;
        gather_right = gather_probe;
    } else {
        right = build;
        gather_right = gather_build;
        left = probe;
        gather_left = gather_probe;
    }

    // Remove non-payload columns
    for col in left
        .columns()
        .iter()
        .map(Column::name)
        .cloned()
        .collect_vec()
    {
        if params.left.on.contains(&col) && should_coalesce {
            continue;
        }
        if !params.output_schema.contains(&col) {
            left.drop_in_place(&col).unwrap();
        }
    }
    for col in right
        .columns()
        .iter()
        .map(Column::name)
        .cloned()
        .collect_vec()
    {
        if params.right.on.contains(&col) && should_coalesce {
            continue;
        }
        let renamed_col = match right.schema().contains(&col) {
            true => Cow::Owned(format_pl_smallstr!("{}{}", col, params.args.suffix())),
            false => Cow::Borrowed(&col),
        };
        if !params.output_schema.contains(&renamed_col) {
            right.drop_in_place(&col).unwrap();
        }
    }

    if df_builders.is_none() {
        *df_builders = Some((
            DataFrameBuilder::new(left.schema().clone()),
            DataFrameBuilder::new(right.schema().clone()),
        ));
    }

    let (left_build, right_build) = df_builders.as_mut().unwrap();
    if params.right.emit_unmatched {
        left_build.opt_gather_extend(&left, gather_left, ShareStrategy::Never);
    } else {
        unsafe { left_build.gather_extend(&left, gather_left, ShareStrategy::Never) };
    }
    if params.left.emit_unmatched {
        right_build.opt_gather_extend(&right, gather_right, ShareStrategy::Never);
    } else {
        unsafe { right_build.gather_extend(&right, gather_right, ShareStrategy::Never) };
    }

    let mut left = left_build.freeze_reset();
    let mut right = right_build.freeze_reset();

    // Coalsesce the key columns
    if params.args.how == JoinType::Left && should_coalesce {
        for c in &params.left.on {
            if right.schema().contains(c) {
                right.drop_in_place(c.as_str())?;
            }
        }
    } else if params.args.how == JoinType::Right && should_coalesce {
        for c in &params.right.on {
            if left.schema().contains(c) {
                left.drop_in_place(c.as_str())?;
            }
        }
    }

    // Rename any right columns to "{}_right"
    let left_cols: PlHashSet<_> = left.columns().iter().map(Column::name).cloned().collect();
    let right_cols_vec = right.get_column_names_owned();
    let renames = right_cols_vec
        .iter()
        .filter(|c| left_cols.contains(*c))
        .map(|c| {
            let renamed = format_pl_smallstr!("{}{}", c, params.args.suffix());
            (c.as_str(), renamed)
        });
    right.rename_many(renames).unwrap();

    left.hstack_mut(right.columns())?;

    if params.args.how == JoinType::Full && should_coalesce {
        // Coalesce key columns
        for (left_keycol, right_keycol) in
            Iterator::zip(params.left.on.iter(), params.right.on.iter())
        {
            let right_keycol = format_pl_smallstr!("{}{}", right_keycol, params.args.suffix());
            let left_col = left.column(left_keycol).unwrap();
            let right_col = left.column(&right_keycol).unwrap();
            let coalesced = coalesce_columns(&[left_col.clone(), right_col.clone()]).unwrap();
            left.replace(left_keycol, coalesced)
                .unwrap()
                .drop_in_place(&right_keycol)
                .unwrap();
        }
    }

    if should_coalesce {
        for col in &params.left.on {
            if left.schema().contains(&col) && !params.output_schema.contains(&col) {
                left.drop_in_place(&col).unwrap();
            }
        }
        for col in &params.right.on {
            let renamed = format_pl_smallstr!("{}{}", col, params.args.suffix());
            if left.schema().contains(&renamed) && !params.output_schema.contains(&renamed) {
                left.drop_in_place(&renamed).unwrap();
            }
        }
    }

    debug_assert_eq!(left.schema(), &params.output_schema);
    Ok(left)
}
