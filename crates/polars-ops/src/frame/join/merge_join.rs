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

#[allow(clippy::too_many_arguments)]
pub fn match_keys(
    left_keys: &Series,
    right_keys: &Series,
    gather_build: &mut Vec<IdxSize>,
    gather_probe: &mut Vec<IdxSize>,
    matched_probe: &mut MutableBitmap,
    probe_mark_matched: bool,
    build_emit_unmatched: bool,
    descending: bool,
    nulls_equal: bool,
    limit_results: usize,
    mut skip_build_rows: usize,
) -> (bool, usize) {
    macro_rules! dispatch {
        ($left_keys_ca:expr) => {
            match_keys_impl(
                $left_keys_ca,
                right_keys.as_ref().as_ref(),
                gather_build,
                gather_probe,
                matched_probe,
                probe_mark_matched,
                build_emit_unmatched,
                descending,
                nulls_equal,
                limit_results,
                skip_build_rows,
            )
        };
    }

    assert_eq!(left_keys.dtype(), right_keys.dtype());
    match left_keys.dtype() {
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                type PhysCa = ChunkedArray<$T>;
                let left_keys_ca: &PhysCa  = left_keys.as_ref().as_ref();
                dispatch!(left_keys_ca)
            })
        },
        DataType::Boolean => dispatch!(left_keys.bool().unwrap()),
        DataType::String => dispatch!(left_keys.str().unwrap()),
        DataType::Binary => dispatch!(left_keys.binary().unwrap()),
        DataType::BinaryOffset => dispatch!(left_keys.binary_offset().unwrap()),
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(cats, _) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
            let left_keys_ca: &PhysCa = left_keys.as_ref().as_ref();
            dispatch!(left_keys_ca)
        }),
        DataType::Null => match_null_keys_impl(
            left_keys.len(),
            right_keys.len(),
            gather_build,
            gather_probe,
            matched_probe,
            probe_mark_matched,
            build_emit_unmatched,
            descending,
            nulls_equal,
            limit_results,
            skip_build_rows,
        ),
        dt => unimplemented!("merge-join kernel not implemented for {:?}", dt),
    }
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn match_keys_impl<'a, T: PolarsDataType>(
    left_keys: &'a ChunkedArray<T>,
    right_keys: &'a ChunkedArray<T>,
    gather_left: &mut Vec<IdxSize>,
    gather_right: &mut Vec<IdxSize>,
    matched_right: &mut MutableBitmap,
    probe_mark_matched: bool,
    build_emit_unmatched: bool,
    descending: bool,
    nulls_equal: bool,
    limit_results: usize,
    mut skip_build_rows: usize,
) -> (bool, usize)
where
    T::Physical<'a>: TotalOrd,
{
    assert!(gather_left.is_empty());
    assert!(gather_right.is_empty());
    if probe_mark_matched {
        assert_eq!(matched_right.len(), right_keys.len());
    }

    let left_key = left_keys.downcast_as_array();
    let right_key = right_keys.downcast_as_array();

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
        if nulls_equal || left_keyval.is_some() {
            for idxr in skip_ahead_right..right_key.len() {
                let right_keyval = unsafe { right_key.get_unchecked(idxr) };
                let right_keyval = right_keyval.as_ref();
                let mut ord: Option<Ordering> = match (&left_keyval, &right_keyval) {
                    (None, None) if nulls_equal => Some(Ordering::Equal),
                    (Some(l), Some(r)) => Some(TotalOrd::tot_cmp(*l, *r)),
                    _ => None,
                };
                if descending {
                    ord = ord.map(Ordering::reverse);
                }
                if ord == Some(Ordering::Equal) {
                    matched = true;
                    if probe_mark_matched {
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
        if build_emit_unmatched && !matched {
            gather_left.push(idxl as IdxSize);
            gather_right.push(IdxSize::MAX);
        }
        skip_build_rows += 1;
    }
    (true, skip_build_rows)
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn match_null_keys_impl(
    left_n: usize,
    right_n: usize,
    gather_build: &mut Vec<IdxSize>,
    gather_probe: &mut Vec<IdxSize>,
    matched_probe: &mut MutableBitmap,
    probe_mark_matched: bool,
    build_emit_unmatched: bool,
    descending: bool,
    nulls_equal: bool,
    limit_results: usize,
    mut skip_build_rows: usize,
) -> (bool, usize) {
    assert!(gather_build.is_empty());
    assert!(gather_probe.is_empty());
    if probe_mark_matched {
        assert_eq!(matched_probe.len(), right_n);
    }
    if !nulls_equal {
        return (true, skip_build_rows);
    }

    for idxl in skip_build_rows..left_n {
        for idxr in 0..right_n {
            gather_build.push(idxl as IdxSize);
            gather_probe.push(idxr as IdxSize);
            if probe_mark_matched {
                matched_probe.set(idxr, true);
            }
        }
        if build_emit_unmatched && right_n == 0 {
            gather_probe.push(idxl as IdxSize);
            gather_probe.push(IdxSize::MAX);
        }
        skip_build_rows += 1;
        if gather_build.len() >= limit_results {
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
    args: &JoinArgs,
    left_on: &[PlSmallStr],
    right_on: &[PlSmallStr],
    left_is_build: bool,
    output_schema: &Schema,
) -> PolarsResult<DataFrame> {
    let should_coalesce = args.should_coalesce();
    let left_emit_unmatched = matches!(args.how, JoinType::Left | JoinType::Full);
    let right_emit_unmatched = matches!(args.how, JoinType::Right | JoinType::Full);

    let mut left;
    let gather_left;
    let mut right;
    let gather_right;
    if left_is_build {
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
        if left_on.contains(&col) && should_coalesce {
            continue;
        }
        if !output_schema.contains(&col) {
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
        if left_on.contains(&col) && should_coalesce {
            continue;
        }
        let renamed = match left.schema().contains(&col) {
            true => Cow::Owned(format_pl_smallstr!("{}{}", col, args.suffix())),
            false => Cow::Borrowed(&col),
        };
        if !output_schema.contains(&renamed) {
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
    if right_emit_unmatched {
        left_build.opt_gather_extend(&left, gather_left, ShareStrategy::Never);
    } else {
        unsafe { left_build.gather_extend(&left, gather_left, ShareStrategy::Never) };
    }
    if left_emit_unmatched {
        right_build.opt_gather_extend(&right, gather_right, ShareStrategy::Never);
    } else {
        unsafe { right_build.gather_extend(&right, gather_right, ShareStrategy::Never) };
    }

    let mut left = left_build.freeze_reset();
    let mut right = right_build.freeze_reset();

    // Coalsesce the key columns
    if args.how == JoinType::Left && should_coalesce {
        for c in left_on {
            if right.schema().contains(c) {
                right.drop_in_place(c.as_str())?;
            }
        }
    } else if args.how == JoinType::Right && should_coalesce {
        for c in right_on {
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
            let renamed = format_pl_smallstr!("{}{}", c, args.suffix());
            (c.as_str(), renamed)
        });
    right.rename_many(renames).unwrap();

    left.hstack_mut(right.columns())?;

    if args.how == JoinType::Full && should_coalesce {
        // Coalesce key columns
        for (left_keycol, right_keycol) in Iterator::zip(left_on.iter(), right_on.iter()) {
            let right_keycol = format_pl_smallstr!("{}{}", right_keycol, args.suffix());
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
        for col in left_on {
            if left.schema().contains(col) && !output_schema.contains(col) {
                left.drop_in_place(col).unwrap();
            }
        }
        for col in right_on {
            let renamed = match left.schema().contains(col) {
                true => Cow::Owned(format_pl_smallstr!("{}{}", col, args.suffix())),
                false => Cow::Borrowed(col),
            };
            if left.schema().contains(&renamed) && !output_schema.contains(&renamed) {
                left.drop_in_place(&renamed).unwrap();
            }
        }
    }

    debug_assert_eq!(**left.schema(), *output_schema);
    Ok(left)
}
