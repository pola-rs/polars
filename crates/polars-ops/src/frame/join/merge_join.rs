use std::borrow::Cow;
use std::cmp::Ordering;
use std::iter::repeat_n;

use arrow::array::Array;
use arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::TotalOrd;
use polars_utils::{IdxSize, format_pl_smallstr};

use crate::frame::{JoinArgs, JoinType};
use crate::series::coalesce_columns;

#[allow(clippy::too_many_arguments)]
pub fn match_keys(
    build_keys: &Series,
    probe_keys: &Series,
    gather_build: &mut Vec<IdxSize>,
    gather_probe: &mut Vec<IdxSize>,
    gather_probe_unmatched: Option<&mut Vec<IdxSize>>,
    build_emit_unmatched: bool,
    descending: bool,
    nulls_equal: bool,
    limit_results: usize,
    build_row_offset: &mut usize,
    probe_row_offset: &mut usize,
    probe_last_matched: &mut usize,
) {
    macro_rules! dispatch {
        ($build_keys_ca:expr) => {
            match_keys_impl(
                $build_keys_ca,
                probe_keys.as_ref().as_ref(),
                gather_build,
                gather_probe,
                gather_probe_unmatched,
                build_emit_unmatched,
                descending,
                nulls_equal,
                limit_results,
                build_row_offset,
                probe_row_offset,
                probe_last_matched,
            )
        };
    }

    assert_eq!(build_keys.dtype(), probe_keys.dtype());
    match build_keys.dtype() {
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                type PhysCa = ChunkedArray<$T>;
                let build_keys_ca: &PhysCa  = build_keys.as_ref().as_ref();
                dispatch!(build_keys_ca)
            })
        },
        DataType::Boolean => dispatch!(build_keys.bool().unwrap()),
        DataType::String => dispatch!(build_keys.str().unwrap()),
        DataType::Binary => dispatch!(build_keys.binary().unwrap()),
        DataType::BinaryOffset => dispatch!(build_keys.binary_offset().unwrap()),
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(cats, _) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
            let build_keys_ca: &PhysCa = build_keys.as_ref().as_ref();
            dispatch!(build_keys_ca)
        }),
        DataType::Null => match_null_keys_impl(
            build_keys.len(),
            probe_keys.len(),
            gather_build,
            gather_probe,
            gather_probe_unmatched,
            build_emit_unmatched,
            descending,
            nulls_equal,
            limit_results,
            build_row_offset,
            probe_row_offset,
            probe_last_matched,
        ),
        dt => unimplemented!("merge-join kernel not implemented for {:?}", dt),
    }
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn match_keys_impl<'a, T: PolarsDataType>(
    build_keys: &'a ChunkedArray<T>,
    probe_keys: &'a ChunkedArray<T>,
    gather_build: &mut Vec<IdxSize>,
    gather_probe: &mut Vec<IdxSize>,
    mut gather_probe_unmatched: Option<&mut Vec<IdxSize>>,
    build_emit_unmatched: bool,
    descending: bool,
    nulls_equal: bool,
    limit_results: usize,
    build_row_offset: &mut usize,
    probe_row_offset: &mut usize,
    probe_first_unmatched: &mut usize,
) where
    T::Physical<'a>: TotalOrd,
{
    assert!(gather_build.is_empty());
    assert!(gather_probe.is_empty());

    let build_key = build_keys.downcast_as_array();
    let probe_key = probe_keys.downcast_as_array();

    while *build_row_offset < build_key.len() {
        if gather_build.len() >= limit_results {
            return;
        }

        let build_keyval = unsafe { build_key.get_unchecked(*build_row_offset) };
        let build_keyval = build_keyval.as_ref();
        let mut build_keyval_matched = false;

        if nulls_equal || build_keyval.is_some() {
            for probe_idx in *probe_row_offset..probe_key.len() {
                let probe_keyval = unsafe { probe_key.get_unchecked(probe_idx) };
                let probe_keyval = probe_keyval.as_ref();

                let mut ord: Ordering = match (&build_keyval, &probe_keyval) {
                    (None, None) if nulls_equal => Ordering::Equal,
                    (Some(l), Some(r)) => TotalOrd::tot_cmp(*l, *r),
                    _ => continue,
                };
                if descending {
                    ord = ord.reverse();
                }

                match ord {
                    Ordering::Equal => {
                        if let Some(probe_unmatched) = gather_probe_unmatched.as_mut() {
                            // All probe keys up to and *excluding* this matched key are unmatched
                            probe_unmatched
                                .extend(*probe_first_unmatched as IdxSize..probe_idx as IdxSize);
                            *probe_first_unmatched = (*probe_first_unmatched).max(probe_idx + 1);
                        }
                        gather_build.push(*build_row_offset as IdxSize);
                        gather_probe.push(probe_idx as IdxSize);
                        build_keyval_matched = true;
                    },
                    Ordering::Greater => {
                        if let Some(probe_unmatched) = gather_probe_unmatched.as_mut() {
                            // All probe keys up to and *including* this matched key are unmatched
                            probe_unmatched
                                .extend(*probe_first_unmatched as IdxSize..=probe_idx as IdxSize);
                            *probe_first_unmatched = (*probe_first_unmatched).max(probe_idx + 1);
                        }
                        *probe_row_offset = probe_idx + 1;
                    },
                    Ordering::Less => {
                        break;
                    },
                }
            }
        }
        if build_emit_unmatched && !build_keyval_matched {
            gather_build.push(*build_row_offset as IdxSize);
            gather_probe.push(IdxSize::MAX);
        }
        *build_row_offset += 1;
    }
    if let Some(probe_unmatched) = gather_probe_unmatched {
        probe_unmatched.extend(*probe_first_unmatched as IdxSize..probe_key.len() as IdxSize);
        *probe_first_unmatched = probe_key.len();
    }
    *probe_row_offset = probe_key.len();
}

#[allow(clippy::mut_range_bound, clippy::too_many_arguments)]
fn match_null_keys_impl(
    build_n: usize,
    probe_n: usize,
    gather_build: &mut Vec<IdxSize>,
    gather_probe: &mut Vec<IdxSize>,
    gather_probe_unmatched: Option<&mut Vec<IdxSize>>,
    build_emit_unmatched: bool,
    _descending: bool,
    nulls_equal: bool,
    limit_results: usize,
    build_row_offset: &mut usize,
    probe_row_offset: &mut usize,
    probe_last_matched: &mut usize,
) {
    assert!(gather_build.is_empty());
    assert!(gather_probe.is_empty());

    if nulls_equal {
        // All keys will match all other keys, so just emit the Cartesian product
        while *build_row_offset < build_n {
            if gather_build.len() >= limit_results {
                return;
            }
            for probe_idx in *probe_row_offset..probe_n {
                gather_build.push(*build_row_offset as IdxSize);
                gather_probe.push(probe_idx as IdxSize);
            }
            *build_row_offset += 1;
        }
    } else {
        // No keys can ever match, so just emit all build keys into gather_build
        // and all probe keys into gather_probe_unmatched.
        if build_emit_unmatched {
            gather_build.extend(0..build_n as IdxSize);
            gather_probe.extend(repeat_n(IdxSize::MAX, build_n));
        }
        if let Some(probe_unmatched) = gather_probe_unmatched {
            probe_unmatched.extend(*probe_last_matched as IdxSize..probe_n as IdxSize);
            *probe_last_matched = probe_n;
        }
    }
    *build_row_offset = build_n;
    *probe_row_offset = probe_n;
}

#[allow(clippy::too_many_arguments)]
pub fn gather_and_postprocess(
    build: DataFrame,
    probe: DataFrame,
    gather_build: Option<&[IdxSize]>,
    gather_probe: Option<&[IdxSize]>,
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

    let (mut left, mut right);
    let (gather_left, gather_right);
    if left_is_build {
        (left, right) = (build, probe);
        (gather_left, gather_right) = (gather_build, gather_probe);
    } else {
        (left, right) = (probe, build);
        (gather_left, gather_right) = (gather_probe, gather_build);
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
    let mut left = match gather_left {
        Some(gather_left) if right_emit_unmatched => {
            left_build.opt_gather_extend(&left, gather_left, ShareStrategy::Never);
            left_build.freeze_reset()
        },
        Some(gather_left) => unsafe {
            left_build.gather_extend(&left, gather_left, ShareStrategy::Never);
            left_build.freeze_reset()
        },
        None => DataFrame::full_null(left.schema(), gather_right.unwrap().len()),
    };
    let mut right = match gather_right {
        Some(gather_right) if left_emit_unmatched => {
            right_build.opt_gather_extend(&right, gather_right, ShareStrategy::Never);
            right_build.freeze_reset()
        },
        Some(gather_right) => unsafe {
            right_build.gather_extend(&right, gather_right, ShareStrategy::Never);
            right_build.freeze_reset()
        },
        None => DataFrame::full_null(right.schema(), gather_left.unwrap().len()),
    };

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
