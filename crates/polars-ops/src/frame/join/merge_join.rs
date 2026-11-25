use std::borrow::Cow;
use std::cmp::Ordering;
use std::iter::repeat_n;
use std::sync::Arc;

use bytemuck::TransparentWrapper;
use polars_arrow::array::Array;
use polars_arrow::array::builder::ShareStrategy;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_defs::join::{JoinArgs, JoinType};
use polars_utils::sort::reorder_cmp;
use polars_utils::total_ord::{TotalOrd, TotalOrdWrap};
use polars_utils::{IdxSize, format_pl_smallstr};

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
    nulls_last: bool,
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
                nulls_last,
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
    nulls_last: bool,
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
        let mut build_keyval_matched = false;

        if nulls_equal || build_keyval.is_some() {
            for probe_idx in *probe_row_offset..probe_key.len() {
                let probe_keyval = unsafe { probe_key.get_unchecked(probe_idx) };

                // Row-encoded keys are null wherever any key column is null, so a null probe
                // key has no place in the order.
                if !nulls_equal && probe_keyval.is_none() {
                    continue;
                }
                let ord = reorder_cmp(
                    TotalOrdWrap::wrap_ref(&build_keyval),
                    TotalOrdWrap::wrap_ref(&probe_keyval),
                    descending,
                    nulls_last,
                );

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

#[derive(Debug)]
struct MergeJoinSideProjection {
    indices: Vec<usize>,
    schema: SchemaRef,
    gather_nulls: bool,
}

impl MergeJoinSideProjection {
    fn new(input: &Schema, indices: Vec<usize>, gather_nulls: bool) -> Self {
        let schema = indices
            .iter()
            .map(|&idx| {
                let (name, dtype) = input.get_at_index(idx).unwrap();
                (name.clone(), dtype.clone())
            })
            .collect();
        Self {
            indices,
            schema: Arc::new(schema),
            gather_nulls,
        }
    }

    fn select<'a>(&self, df: &'a DataFrame) -> Cow<'a, DataFrame> {
        if self.indices.len() == df.width() {
            return Cow::Borrowed(df);
        }
        let columns = self
            .indices
            .iter()
            .map(|&i| df.columns()[i].clone())
            .collect();
        // SAFETY: The projection preserves column lengths, names, and types.
        Cow::Owned(unsafe { DataFrame::new_unchecked(df.height(), columns) })
    }

    fn gather(
        &self,
        df: &DataFrame,
        indices: Option<&[IdxSize]>,
        height: usize,
        builder: &mut DataFrameBuilder,
    ) -> DataFrame {
        match indices {
            Some(indices) if self.gather_nulls => {
                builder.opt_gather_extend(df, indices, ShareStrategy::Never);
                builder.freeze_reset()
            },
            Some(indices) => {
                // SAFETY: The matcher emits only in-bounds indices on this side.
                unsafe { builder.gather_extend(df, indices, ShareStrategy::Never) };
                builder.freeze_reset()
            },
            None => DataFrame::full_null(&self.schema, height),
        }
    }
}

#[derive(Debug)]
pub struct MergeJoinProjection {
    left: MergeJoinSideProjection,
    right: MergeJoinSideProjection,
    coalesced_keys: Vec<(usize, usize)>,
    // Right gather columns with no output name contribute only to a coalesced left key.
    right_output_names: Vec<Option<PlSmallStr>>,
}

impl MergeJoinProjection {
    pub fn new(
        left: &Schema,
        right: &Schema,
        output: &Schema,
        pre_drop: &Schema,
        args: &JoinArgs,
        left_on: &[PlSmallStr],
        right_on: &[PlSmallStr],
    ) -> Self {
        let should_coalesce = args.should_coalesce();
        let left_keys_coalesced_away = should_coalesce && matches!(args.how, JoinType::Right);
        let right_keys_coalesced_away = should_coalesce && !left_keys_coalesced_away;
        let coalesce_full = should_coalesce && matches!(args.how, JoinType::Full);
        let left_keys: PlHashSet<_> = left_on.iter().collect();
        let right_keys: PlHashSet<_> = right_on.iter().collect();
        let coalesced_keys: Vec<_> = if coalesce_full {
            left_on
                .iter()
                .zip(right_on)
                .filter(|(l, _)| output.contains(l))
                .collect()
        } else {
            Vec::new()
        };
        let needed_right_keys: PlHashSet<_> = coalesced_keys.iter().map(|&(_, r)| r).collect();

        let left_indices = left
            .iter_names()
            .enumerate()
            .filter(|(_, col)| {
                output.contains(col) && !(left_keys_coalesced_away && left_keys.contains(col))
            })
            .map(|(idx, _)| idx)
            .collect();

        let mut right_indices = Vec::new();
        let mut right_output_names = Vec::new();
        for (idx, col) in right.iter_names().enumerate() {
            if right_keys_coalesced_away && right_keys.contains(col) {
                if needed_right_keys.contains(col) {
                    right_indices.push(idx);
                    right_output_names.push(None);
                }
                continue;
            }

            // Resolve suffixes before pruning the left namesake from the gather.
            let collides = pre_drop.contains(col)
                && left.contains(col)
                && !(left_keys_coalesced_away && left_keys.contains(col));
            let name = if collides {
                format_pl_smallstr!("{col}{}", args.suffix())
            } else {
                col.clone()
            };
            if output.contains(&name) {
                right_indices.push(idx);
                right_output_names.push(Some(name));
            }
        }

        let left = MergeJoinSideProjection::new(
            left,
            left_indices,
            matches!(args.how, JoinType::Right | JoinType::Full),
        );
        let right = MergeJoinSideProjection::new(
            right,
            right_indices,
            matches!(args.how, JoinType::Left | JoinType::Full),
        );
        let coalesced_keys = coalesced_keys
            .into_iter()
            .map(|(l, r)| {
                (
                    left.schema.index_of(l).unwrap(),
                    right.schema.index_of(r).unwrap(),
                )
            })
            .collect();

        Self {
            left,
            right,
            coalesced_keys,
            right_output_names,
        }
    }

    pub fn select<'a>(
        &self,
        build: &'a DataFrame,
        probe: &'a DataFrame,
        left_is_build: bool,
    ) -> (Cow<'a, DataFrame>, Cow<'a, DataFrame>) {
        if left_is_build {
            (self.left.select(build), self.right.select(probe))
        } else {
            (self.right.select(build), self.left.select(probe))
        }
    }

    pub fn gather_and_postprocess(
        &self,
        build: &DataFrame,
        probe: &DataFrame,
        gather_build: Option<&[IdxSize]>,
        gather_probe: Option<&[IdxSize]>,
        df_builders: &mut Option<(DataFrameBuilder, DataFrameBuilder)>,
        left_is_build: bool,
    ) -> PolarsResult<DataFrame> {
        let ((left, gather_left), (right, gather_right)) = if left_is_build {
            ((build, gather_build), (probe, gather_probe))
        } else {
            ((probe, gather_probe), (build, gather_build))
        };
        let height = gather_left.or(gather_right).unwrap().len();
        let (left_builder, right_builder) = df_builders.get_or_insert_with(|| {
            (
                DataFrameBuilder::new(self.left.schema.clone()),
                DataFrameBuilder::new(self.right.schema.clone()),
            )
        });
        let mut left = self
            .left
            .gather(left, gather_left, height, left_builder)
            .into_columns();
        let right = self
            .right
            .gather(right, gather_right, height, right_builder)
            .into_columns();

        // Keep right key columns available until every left key has consumed its
        // partners, including repeated right keys. Repeated left keys coalesce in
        // the order of the join condition.
        for &(l, r) in &self.coalesced_keys {
            left[l] = coalesce_columns(&[left[l].clone(), right[r].clone()])?;
        }
        left.extend(right.into_iter().zip(&self.right_output_names).filter_map(
            |(mut col, name)| {
                let name = name.as_ref()?;
                if col.name() != name {
                    col.rename(name.clone());
                }
                Some(col)
            },
        ));
        DataFrame::new(height, left)
    }
}

#[cfg(test)]
mod tests {
    use polars_core::df;
    use polars_defs::join::JoinCoalesce;

    use super::*;

    #[test]
    fn merge_join_projection_coalesces_repeated_keys() {
        let left = df!("a" => [] as [i32; 0], "b" => [] as [i32; 0]).unwrap();
        let right = df!("x" => [7, 10], "y" => [8, 11]).unwrap();
        let mut args = JoinArgs::new(JoinType::Full);
        args.coalesce = JoinCoalesce::CoalesceColumns;
        let projection = MergeJoinProjection::new(
            left.schema(),
            right.schema(),
            left.schema(),
            left.schema(),
            &args,
            &["a".into(), "a".into(), "b".into()],
            &["x".into(), "y".into(), "x".into()],
        );
        let (build, probe) = projection.select(&left, &right, true);
        let result = projection
            .gather_and_postprocess(&build, &probe, None, Some(&[0, 1]), &mut None, true)
            .unwrap();

        assert_eq!(result, df!("a" => [7, 10], "b" => [7, 10]).unwrap());
    }
}
