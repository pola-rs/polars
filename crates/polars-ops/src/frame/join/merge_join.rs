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

pub struct MergeJoin<'a> {
    build: DataFrame,
    probe: DataFrame,
    build_keys: Series,
    probe_keys: Series,
    params: &'a MergeJoinParams,
    build_sp: &'a MergeJoinSideParams,
    probe_sp: &'a MergeJoinSideParams,
    gather_build: &'a mut Vec<IdxSize>,
    gather_probe: &'a mut Vec<IdxSize>,
    matched_probeside: &'a mut MutableBitmap,
    df_builders: &'a mut Option<(DataFrameBuilder, DataFrameBuilder)>,
    match_keys_done: bool,
    skip_build_rows: usize,
}

impl<'a> MergeJoin<'a> {
    pub fn new(
        params: &'a MergeJoinParams,
        mut left: DataFrame,
        mut right: DataFrame,
        gather_left: &'a mut Vec<IdxSize>,
        gather_right: &'a mut Vec<IdxSize>,
        matched_probeside: &'a mut MutableBitmap,
        df_builders: &'a mut Option<(DataFrameBuilder, DataFrameBuilder)>,
    ) -> PolarsResult<Self> {
        left.rechunk_mut();
        right.rechunk_mut();

        let build;
        let build_sp;
        let gather_build;
        let probe;
        let probe_sp;
        let gather_probe;
        if params.left_is_build() {
            build = left;
            build_sp = &params.left;
            gather_build = gather_left;
            probe = right;
            probe_sp = &params.right;
            gather_probe = gather_right;
        } else {
            build = right;
            build_sp = &params.right;
            gather_build = gather_right;
            probe = left;
            probe_sp = &params.left;
            gather_probe = gather_left;
        }

        // TODO: [amber] Possible unnecessary clones here

        let mut build_keys = build
            .column(&build_sp.key_col)
            .unwrap()
            .as_materialized_series()
            .to_owned();
        let mut probe_keys = probe
            .column(&probe_sp.key_col)
            .unwrap()
            .as_materialized_series()
            .to_owned();

        #[cfg(feature = "dtype-categorical")]
        {
            // Categoricals are lexicographically ordered, not by their physical values.
            if matches!(build_keys.dtype(), DataType::Categorical(_, _)) {
                build_keys = build_keys.cast(&DataType::String)?;
            }
            if matches!(probe_keys.dtype(), DataType::Categorical(_, _)) {
                probe_keys = probe_keys.cast(&DataType::String)?;
            }
        }

        let build_keys = build_keys.to_physical_repr().into_owned();
        let probe_keys = probe_keys.to_physical_repr().into_owned();

        matched_probeside.clear();
        matched_probeside.resize(probe_keys.len(), false);

        Ok(Self {
            build,
            probe,
            build_keys,
            probe_keys,
            params,
            build_sp,
            probe_sp,
            gather_build,
            gather_probe,
            matched_probeside,
            df_builders,
            match_keys_done: false,
            skip_build_rows: 0,
        })
    }

    pub fn next_matched_chunk(&mut self, rough_limit: usize) -> PolarsResult<Option<DataFrame>> {
        if self.match_keys_done {
            return Ok(None);
        }
        self.gather_build.clear();
        self.gather_probe.clear();
        (self.match_keys_done, self.skip_build_rows) = match_keys(
            &self.build_keys,
            &self.probe_keys,
            self.gather_build,
            self.gather_probe,
            self.matched_probeside,
            self.skip_build_rows,
            rough_limit,
            self.build_sp,
            self.probe_sp,
            self.params,
        );
        let df = gather_and_postprocess(
            self.build.clone(),
            self.probe.clone(),
            self.gather_build,
            self.gather_probe,
            self.df_builders,
            self.params,
        )?;
        Ok(Some(df))
    }

    pub fn unmatched(self) -> PolarsResult<DataFrame> {
        assert!(self.match_keys_done);
        if !self.probe_sp.emit_unmatched {
            return Ok(DataFrame::empty_with_schema(&self.params.output_schema));
        }
        self.gather_build.clear();
        self.gather_probe.clear();
        for (idx, _) in self
            .matched_probeside
            .iter()
            .enumerate_idx()
            .filter(|(_, m)| !m)
        {
            self.gather_build.push(IdxSize::MAX);
            self.gather_probe.push(idx);
        }
        let df_unmatched = gather_and_postprocess(
            self.build,
            self.probe,
            self.gather_build,
            self.gather_probe,
            self.df_builders,
            self.params,
        )?;
        Ok(df_unmatched)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn match_keys(
    left_keys: &Series,
    right_keys: &Series,
    gather_build: &mut Vec<IdxSize>,
    gather_probe: &mut Vec<IdxSize>,
    matched_probeside: &mut MutableBitmap,
    skip_build_rows: usize,
    limit_results: usize,
    left_sp: &MergeJoinSideParams,
    right_sp: &MergeJoinSideParams,
    params: &MergeJoinParams,
) -> (bool, usize) {
    macro_rules! dispatch {
        ($left_keys_ca:expr) => {
            match_keys_impl(
                $left_keys_ca,
                right_keys.as_ref().as_ref(),
                gather_build,
                gather_probe,
                matched_probeside,
                skip_build_rows,
                limit_results,
                left_sp,
                right_sp,
                params,
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
            matched_probeside,
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
fn match_keys_impl<'a, T: PolarsDataType>(
    left_keys: &'a ChunkedArray<T>,
    right_keys: &'a ChunkedArray<T>,
    gather_build: &mut Vec<IdxSize>,
    gather_probe: &mut Vec<IdxSize>,
    matched_probe: &mut MutableBitmap,
    mut skip_build_rows: usize,
    limit_results: usize,
    build_sp: &MergeJoinSideParams,
    probe_sp: &MergeJoinSideParams,
    params: &MergeJoinParams,
) -> (bool, usize)
where
    T::Physical<'a>: TotalOrd,
{
    debug_assert!(gather_build.is_empty());
    debug_assert!(gather_probe.is_empty());
    if probe_sp.emit_unmatched {
        debug_assert!(matched_probe.len() == right_keys.len());
    }

    let descending = params.key_descending;
    let left_key = left_keys.downcast_as_array();
    let right_key = right_keys.downcast_as_array();

    let mut iterator = left_key.iter().enumerate().skip(skip_build_rows).peekable();
    if iterator.peek().is_none() {
        return (true, skip_build_rows);
    }
    let mut skip_ahead_right = 0;
    for (idxl, left_keyval) in iterator {
        if gather_build.len() >= limit_results {
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
                        matched_probe.set(idxr, true);
                    }
                    gather_build.push(idxl as IdxSize);
                    gather_probe.push(idxr as IdxSize);
                } else if ord == Some(Ordering::Greater) {
                    skip_ahead_right = idxr;
                } else if ord == Some(Ordering::Less) {
                    break;
                }
            }
        }
        if build_sp.emit_unmatched && !matched {
            gather_build.push(idxl as IdxSize);
            gather_probe.push(IdxSize::MAX);
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
    matched_right: &mut MutableBitmap,
    mut skip_build_rows: usize,
    limit_results: usize,
    left_sp: &MergeJoinSideParams,
    right_sp: &MergeJoinSideParams,
    params: &MergeJoinParams,
) -> (bool, usize) {
    debug_assert!(gather_build.is_empty());
    debug_assert!(gather_probe.is_empty());
    if right_sp.emit_unmatched {
        debug_assert!(matched_right.len() == right_n);
    }
    if !params.args.nulls_equal {
        return (true, skip_build_rows);
    }

    for idxl in skip_build_rows..left_n {
        gather_build.push(idxl as IdxSize);
        for idxr in 0..right_n {
            gather_probe.push(idxr as IdxSize);
            if right_sp.emit_unmatched {
                matched_right.set(idxr, true);
            }
        }
        if left_sp.emit_unmatched && right_n == 0 {
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
        if params.left.on.contains(&col) && should_coalesce {
            continue;
        }
        let renamed = match left.schema().contains(&col) {
            true => Cow::Owned(format_pl_smallstr!("{}{}", col, params.args.suffix())),
            false => Cow::Borrowed(&col),
        };
        if !params.output_schema.contains(&renamed) {
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
    if *left_build.schema() != **left.schema() {
        *left_build = DataFrameBuilder::new(left.schema().clone());
    }
    if *right_build.schema() != **right.schema() {
        *right_build = DataFrameBuilder::new(right.schema().clone());
    }

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
            if left.schema().contains(col) && !params.output_schema.contains(col) {
                left.drop_in_place(col).unwrap();
            }
        }
        for col in &params.right.on {
            let renamed = match left.schema().contains(col) {
                true => Cow::Owned(format_pl_smallstr!("{}{}", col, params.args.suffix())),
                false => Cow::Borrowed(col),
            };
            if left.schema().contains(&renamed) && !params.output_schema.contains(&renamed) {
                left.drop_in_place(&renamed).unwrap();
            }
        }
    }

    debug_assert_eq!(left.schema(), &params.output_schema);
    Ok(left)
}
