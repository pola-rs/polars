//! Reductions split into a stage that outputs per-group states and one that merges them.

#[cfg(feature = "serde")]
mod stages;

#[cfg(all(feature = "serde", feature = "cov"))]
pub(super) use stages::Encode;
#[cfg(feature = "serde")]
pub(super) use stages::{MergeGroupedReduction, SplitState};
#[cfg(feature = "serde")]
use stages::{ReducerFinish, StateReducer};

use super::*;

// Only the constructors behind `serde` request the `State` and `Merge` stages.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
pub(super) enum SplitStage {
    Whole,
    State,
    /// Merges states and finalizes them as `Whole` would.
    Merge,
}

/// Combines states and finalizes them.
#[cfg_attr(not(any(feature = "serde", feature = "cov")), allow(dead_code))]
pub(super) trait StateFinish: Clone + Send + Sync + 'static {
    type State: Clone + Send + Sync + 'static;
    fn init(&self) -> Self::State;
    fn combine(&self, a: &mut Self::State, b: &Self::State);
    fn finish(&self, states: Vec<Self::State>) -> PolarsResult<Series>;
}

/// Builds `stage` of a reduction over a `Null` input, which outputs `whole` and has null states.
pub(super) fn null_reduction(stage: SplitStage, whole: Scalar) -> Box<dyn GroupedReduction> {
    let output = match stage {
        SplitStage::State => Scalar::null(DataType::Null),
        SplitStage::Whole | SplitStage::Merge => whole,
    };
    Box::new(NullGroupedReduction::new(output))
}

/// Builds `stage` of `reducer`'s reduction over values of `dtype`.
#[cfg(feature = "serde")]
pub(super) fn split_reduction<R>(
    dtype: DataType,
    reducer: R,
    stage: SplitStage,
) -> Box<dyn GroupedReduction>
where
    R: Reducer,
    R::Value: SplitState,
{
    match stage {
        SplitStage::Whole => Box::new(VecGroupedReduction::new(dtype, reducer)),
        SplitStage::State => Box::new(VecGroupedReduction::new(dtype, StateReducer(reducer))),
        SplitStage::Merge => unreachable!("built by `merge_reduction`"),
    }
}

/// Merges the states of `reducer`'s reduction over values of `values_dtype` and finalizes them.
#[cfg(feature = "serde")]
pub(super) fn merge_reduction<R>(values_dtype: DataType, reducer: R) -> Box<dyn GroupedReduction>
where
    R: Reducer,
    R::Value: SplitState,
{
    Box::new(MergeGroupedReduction::new(ReducerFinish {
        inner: reducer,
        values_dtype,
    }))
}

#[cfg(not(feature = "serde"))]
pub(super) fn split_reduction<R: Reducer>(
    dtype: DataType,
    reducer: R,
    stage: SplitStage,
) -> Box<dyn GroupedReduction> {
    let SplitStage::Whole = stage else {
        unreachable!()
    };
    Box::new(VecGroupedReduction::new(dtype, reducer))
}

#[cfg(not(feature = "serde"))]
pub(super) fn merge_reduction<R: Reducer>(_: DataType, _: R) -> Box<dyn GroupedReduction> {
    unreachable!()
}

// The state and merge stages have no Python entry point, so they are tested here.
#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::super::var_std::*;
    use super::*;

    type Reduction = Box<dyn GroupedReduction>;
    /// The values of one partition and the group of each row.
    type Partition = (Vec<Series>, Vec<IdxSize>);

    const NUM_GROUPS: IdxSize = 4;

    /// Reduces `values[..][i]` into group `groups[i]`.
    fn update(r: &mut dyn GroupedReduction, values: &[Series], groups: &[IdxSize]) {
        let cols: Vec<Column> = values.iter().cloned().map(Column::from).collect();
        let cols: Vec<&Column> = cols.iter().collect();
        let subset: Vec<IdxSize> = (0..groups.len() as IdxSize).collect();
        unsafe { r.update_groups_subset(&cols, &subset, groups, 0).unwrap() };
    }

    fn run(r: &dyn GroupedReduction, values: &[Series], groups: &[IdxSize]) -> Series {
        let mut r = r.new_empty();
        r.resize(NUM_GROUPS);
        update(&mut *r, values, groups);
        r.finalize().unwrap()
    }

    /// Runs `state` on each partition and merges the states with `merge`.
    fn run_split(
        state: &dyn GroupedReduction,
        merge: &dyn GroupedReduction,
        partitions: &[Partition],
    ) -> Series {
        let mut merge = merge.new_empty();
        merge.resize(NUM_GROUPS);
        let all: Vec<IdxSize> = (0..NUM_GROUPS).collect();
        for (values, groups) in partitions {
            let states = run(state, values, groups);
            assert_eq!(states.dtype(), &reduction_output_dtype(state).unwrap());
            update(&mut *merge, &[states], &all);
        }
        merge.finalize().unwrap()
    }

    /// Runs `whole` on all partitions at once.
    fn run_whole(whole: &dyn GroupedReduction, partitions: &[Partition]) -> Series {
        let mut whole = whole.new_empty();
        whole.resize(NUM_GROUPS);
        if let Some(((values, groups), rest)) = partitions.split_first() {
            let mut values = values.clone();
            let mut groups = groups.clone();
            for (more_values, more_groups) in rest {
                for (v, more) in values.iter_mut().zip(more_values) {
                    v.append(more).unwrap();
                }
                groups.extend_from_slice(more_groups);
            }
            let values: Vec<Series> = values.iter().map(|v| v.rechunk()).collect();
            update(&mut *whole, &values, &groups);
        }
        whole.finalize().unwrap()
    }

    /// Asserts equal up to floating-point rounding.
    fn assert_close(left: &Series, right: &Series) {
        assert_eq!(left.dtype(), right.dtype());
        let left = left.cast(&DataType::Float64).unwrap();
        let right = right.cast(&DataType::Float64).unwrap();
        for (l, r) in left.f64().unwrap().iter().zip(right.f64().unwrap().iter()) {
            match (l, r) {
                (Some(l), Some(r)) => assert!(
                    (l - r).abs() <= 1e-9 * l.abs().max(1.0) || (l.is_nan() && r.is_nan()),
                    "{left:?} != {right:?}"
                ),
                (l, r) => assert_eq!(l, r, "{left:?} != {right:?}"),
            }
        }
        assert_eq!(left.len(), right.len());
    }

    /// The whole, state and merge reductions of one family, and its state dtype.
    struct Family {
        whole: Reduction,
        state: Reduction,
        merge: Reduction,
        state_dtype: DataType,
    }

    /// Asserts that merging the states of each partition matches reducing all of them at once.
    fn check_family(f: &Family, cases: &[Vec<Partition>]) {
        assert_eq!(reduction_output_dtype(&*f.state).unwrap(), f.state_dtype);
        assert_eq!(
            reduction_output_dtype(&*f.merge).unwrap(),
            reduction_output_dtype(&*f.whole).unwrap()
        );
        for partitions in cases {
            let whole = run_whole(&*f.whole, partitions);
            let split = run_split(&*f.state, &*f.merge, partitions);
            assert_close(&split, &whole);
        }
    }

    /// `partitions`, one empty partition, and no partitions at all.
    fn with_empty_cases(partitions: Vec<Partition>) -> Vec<Vec<Partition>> {
        let empty = partitions[0].0.iter().map(|s| s.head(Some(0))).collect();
        vec![partitions, vec![(empty, vec![])], vec![]]
    }

    #[test]
    fn moment_split_matches_whole() {
        use polars_compute::moment::VarState;

        let p1 = Series::new(
            "x".into(),
            [
                Some(1.5),
                Some(2.0),
                None,
                Some(-3.0),
                Some(7.25),
                Some(0.0),
            ],
        );
        let p2 = Series::new("x".into(), [Some(4.0), Some(0.5), Some(10.0), None]);
        for dtype in [
            DataType::Float64,
            DataType::Float32,
            DataType::Int32,
            DataType::Boolean,
            DataType::Null,
        ] {
            let cast = |s: &Series| s.cast(&dtype).unwrap();
            // Group 3 gets no values and group 2 a single one.
            let cases = with_empty_cases(vec![
                (vec![cast(&p1)], vec![0, 1, 0, 1, 0, 1]),
                (vec![cast(&p2)], vec![1, 0, 2, 3]),
            ]);
            let state_dtype = |numeric: DataType| match dtype {
                DataType::Null => DataType::Null,
                DataType::Boolean => BoolVarState::state_dtype(),
                _ => numeric,
            };
            let dt = || dtype.clone();
            #[cfg_attr(not(feature = "moment"), allow(unused_mut))]
            let mut families = vec![
                Family {
                    whole: new_var_std_reduction(dt(), false, 1).unwrap(),
                    state: new_var_std_state_reduction(dt()).unwrap(),
                    merge: new_var_std_merge_reduction(dt(), false, 1).unwrap(),
                    state_dtype: state_dtype(VarState::state_dtype()),
                },
                Family {
                    whole: new_var_std_reduction(dt(), true, 0).unwrap(),
                    state: new_var_std_state_reduction(dt()).unwrap(),
                    merge: new_var_std_merge_reduction(dt(), true, 0).unwrap(),
                    state_dtype: state_dtype(VarState::state_dtype()),
                },
            ];
            #[cfg(feature = "moment")]
            if !dtype.is_bool() {
                use polars_compute::moment::{KurtosisState, SkewState};

                use super::super::skew_kurtosis::*;

                families.push(Family {
                    whole: new_skew_reduction(dt(), false).unwrap(),
                    state: new_skew_state_reduction(dt()).unwrap(),
                    merge: new_skew_merge_reduction(dt(), false).unwrap(),
                    state_dtype: state_dtype(SkewState::state_dtype()),
                });
                families.push(Family {
                    whole: new_kurtosis_reduction(dt(), true, false).unwrap(),
                    state: new_kurtosis_state_reduction(dt()).unwrap(),
                    merge: new_kurtosis_merge_reduction(dt(), true, false).unwrap(),
                    state_dtype: state_dtype(KurtosisState::state_dtype()),
                });
            }
            for f in &families {
                check_family(f, &cases);
            }
        }
    }

    /// A state column with one state per group of `values`.
    fn var_states(values: &[f64], groups: &[IdxSize]) -> Column {
        let state = new_var_std_state_reduction(DataType::Float64).unwrap();
        let values = Series::new("x".into(), values);
        Column::from(run(&*state, &[values], groups))
    }

    fn var_merge() -> Reduction {
        new_var_std_merge_reduction(DataType::Float64, false, 0).unwrap()
    }

    #[test]
    fn merge_reads_sparse_subset() {
        let states = var_states(&[1.0, 2.0, 10.0, 20.0, 40.0, 5.0], &[0, 0, 1, 1, 1, 2]);
        let mut merge = var_merge();
        merge.resize(2);
        unsafe {
            merge
                .update_groups_subset(&[&states], &[2, 0], &[1, 0], 0)
                .unwrap()
        };
        let out = merge.finalize().unwrap();
        assert_close(&out, &Series::new("".into(), [0.25, 0.0]));
    }

    #[test]
    fn merge_evicts_on_null_state() {
        let states = var_states(&[1.0, 2.0], &[0, 0]).head(Some(1));
        let states = Column::from(
            states
                .as_materialized_series()
                .extend_constant(AnyValue::Null, 1)
                .unwrap()
                .rechunk(),
        );
        let mut merge = var_merge();
        merge.resize(1);
        unsafe {
            merge
                .update_groups_while_evicting(&[&states], &[0], &[EvictIdx::new(0, false)], 0)
                .unwrap();
            merge
                .update_groups_while_evicting(&[&states], &[1], &[EvictIdx::new(0, true)], 0)
                .unwrap();
        }
        let evicted = merge.take_evictions().finalize().unwrap();
        assert_close(&evicted, &Series::new("".into(), [0.25]));
        let out = merge.finalize().unwrap();
        assert_close(&out, &Series::new("".into(), [None::<f64>]));
    }

    #[test]
    fn merge_combines_subset() {
        let states = var_states(&[1.0, 2.0, 10.0, 20.0], &[0, 0, 1, 1]);
        let mut a = var_merge();
        let mut b = var_merge();
        a.resize(2);
        b.resize(2);
        unsafe {
            a.update_groups_subset(&[&states], &[0], &[0], 0).unwrap();
            b.update_groups_subset(&[&states], &[0, 1], &[0, 1], 0)
                .unwrap();
            // a[0] = {1, 2}, a[1] = {1, 2} + {10, 20}.
            a.combine_subset(&*b, &[1, 0], &[1, 1]).unwrap();
        }
        let out = a.finalize().unwrap();
        assert_close(&out, &Series::new("".into(), [0.25, 58.1875]));
    }

    #[cfg(feature = "cov")]
    #[test]
    fn cov_corr_split_matches_whole() {
        use polars_compute::moment::{CovState, PearsonState};

        use super::super::cov::*;

        let x1 = Series::new(
            "x".into(),
            [Some(1.0), Some(2.0), None, Some(4.0), Some(-1.0)],
        );
        let y1 = Series::new(
            "y".into(),
            [Some(2.0), Some(3.5), Some(1.0), Some(9.0), None],
        );
        let x2 = Series::new("x".into(), [Some(3.0), Some(0.5), Some(6.0), Some(7.0)]);
        let y2 = Series::new("y".into(), [Some(1.0), Some(4.0), Some(2.0), Some(7.5)]);
        for (dx, dy) in [
            (DataType::Float64, DataType::Float64),
            (DataType::Int32, DataType::Float32),
            (DataType::Float32, DataType::Float32),
        ] {
            let pair = |x: &Series, y: &Series| vec![x.cast(&dx).unwrap(), y.cast(&dy).unwrap()];
            let cases = with_empty_cases(vec![
                (pair(&x1, &y1), vec![0, 1, 0, 0, 1]),
                (pair(&x2, &y2), vec![1, 0, 1, 2]),
            ]);
            let (x, y) = (|| dx.clone(), || dy.clone());
            let families = [
                Family {
                    whole: new_cov_reduction(x(), y(), 1).unwrap(),
                    state: new_cov_state_reduction(x(), y()).unwrap(),
                    merge: new_cov_merge_reduction(x(), y(), 1).unwrap(),
                    state_dtype: CovState::state_dtype(),
                },
                Family {
                    whole: new_pearson_corr_reduction(x(), y()).unwrap(),
                    state: new_pearson_corr_state_reduction(x(), y()).unwrap(),
                    merge: new_pearson_corr_merge_reduction(x(), y()).unwrap(),
                    state_dtype: PearsonState::state_dtype(),
                },
            ];
            for f in &families {
                check_family(f, &cases);
            }
        }
    }

    #[cfg(feature = "approx_unique")]
    #[test]
    fn approx_n_unique_split_matches_whole() {
        use polars_utils::cardinality_sketch::CardinalitySketch;

        use super::super::approx_n_unique::*;

        let int = |v: &[Option<i32>]| vec![Series::new("x".into(), v)];
        let null = |n: usize| vec![Series::full_null("x".into(), n, &DataType::Null)];
        let mut int_cases = with_empty_cases(vec![
            (int(&[Some(1), Some(2), None, Some(2)]), vec![0, 1, 0, 1]),
            (int(&[Some(3), Some(1), Some(2)]), vec![1, 0, 2]),
        ]);
        int_cases.push(vec![
            (int(&[None, None]), vec![0, 1]),
            (int(&[None]), vec![0]),
        ]);
        let null_cases = with_empty_cases(vec![(null(2), vec![0, 0]), (null(1), vec![1])]);
        for (dtype, state_dtype, cases) in [
            (
                DataType::Int32,
                CardinalitySketch::state_dtype(),
                &int_cases,
            ),
            (DataType::Null, DataType::Null, &null_cases),
        ] {
            let f = Family {
                whole: new_approx_n_unique_reduction(dtype.clone()).unwrap(),
                state: new_approx_n_unique_state_reduction(dtype.clone()).unwrap(),
                merge: new_approx_n_unique_merge_reduction(dtype).unwrap(),
                state_dtype,
            };
            check_family(&f, cases);
        }

        // A `Null` input has one distinct value in every group, even an empty one.
        let state = new_approx_n_unique_state_reduction(DataType::Null).unwrap();
        let merge = new_approx_n_unique_merge_reduction(DataType::Null).unwrap();
        let out = run_split(&*state, &*merge, &null_cases[0]);
        assert!(out.equals(&Series::new("".into(), [1 as IdxSize; NUM_GROUPS as usize])));
    }
}
