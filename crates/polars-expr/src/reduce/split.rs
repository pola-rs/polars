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

    /// Reduces `values[..][i]` into group `groups[i]` of `num_groups` and finalizes.
    fn run(
        r: &dyn GroupedReduction,
        values: &[Series],
        groups: &[IdxSize],
        num_groups: IdxSize,
    ) -> PolarsResult<Series> {
        let mut r = r.new_empty();
        r.resize(num_groups);
        let cols: Vec<Column> = values.iter().cloned().map(Column::from).collect();
        let cols: Vec<&Column> = cols.iter().collect();
        let subset: Vec<IdxSize> = (0..groups.len() as IdxSize).collect();
        unsafe { r.update_groups_subset(&cols, &subset, groups, 0)? };
        r.finalize()
    }

    struct Partition {
        values: Vec<Series>,
        groups: Vec<IdxSize>,
    }

    fn partition(values: Vec<Series>, groups: &[IdxSize]) -> Partition {
        Partition {
            values,
            groups: groups.to_vec(),
        }
    }

    /// Runs `state` on each partition and merges the states with `merge`.
    fn run_split(
        state: &dyn GroupedReduction,
        merge: &dyn GroupedReduction,
        partitions: &[Partition],
        num_groups: IdxSize,
    ) -> PolarsResult<Series> {
        let mut merge = merge.new_empty();
        merge.resize(num_groups);
        let all: Vec<IdxSize> = (0..num_groups).collect();
        for p in partitions {
            let states = Column::from(run(state, &p.values, &p.groups, num_groups)?);
            assert_eq!(states.dtype(), &reduction_output_dtype(state)?);
            unsafe { merge.update_groups_subset(&[&states], &all, &all, 0)? };
        }
        merge.finalize()
    }

    /// Runs `whole` on all partitions at once.
    fn run_whole(
        whole: &dyn GroupedReduction,
        partitions: &[Partition],
        num_groups: IdxSize,
    ) -> PolarsResult<Series> {
        let mut values = partitions[0].values.clone();
        let mut groups = partitions[0].groups.clone();
        for p in &partitions[1..] {
            for (v, pv) in values.iter_mut().zip(&p.values) {
                v.append(pv)?;
            }
            groups.extend_from_slice(&p.groups);
        }
        let values: Vec<Series> = values.iter().map(|v| v.rechunk()).collect();
        run(whole, &values, &groups, num_groups)
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

    /// The whole, state and merge reductions of one family.
    struct Family {
        whole: Reduction,
        state: Reduction,
        merge: Reduction,
    }

    fn moment_families(dtype: &DataType) -> Vec<Family> {
        let dt = || dtype.clone();
        #[cfg_attr(not(feature = "moment"), allow(unused_mut))]
        let mut out = vec![
            Family {
                whole: new_var_std_reduction(dt(), false, 1).unwrap(),
                state: new_var_std_state_reduction(dt()).unwrap(),
                merge: new_var_std_merge_reduction(dt(), false, 1).unwrap(),
            },
            Family {
                whole: new_var_std_reduction(dt(), true, 0).unwrap(),
                state: new_var_std_state_reduction(dt()).unwrap(),
                merge: new_var_std_merge_reduction(dt(), true, 0).unwrap(),
            },
        ];
        #[cfg(feature = "moment")]
        if !dtype.is_bool() {
            use super::super::skew_kurtosis::*;
            out.push(Family {
                whole: new_skew_reduction(dt(), false).unwrap(),
                state: new_skew_state_reduction(dt()).unwrap(),
                merge: new_skew_merge_reduction(dt(), false).unwrap(),
            });
            out.push(Family {
                whole: new_kurtosis_reduction(dt(), true, false).unwrap(),
                state: new_kurtosis_state_reduction(dt()).unwrap(),
                merge: new_kurtosis_merge_reduction(dt(), true, false).unwrap(),
            });
        }
        out
    }

    /// Two partitions over four groups; group 3 gets no values and group 2 a single one.
    fn moment_partitions(dtype: &DataType) -> Vec<Partition> {
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
        let cast = |s: Series| s.cast(dtype).unwrap();
        vec![
            partition(vec![cast(p1)], &[0, 1, 0, 1, 0, 1]),
            partition(vec![cast(p2)], &[1, 0, 2, 3]),
        ]
    }

    #[test]
    fn moment_split_matches_whole() {
        for dtype in [
            DataType::Float64,
            DataType::Float32,
            DataType::Int32,
            DataType::Boolean,
        ] {
            let partitions = moment_partitions(&dtype);
            for f in moment_families(&dtype) {
                let whole = run_whole(&*f.whole, &partitions, 4).unwrap();
                let split = run_split(&*f.state, &*f.merge, &partitions, 4).unwrap();
                assert_close(&split, &whole);
                assert!(reduction_output_dtype(&*f.state).unwrap().is_struct());
                assert_eq!(reduction_output_dtype(&*f.merge).unwrap(), *whole.dtype());
            }
        }
    }

    #[test]
    fn moment_split_of_empty_groups_matches_whole() {
        let dtype = DataType::Float64;
        let empty = vec![partition(vec![Series::new_empty("x".into(), &dtype)], &[])];
        for f in moment_families(&dtype) {
            let whole = run_whole(&*f.whole, &empty, 2).unwrap();
            let split = run_split(&*f.state, &*f.merge, &[], 2).unwrap();
            assert_close(&split, &whole);
            let split = run_split(&*f.state, &*f.merge, &empty, 2).unwrap();
            assert_close(&split, &whole);
        }
    }

    #[test]
    fn moment_split_of_null_input() {
        let dtype = DataType::Null;
        let partitions = vec![partition(
            vec![Series::full_null("x".into(), 3, &dtype)],
            &[0, 1, 0],
        )];
        for f in moment_families(&dtype) {
            assert_eq!(reduction_output_dtype(&*f.state).unwrap(), DataType::Null);
            let whole = run_whole(&*f.whole, &partitions, 2).unwrap();
            let split = run_split(&*f.state, &*f.merge, &partitions, 2).unwrap();
            assert!(split.equals_missing(&whole));
            assert_eq!(split.dtype(), &DataType::Null);
        }
    }

    /// A state column with one state per group of `values`.
    fn var_states(values: &[f64], groups: &[IdxSize], num_groups: IdxSize) -> Column {
        let state = new_var_std_state_reduction(DataType::Float64).unwrap();
        let values = Series::new("x".into(), values);
        Column::from(run(&*state, &[values], groups, num_groups).unwrap())
    }

    fn var_merge() -> Reduction {
        new_var_std_merge_reduction(DataType::Float64, false, 0).unwrap()
    }

    #[test]
    fn merge_reads_sparse_subset() {
        let states = var_states(&[1.0, 2.0, 10.0, 20.0, 40.0, 5.0], &[0, 0, 1, 1, 1, 2], 3);
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
        let states = var_states(&[1.0, 2.0], &[0, 0], 1);
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
        let states = var_states(&[1.0, 2.0, 10.0, 20.0], &[0, 0, 1, 1], 2);
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
        let expected = var_states(&[1.0, 2.0, 1.0, 2.0, 10.0, 20.0], &[0, 0, 1, 1, 1, 1], 2);
        let mut merge = var_merge();
        merge.resize(2);
        unsafe {
            merge
                .update_groups_subset(&[&expected], &[0, 1], &[0, 1], 0)
                .unwrap()
        };
        assert_close(&out, &merge.finalize().unwrap());
    }

    #[cfg(feature = "cov")]
    #[test]
    fn cov_corr_split_matches_whole() {
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
            let cast = |s: &Series, dtype: &DataType| s.cast(dtype).unwrap();
            let partitions = [
                partition(vec![cast(&x1, &dx), cast(&y1, &dy)], &[0, 1, 0, 0, 1]),
                partition(vec![cast(&x2, &dx), cast(&y2, &dy)], &[1, 0, 1, 2]),
            ];
            let empty = [partition(
                vec![cast(&x1, &dx).head(Some(0)), cast(&y1, &dy).head(Some(0))],
                &[],
            )];
            let (x, y) = (|| dx.clone(), || dy.clone());
            let families = [
                Family {
                    whole: new_cov_reduction(x(), y(), 1).unwrap(),
                    state: new_cov_state_reduction(x(), y()).unwrap(),
                    merge: new_cov_merge_reduction(x(), y(), 1).unwrap(),
                },
                Family {
                    whole: new_pearson_corr_reduction(x(), y()).unwrap(),
                    state: new_pearson_corr_state_reduction(x(), y()).unwrap(),
                    merge: new_pearson_corr_merge_reduction(x(), y()).unwrap(),
                },
            ];
            for f in families {
                for partitions in [&partitions[..], &empty[..]] {
                    let whole = run_whole(&*f.whole, partitions, 4).unwrap();
                    let split = run_split(&*f.state, &*f.merge, partitions, 4).unwrap();
                    assert_close(&split, &whole);
                }
                assert!(reduction_output_dtype(&*f.state).unwrap().is_struct());
                assert_eq!(
                    reduction_output_dtype(&*f.merge).unwrap(),
                    reduction_output_dtype(&*f.whole).unwrap()
                );
            }
        }
    }

    #[cfg(feature = "approx_unique")]
    #[test]
    fn approx_n_unique_split_matches_whole() {
        use polars_utils::cardinality_sketch::CardinalitySketch;

        use super::super::approx_n_unique::*;

        let int = |v: &[Option<i32>]| Series::new("x".into(), v);
        let null = |n: usize| Series::full_null("x".into(), n, &DataType::Null);
        let cases = [
            // Mixed.
            vec![
                partition(vec![int(&[Some(1), Some(2), None, Some(2)])], &[0, 1, 0, 1]),
                partition(vec![int(&[Some(3), Some(1), Some(2)])], &[1, 0, 2]),
            ],
            // Empty.
            vec![partition(vec![int(&[])], &[])],
            // All null.
            vec![
                partition(vec![int(&[None, None])], &[0, 1]),
                partition(vec![int(&[None])], &[0]),
            ],
            // `Null` dtype.
            vec![
                partition(vec![null(2)], &[0, 0]),
                partition(vec![null(1)], &[1]),
            ],
            vec![partition(vec![null(0)], &[])],
        ];
        let merge = new_approx_n_unique_merge_reduction();
        for partitions in cases {
            let dtype = partitions[0].values[0].dtype().clone();
            let whole = new_approx_n_unique_reduction(dtype.clone()).unwrap();
            let state = new_approx_n_unique_state_reduction(dtype).unwrap();
            assert_eq!(
                reduction_output_dtype(&*state).unwrap(),
                CardinalitySketch::state_dtype()
            );
            let expected = run_whole(&*whole, &partitions, 4).unwrap();
            let split = run_split(&*state, &*merge, &partitions, 4).unwrap();
            assert!(split.equals_missing(&expected), "{split:?} != {expected:?}");
        }
        assert_eq!(reduction_output_dtype(&*merge).unwrap(), IDX_DTYPE);
    }
}
