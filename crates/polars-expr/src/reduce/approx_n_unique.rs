use std::marker::PhantomData;

use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalHash};

#[cfg(feature = "serde")]
use super::split::SplitState;
use super::split::{SplitStage, split_reduction};
use super::*;

pub fn new_approx_n_unique_reduction(dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    approx_n_unique_reduction(dtype, SplitStage::Whole)
}

/// Like [`new_approx_n_unique_reduction`], but outputs the sketch of each group, to be merged by
/// [`new_approx_n_unique_merge_reduction`].
#[cfg(feature = "serde")]
pub fn new_approx_n_unique_state_reduction(
    dtype: DataType,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    approx_n_unique_reduction(dtype, SplitStage::State)
}

/// Merges the states of [`new_approx_n_unique_state_reduction`] over values of any dtype, and
/// outputs what [`new_approx_n_unique_reduction`] does.
#[cfg(feature = "serde")]
pub fn new_approx_n_unique_merge_reduction() -> Box<dyn GroupedReduction> {
    // `finish` ignores the input type, so one reducer merges the states of every input.
    split_reduction(
        DataType::UInt32,
        ApproxNUniqueReducer::<UInt32Type>::default(),
        SplitStage::Merge,
    )
}

fn approx_n_unique_reduction(
    dtype: DataType,
    stage: SplitStage,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use ApproxNUniqueReducer as R;
    use DataType::*;
    Ok(match dtype {
        Boolean => split_reduction(dtype, R::<BooleanType>::default(), stage),
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                split_reduction(dtype, R::<$T>::default(), stage)
            })
        },
        String => split_reduction(dtype, R::<StringType>::default(), stage),
        Binary => split_reduction(dtype, R::<BinaryType>::default(), stage),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => split_reduction(dtype, R::<Int128Type>::default(), stage),
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(_, _) | DataType::Categorical(_, _) => match dtype.cat_physical().unwrap() {
            CategoricalPhysical::U8 => split_reduction(dtype, R::<UInt8Type>::default(), stage),
            CategoricalPhysical::U16 => split_reduction(dtype, R::<UInt16Type>::default(), stage),
            CategoricalPhysical::U32 => split_reduction(dtype, R::<UInt32Type>::default(), stage),
        },
        Null => match stage {
            #[cfg(feature = "serde")]
            SplitStage::State => Box::new(super::NullGroupedReduction::new(null_input_state()?)),
            _ => Box::new(super::NullGroupedReduction::new(Scalar::new_idxsize(1))),
        },
        _ => {
            polars_bail!(InvalidOperation: "`approx_n_unique` operation not supported for dtype `{dtype}`")
        },
    })
}

/// The state of every group of a `Null` input: a sketch of one null, which estimates the `1`
/// that `Whole` returns, even for an empty group.
#[cfg(feature = "serde")]
fn null_input_state() -> PolarsResult<Scalar> {
    let mut sketch = CardinalitySketch::new();
    sketch.insert(PlFixedStateQuality::default().tot_hash_one(None::<bool>));
    let states = CardinalitySketch::encode(vec![sketch])?;
    Ok(Scalar::new(
        states.dtype().clone(),
        states.get(0)?.into_static(),
    ))
}

struct ApproxNUniqueReducer<T> {
    hasher: PlFixedStateQuality,
    marker: PhantomData<T>,
}

impl<T> Default for ApproxNUniqueReducer<T> {
    fn default() -> Self {
        Self {
            hasher: PlFixedStateQuality::default(),
            marker: PhantomData,
        }
    }
}

impl<T> Clone for ApproxNUniqueReducer<T> {
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
            marker: PhantomData,
        }
    }
}

impl<T> Reducer for ApproxNUniqueReducer<T>
where
    T: PolarsPhysicalType,
    for<'a> T::Physical<'a>: TotalHash,
{
    type Dtype = T;
    type Value = CardinalitySketch;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        CardinalitySketch::new()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.combine(b);
    }

    #[inline(always)]
    fn reduce_one(
        &self,
        a: &mut Self::Value,
        b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
        _seq_id: u64,
    ) {
        let hash = self.hasher.tot_hash_one(b);
        a.insert(hash);
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        for val in ca.iter() {
            let hash = self.hasher.tot_hash_one(val);
            v.insert(hash);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let ca: IdxCa = v
            .into_iter()
            .map(|sketch| sketch.estimate().min(IdxSize::MAX as usize) as IdxSize)
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}
