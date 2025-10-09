use std::marker::PhantomData;

use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalHash};

use super::*;

pub fn new_approx_n_unique_reduction(dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use DataType::*;
    use {ApproxNUniqueReducer as R, VecGroupedReduction as VGR};
    Ok(match dtype {
        Boolean => Box::new(VGR::new(dtype, R::<BooleanType>::default())),
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, R::<$T>::default()))
            })
        },
        String => Box::new(VGR::new(dtype, R::<StringType>::default())),
        Binary => Box::new(VGR::new(dtype, R::<BinaryType>::default())),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(dtype, R::<Int128Type>::default())),
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(_, _) | DataType::Categorical(_, _) => match dtype.cat_physical().unwrap() {
            CategoricalPhysical::U8 => Box::new(VGR::new(dtype, R::<UInt8Type>::default())),
            CategoricalPhysical::U16 => Box::new(VGR::new(dtype, R::<UInt16Type>::default())),
            CategoricalPhysical::U32 => Box::new(VGR::new(dtype, R::<UInt32Type>::default())),
        },
        Null => Box::new(super::NullGroupedReduction::new(Scalar::new_idxsize(1))),
        _ => {
            polars_bail!(InvalidOperation: "`approx_n_unique` operation not supported for dtype `{dtype}`")
        },
    })
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
            hasher: self.hasher,
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
