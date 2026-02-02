#![allow(unsafe_op_in_unsafe_fn)]
use std::borrow::Cow;
use std::marker::PhantomData;

use num_traits::Bounded;
use polars_core::chunked_array::arg_min_max::{
    arg_max_binary, arg_max_bool, arg_max_numeric, arg_min_binary, arg_min_bool, arg_min_numeric,
};
use polars_core::with_match_physical_integer_polars_type;
use polars_utils::arg_min_max::ArgMinMax;
use polars_utils::float::IsFloat;
use polars_utils::min_max::MinMax;

use super::*;
use crate::reduce::first_last::new_last_reduction;

pub fn new_min_by_reduction(
    dtype: DataType,
    by_dtype: DataType,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use DataType::*;
    use SelectPayloadGroupedReduction as SPGR;
    let payload = new_last_reduction(dtype.clone());
    Ok(match &by_dtype {
        Boolean => Box::new(SPGR::new(by_dtype, BooleanMinSelector, payload)),
        #[cfg(all(feature = "dtype-f16", feature = "propagate_nans"))]
        #[cfg(feature = "dtype-f16")]
        Float16 => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Float16Type>(PhantomData),
            payload,
        )),
        Float32 => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Float32Type>(PhantomData),
            payload,
        )),
        Float64 => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Float64Type>(PhantomData),
            payload,
        )),
        Null => Box::new(NullGroupedReduction::new(Scalar::null(dtype))),
        String | Binary => Box::new(SPGR::new(by_dtype, BinaryMinSelector, payload)),
        _ if by_dtype.is_integer() || by_dtype.is_temporal() || by_dtype.is_enum() => {
            with_match_physical_integer_polars_type!(by_dtype.to_physical(), |$T| {
                Box::new(SPGR::new(by_dtype, MinSelector::<$T>(PhantomData), payload))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Int128Type>(PhantomData),
            payload,
        )),
        #[cfg(feature = "dtype-categorical")]
        Categorical(cats, map) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            let map = map.clone();
            Box::new(SPGR::new(by_dtype, CatMinSelector::<$C>(map, PhantomData), payload))
        }),
        _ => {
            polars_bail!(InvalidOperation: "`min_by` operation not supported for by dtype `{by_dtype}`")
        },
    })
}

pub fn new_max_by_reduction(
    dtype: DataType,
    by_dtype: DataType,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use DataType::*;
    use SelectPayloadGroupedReduction as SPGR;
    let payload = new_last_reduction(dtype.clone());
    Ok(match &by_dtype {
        Boolean => Box::new(SPGR::new(by_dtype, BooleanMaxSelector, payload)),
        #[cfg(all(feature = "dtype-f16", feature = "propagate_nans"))]
        #[cfg(feature = "dtype-f16")]
        Float16 => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Float16Type>(PhantomData),
            payload,
        )),
        Float32 => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Float32Type>(PhantomData),
            payload,
        )),
        Float64 => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Float64Type>(PhantomData),
            payload,
        )),
        Null => Box::new(NullGroupedReduction::new(Scalar::null(dtype))),
        String | Binary => Box::new(SPGR::new(by_dtype, BinaryMaxSelector, payload)),
        _ if by_dtype.is_integer() || by_dtype.is_temporal() || by_dtype.is_enum() => {
            with_match_physical_integer_polars_type!(by_dtype.to_physical(), |$T| {
                Box::new(SPGR::new(by_dtype, MaxSelector::<$T>(PhantomData), payload))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Int128Type>(PhantomData),
            payload,
        )),
        #[cfg(feature = "dtype-categorical")]
        Categorical(cats, map) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            let map = map.clone();
            Box::new(SPGR::new(by_dtype, CatMaxSelector::<$C>(map, PhantomData), payload))
        }),
        _ => {
            polars_bail!(InvalidOperation: "`max_by` operation not supported for by dtype `{by_dtype}`")
        },
    })
}

trait SelectReducer: Clone + Send + Sync + 'static {
    type Value: Clone + Send + Sync + 'static;
    type Dtype: PolarsPhysicalType;

    fn init(&self) -> Self::Value;

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Borrowed(s)
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize>;

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool;

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool;
}

struct MinSelector<T>(PhantomData<T>);
struct MaxSelector<T>(PhantomData<T>);

impl<T> Clone for MinSelector<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T> SelectReducer for MinSelector<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    type Value = T::Native;
    type Dtype = T;

    fn init(&self) -> Self::Value {
        if T::Native::is_float() {
            T::Native::nan_value()
        } else {
            T::Native::max_value()
        }
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_min_numeric(ca).filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        let better = b.nan_max_lt(a);
        if better {
            *a = b;
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, *b)
    }
}

impl<T> Clone for MaxSelector<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T> SelectReducer for MaxSelector<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    type Value = T::Native;
    type Dtype = T;

    fn init(&self) -> Self::Value {
        if T::Native::is_float() {
            T::Native::nan_value()
        } else {
            T::Native::min_value()
        }
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_max_numeric(ca).filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        let better = b.nan_min_gt(a);
        if better {
            *a = b;
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, *b)
    }
}

#[derive(Clone)]
struct BinaryMinSelector;
#[derive(Clone)]
struct BinaryMaxSelector;

impl SelectReducer for BinaryMinSelector {
    type Dtype = BinaryType;
    type Value = Option<Vec<u8>>;

    fn init(&self) -> Self::Value {
        // There's no "maximum string" initializer.
        None
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_min_binary(ca).filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        if let Some(av) = a {
            if b < av.as_slice() {
                av.clear();
                av.extend_from_slice(b);
                true
            } else {
                false
            }
        } else {
            *a = Some(b.to_vec());
            true
        }
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        if let Some(bv) = b {
            self.select_one(a, bv)
        } else {
            false
        }
    }
}

impl SelectReducer for BinaryMaxSelector {
    type Dtype = BinaryType;
    type Value = Vec<u8>;

    fn init(&self) -> Self::Value {
        // Empty string is <= any other string, so can initialize max with it.
        Vec::new()
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_max_binary(ca).filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        let better = b > a.as_slice();
        if better {
            a.clear();
            a.extend_from_slice(b);
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, b)
    }
}

#[derive(Clone)]
struct BooleanMinSelector;
#[derive(Clone)]
struct BooleanMaxSelector;

impl SelectReducer for BooleanMinSelector {
    type Value = bool;
    type Dtype = BooleanType;

    fn init(&self) -> Self::Value {
        true
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_min_bool(ca).filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        #[allow(clippy::bool_comparison)]
        let better = b < *a;
        if better {
            *a = b;
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, *b)
    }
}

impl SelectReducer for BooleanMaxSelector {
    type Value = bool;
    type Dtype = BooleanType;

    fn init(&self) -> Self::Value {
        false
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_max_bool(ca).filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        #[allow(clippy::bool_comparison)]
        let better = b > *a;
        if better {
            *a = b;
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, *b)
    }
}

#[cfg(feature = "dtype-categorical")]
struct CatMinSelector<T>(Arc<CategoricalMapping>, PhantomData<T>);

#[cfg(feature = "dtype-categorical")]
impl<T> Clone for CatMinSelector<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

#[cfg(feature = "dtype-categorical")]
impl<T: PolarsCategoricalType> SelectReducer for CatMinSelector<T> {
    type Dtype = T::PolarsPhysical;
    type Value = T::Native;

    fn init(&self) -> Self::Value {
        T::Native::max_value() // Ensures it's invalid, preferring the other value.
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        use polars_core::chunked_array::arg_min_max::arg_min_opt_iter;
        let arg_min = arg_min_opt_iter(ca.iter().map(|cat| self.0.cat_to_str(cat?.as_cat())));
        arg_min.filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        let Some(b_s) = self.0.cat_to_str(b.as_cat()) else {
            return false;
        };
        let Some(a_s) = self.0.cat_to_str(a.as_cat()) else {
            *a = b;
            return true;
        };

        let better = b_s < a_s;
        if better {
            *a = b;
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, *b)
    }
}

#[cfg(feature = "dtype-categorical")]
struct CatMaxSelector<T>(Arc<CategoricalMapping>, PhantomData<T>);

#[cfg(feature = "dtype-categorical")]
impl<T> Clone for CatMaxSelector<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

#[cfg(feature = "dtype-categorical")]
impl<T: PolarsCategoricalType> SelectReducer for CatMaxSelector<T> {
    type Dtype = T::PolarsPhysical;
    type Value = T::Native;

    fn init(&self) -> Self::Value {
        T::Native::max_value() // Ensures it's invalid, preferring the other value.
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn select_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        use polars_core::chunked_array::arg_min_max::arg_max_opt_iter;
        let arg_max = arg_max_opt_iter(ca.iter().map(|cat| self.0.cat_to_str(cat?.as_cat())));
        arg_max.filter(|idx| {
            let val = unsafe { ca.value_unchecked(*idx) };
            self.select_one(v, val)
        })
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        let Some(b_s) = self.0.cat_to_str(b.as_cat()) else {
            return false;
        };
        let Some(a_s) = self.0.cat_to_str(a.as_cat()) else {
            *a = b;
            return true;
        };

        let better = b_s > a_s;
        if better {
            *a = b;
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, *b)
    }
}

struct SelectPayloadGroupedReduction<R: SelectReducer> {
    values: Vec<R::Value>,
    mask: MutableBitmap,
    evicted_values: Vec<R::Value>,
    evicted_mask: BitmapBuilder,
    in_dtype: DataType,
    reducer: R,
    payload: Box<dyn GroupedReduction>,

    tmp_subset: Vec<IdxSize>,
    tmp_group_idxs: Vec<IdxSize>,
}

impl<R: SelectReducer> SelectPayloadGroupedReduction<R> {
    fn new(in_dtype: DataType, reducer: R, payload: Box<dyn GroupedReduction>) -> Self {
        Self {
            values: Vec::new(),
            mask: MutableBitmap::new(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            in_dtype,
            reducer,
            payload,
            tmp_subset: Vec::new(),
            tmp_group_idxs: Vec::new(),
        }
    }
}

impl<R> GroupedReduction for SelectPayloadGroupedReduction<R>
where
    R: SelectReducer,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(
            self.in_dtype.clone(),
            self.reducer.clone(),
            self.payload.new_empty(),
        ))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.mask.reserve(additional);
        self.payload.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, self.reducer.init());
        self.mask.resize(num_groups as usize, false);
        self.payload.resize(num_groups);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == 2);
        let payload_values = values[0];
        let ord_values = values[1];
        assert_eq!(ord_values.dtype(), &self.in_dtype);

        let ord_values = self
            .reducer
            .cast_series(ord_values.as_materialized_series());
        let ca: &ChunkedArray<R::Dtype> = ord_values.as_ref().as_ref().as_ref();

        if let Some(selected) = self
            .reducer
            .select_ca(&mut self.values[group_idx as usize], ca)
        {
            self.mask.set(group_idx as usize, true);
            let selected_val = payload_values.new_from_index(selected, 1);
            self.payload.update_group(&[&selected_val], group_idx, 0)?;
        }

        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == 2);
        let payload_values = values[0];
        let ord_values = values[1];
        assert!(ord_values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());

        // TODO: @scalar-opt
        let ord_values = self
            .reducer
            .cast_series(ord_values.as_materialized_series());
        let ca: &ChunkedArray<R::Dtype> = ord_values.as_ref().as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            self.tmp_subset.clear();
            self.tmp_group_idxs.clear();

            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                let grp = self.values.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evicted_values
                        .push(core::mem::replace(grp, self.reducer.init()));
                    self.evicted_mask.push(self.mask.get_unchecked(g.idx()));
                    self.mask.set_unchecked(g.idx(), ov.is_some());
                    if let Some(v) = ov {
                        self.reducer.select_one(grp, v);
                    }
                    self.tmp_subset.push(*i);
                    self.tmp_group_idxs.push(g.0);
                } else if let Some(v) = ov {
                    if self.mask.get_unchecked(g.idx()) {
                        if self.reducer.select_one(grp, v) {
                            self.tmp_subset.push(*i);
                            self.tmp_group_idxs.push(g.0);
                        }
                    } else {
                        self.mask.set_unchecked(g.idx(), true);
                        self.reducer.select_one(grp, v);
                        self.tmp_subset.push(*i);
                        self.tmp_group_idxs.push(g.0);
                    }
                }
            }

            self.payload.update_groups_while_evicting(
                &[payload_values],
                &self.tmp_subset,
                EvictIdx::cast_slice(&self.tmp_group_idxs),
                0, // seq_id is unused
            )?;
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(self.in_dtype == other.in_dtype);
        assert!(subset.len() == group_idxs.len());
        unsafe {
            self.tmp_subset.clear();
            self.tmp_group_idxs.clear();

            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let o = other.mask.get_unchecked(*i as usize);
                if o {
                    let v = other.values.get_unchecked(*i as usize);
                    let grp = self.values.get_unchecked_mut(*g as usize);
                    if self.reducer.select_combine(grp, v) | !self.mask.get_unchecked(*g as usize) {
                        self.tmp_subset.push(*i);
                        self.tmp_group_idxs.push(*g);
                    }
                    self.mask.set_unchecked(*g as usize, true);
                }
            }

            self.payload.combine_subset(
                other.payload.as_ref(),
                &self.tmp_subset,
                &self.tmp_group_idxs,
            )?;
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values),
            mask: core::mem::take(&mut self.evicted_mask).into_mut(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
            payload: self.payload.take_evictions(),
            tmp_group_idxs: Vec::new(),
            tmp_subset: Vec::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let mask = core::mem::take(&mut self.mask);
        drop(core::mem::take(&mut self.values));
        drop(core::mem::take(&mut self.tmp_group_idxs));
        drop(core::mem::take(&mut self.tmp_subset));

        // TODO @ minmax-by: better way to combine payload and mask.
        let data = self.payload.finalize()?;
        let mca = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, mask.freeze());
        let nulls = Series::full_null(data.name().clone(), 1, data.dtype());
        data.zip_with(&mca, &nulls)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
