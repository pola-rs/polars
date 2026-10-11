#![allow(unsafe_op_in_unsafe_fn)]
use std::borrow::Cow;
use std::marker::PhantomData;

use num_traits::Bounded;
use polars_core::chunked_array::arg_min_max::{
    arg_max_binary, arg_max_binary_offset, arg_max_bool, arg_max_numeric, arg_min_binary,
    arg_min_binary_offset, arg_min_bool, arg_min_numeric,
};
use polars_core::with_match_physical_integer_polars_type;
use polars_plan::plans::MinMaxByOrder;
use polars_utils::arg_min_max::ArgMinMax;
use polars_utils::float::IsFloat;
use polars_utils::min_max::MinMax;

use super::*;
use crate::expressions::first_nan;
use crate::reduce::first_last::new_last_reduction;

pub fn new_min_by_reduction(
    dtype: DataType,
    by_dtype: DataType,
    order: MinMaxByOrder,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use DataType::*;
    use SelectPayloadGroupedReduction as SPGR;
    let null_keys = NullKeys::new(order, false);
    let payload = new_last_reduction(dtype.clone());
    Ok(match &by_dtype {
        Boolean => Box::new(SPGR::new(by_dtype, BooleanMinSelector, payload, null_keys)),
        #[cfg(all(feature = "dtype-f16", feature = "propagate_nans"))]
        #[cfg(feature = "dtype-f16")]
        Float16 => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Float16Type>(PhantomData),
            payload,
            null_keys,
        )),
        Float32 => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Float32Type>(PhantomData),
            payload,
            null_keys,
        )),
        Float64 => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Float64Type>(PhantomData),
            payload,
            null_keys,
        )),
        Null => Box::new(NullGroupedReduction::new(Scalar::null(dtype))),
        String | Binary => Box::new(SPGR::new(by_dtype, BinaryMinSelector, payload, null_keys)),
        BinaryOffset => Box::new(SPGR::new(
            by_dtype,
            BinaryOffsetMinSelector,
            payload,
            null_keys,
        )),
        _ if by_dtype.is_integer() || by_dtype.is_temporal() || by_dtype.is_enum() => {
            with_match_physical_integer_polars_type!(by_dtype.to_physical(), |$T| {
                Box::new(SPGR::new(by_dtype, MinSelector::<$T>(PhantomData), payload, null_keys))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(SPGR::new(
            by_dtype,
            MinSelector::<Int128Type>(PhantomData),
            payload,
            null_keys,
        )),
        #[cfg(feature = "dtype-categorical")]
        Categorical(cats, map) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            let map = map.clone();
            Box::new(SPGR::new(by_dtype, CatMinSelector::<$C>(map, PhantomData), payload, null_keys))
        }),
        _ => {
            polars_bail!(InvalidOperation: "`min_by` operation not supported for by dtype `{by_dtype}`")
        },
    })
}

pub fn new_max_by_reduction(
    dtype: DataType,
    by_dtype: DataType,
    order: MinMaxByOrder,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use DataType::*;
    use SelectPayloadGroupedReduction as SPGR;
    let null_keys = NullKeys::new(order, true);
    let payload = new_last_reduction(dtype.clone());
    Ok(match &by_dtype {
        Boolean => Box::new(SPGR::new(by_dtype, BooleanMaxSelector, payload, null_keys)),
        #[cfg(all(feature = "dtype-f16", feature = "propagate_nans"))]
        #[cfg(feature = "dtype-f16")]
        Float16 => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Float16Type>(PhantomData),
            payload,
            null_keys,
        )),
        Float32 if null_keys != NullKeys::Skip => Box::new(SPGR::new(
            by_dtype,
            NanMaxSelector::<Float32Type>(PhantomData),
            payload,
            null_keys,
        )),
        Float64 if null_keys != NullKeys::Skip => Box::new(SPGR::new(
            by_dtype,
            NanMaxSelector::<Float64Type>(PhantomData),
            payload,
            null_keys,
        )),
        Float32 => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Float32Type>(PhantomData),
            payload,
            null_keys,
        )),
        Float64 => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Float64Type>(PhantomData),
            payload,
            null_keys,
        )),
        Null => Box::new(NullGroupedReduction::new(Scalar::null(dtype))),
        String | Binary => Box::new(SPGR::new(by_dtype, BinaryMaxSelector, payload, null_keys)),
        BinaryOffset => Box::new(SPGR::new(
            by_dtype,
            BinaryOffsetMaxSelector,
            payload,
            null_keys,
        )),
        _ if by_dtype.is_integer() || by_dtype.is_temporal() || by_dtype.is_enum() => {
            with_match_physical_integer_polars_type!(by_dtype.to_physical(), |$T| {
                Box::new(SPGR::new(by_dtype, MaxSelector::<$T>(PhantomData), payload, null_keys))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(SPGR::new(
            by_dtype,
            MaxSelector::<Int128Type>(PhantomData),
            payload,
            null_keys,
        )),
        #[cfg(feature = "dtype-categorical")]
        Categorical(cats, map) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            let map = map.clone();
            Box::new(SPGR::new(by_dtype, CatMaxSelector::<$C>(map, PhantomData), payload, null_keys))
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

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize>;

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

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_min_numeric(ca)
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

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_max_numeric(ca)
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

/// A max selector for floats where NaN is the largest value.
struct NanMaxSelector<T>(PhantomData<T>);

impl<T> Clone for NanMaxSelector<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T> SelectReducer for NanMaxSelector<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    type Value = T::Native;
    type Dtype = T;

    fn init(&self) -> Self::Value {
        T::Native::neg_inf_value()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        first_nan(ca).or_else(|| arg_max_numeric(ca))
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        let better = b.nan_max_gt(a);
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

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_min_binary(ca)
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

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_max_binary(ca)
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
struct BinaryOffsetMinSelector;
#[derive(Clone)]
struct BinaryOffsetMaxSelector;

impl SelectReducer for BinaryOffsetMinSelector {
    type Dtype = BinaryOffsetType;
    type Value = Option<Vec<u8>>;

    fn init(&self) -> Self::Value {
        // There's no "maximum string" initializer.
        None
    }

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_min_binary_offset(ca)
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

impl SelectReducer for BinaryOffsetMaxSelector {
    type Dtype = BinaryOffsetType;
    type Value = Vec<u8>;

    fn init(&self) -> Self::Value {
        // Empty string is <= any other string, so can initialize max with it.
        Vec::new()
    }

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_max_binary_offset(ca)
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
    type Value = Option<bool>;
    type Dtype = BooleanType;

    fn init(&self) -> Self::Value {
        None
    }

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_min_bool(ca)
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        #[allow(clippy::bool_comparison)]
        let better = a.is_none_or(|a| b <= a);
        if better {
            *a = Some(b);
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, b.unwrap_or(true))
    }
}

impl SelectReducer for BooleanMaxSelector {
    type Value = Option<bool>;
    type Dtype = BooleanType;

    fn init(&self) -> Self::Value {
        None
    }

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        arg_max_bool(ca)
    }

    fn select_one(
        &self,
        a: &mut Self::Value,
        b: <Self::Dtype as PolarsDataType>::Physical<'_>,
    ) -> bool {
        #[allow(clippy::bool_comparison)]
        let better = a.is_none_or(|a| b > a);
        if better {
            *a = Some(b);
        }
        better
    }

    fn select_combine(&self, a: &mut Self::Value, b: &Self::Value) -> bool {
        self.select_one(a, b.unwrap_or(false))
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

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        use polars_core::chunked_array::arg_min_max::arg_min_opt_iter;
        arg_min_opt_iter(ca.iter().map(|cat| self.0.cat_to_str(cat?.as_cat())))
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

    fn arg_ca(&self, ca: &ChunkedArray<Self::Dtype>) -> Option<usize> {
        use polars_core::chunked_array::arg_min_max::arg_max_opt_iter;
        arg_max_opt_iter(ca.iter().map(|cat| self.0.cat_to_str(cat?.as_cat())))
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

/// What a null key does in a `min_by` or a `max_by`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NullKeys {
    /// It is skipped.
    Skip,
    /// It is picked over any other key.
    Win,
    /// It is picked only if the group has no other key.
    Lose,
}

impl NullKeys {
    fn new(order: MinMaxByOrder, is_max: bool) -> Self {
        match order {
            MinMaxByOrder::SkipNulls => Self::Skip,
            MinMaxByOrder::Total { nulls_largest } if nulls_largest == is_max => Self::Win,
            MinMaxByOrder::Total { .. } => Self::Lose,
        }
    }
}

struct SelectPayloadGroupedReduction<R: SelectReducer> {
    values: Vec<R::Value>,
    mask: MutableBitmap,
    /// Whether the pick of a group is a null key. Its value is then unused.
    null_picks: MutableBitmap,
    evicted_values: Vec<R::Value>,
    evicted_mask: BitmapBuilder,
    evicted_null_picks: BitmapBuilder,
    in_dtype: DataType,
    reducer: R,
    null_keys: NullKeys,
    payload: Box<dyn GroupedReduction>,

    tmp_subset: Vec<IdxSize>,
    tmp_group_idxs: Vec<IdxSize>,
}

impl<R: SelectReducer> SelectPayloadGroupedReduction<R> {
    fn new(
        in_dtype: DataType,
        reducer: R,
        payload: Box<dyn GroupedReduction>,
        null_keys: NullKeys,
    ) -> Self {
        Self {
            values: Vec::new(),
            mask: MutableBitmap::new(),
            null_picks: MutableBitmap::new(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            evicted_null_picks: BitmapBuilder::new(),
            in_dtype,
            reducer,
            null_keys,
            payload,
            tmp_subset: Vec::new(),
            tmp_group_idxs: Vec::new(),
        }
    }

    /// Picks a null key for the group if it comes before the current pick.
    ///
    /// # Safety
    /// `group_idx` is in bounds.
    unsafe fn select_null(&mut self, group_idx: usize) -> bool {
        let better = !self.mask.get_unchecked(group_idx)
            || (self.null_keys == NullKeys::Win && !self.null_picks.get_unchecked(group_idx));
        if better {
            self.mask.set_unchecked(group_idx, true);
            self.null_picks.set_unchecked(group_idx, true);
        }
        better
    }

    /// Picks a non-null key for the group with `select` if it comes before the current pick.
    ///
    /// # Safety
    /// `group_idx` is in bounds.
    unsafe fn select_value(
        &mut self,
        group_idx: usize,
        select: impl FnOnce(&R, &mut R::Value) -> bool,
    ) -> bool {
        let grp = self.values.get_unchecked_mut(group_idx);
        if !self.mask.get_unchecked(group_idx) {
            self.mask.set_unchecked(group_idx, true);
            select(&self.reducer, grp);
            true
        } else if self.null_picks.get_unchecked(group_idx) {
            if self.null_keys == NullKeys::Win {
                return false;
            }
            // A null key only loses its pick when it was the first key, so the value is `init`.
            self.null_picks.set_unchecked(group_idx, false);
            select(&self.reducer, grp);
            true
        } else {
            select(&self.reducer, grp)
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
            self.null_keys,
        ))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.mask.reserve(additional);
        self.null_picks.reserve(additional);
        self.payload.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, self.reducer.init());
        self.mask.resize(num_groups as usize, false);
        self.null_picks.resize(num_groups as usize, false);
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
        assert!((group_idx as usize) < self.values.len());
        if self.null_keys == NullKeys::Win && self.null_picks.get(group_idx as usize) {
            return Ok(());
        }

        let ord_values = self
            .reducer
            .cast_series(ord_values.as_materialized_series());
        let ca: &ChunkedArray<R::Dtype> = ord_values.as_ref().as_ref().as_ref();

        let first_null = || ca.first_null().map(|i| (i, true));
        let arg = || self.reducer.arg_ca(ca).map(|i| (i, false));
        let selected = match self.null_keys {
            NullKeys::Skip => arg(),
            NullKeys::Win => first_null().or_else(arg),
            NullKeys::Lose => arg().or_else(first_null),
        };

        if let Some((selected, is_null)) = selected {
            // SAFETY: `group_idx` is in bounds, checked above.
            let better = unsafe {
                if is_null {
                    self.select_null(group_idx as usize)
                } else {
                    let val = ca.value_unchecked(selected);
                    self.select_value(group_idx as usize, |r, grp| r.select_one(grp, val))
                }
            };
            if better {
                // A scalar payload, such as a literal, arrives as a unit-length column.
                let payload_idx = if payload_values.len() == 1 {
                    0
                } else {
                    selected
                };
                let selected_val = payload_values.new_from_index(payload_idx, 1);
                self.payload.update_group(&[&selected_val], group_idx, 0)?;
            }
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
                let evict = g.should_evict();
                if evict {
                    let grp = self.values.get_unchecked_mut(g.idx());
                    self.evicted_values
                        .push(core::mem::replace(grp, self.reducer.init()));
                    self.evicted_mask.push(self.mask.get_unchecked(g.idx()));
                    self.evicted_null_picks
                        .push(self.null_picks.get_unchecked(g.idx()));
                    self.mask.set_unchecked(g.idx(), false);
                    self.null_picks.set_unchecked(g.idx(), false);
                }
                let better = match arr.get_unchecked(*i as usize) {
                    Some(v) => self.select_value(g.idx(), |r, grp| r.select_one(grp, v)),
                    None if self.null_keys == NullKeys::Skip => false,
                    None => self.select_null(g.idx()),
                };
                // The payload must evict the same groups.
                if better || evict {
                    self.tmp_subset.push(*i);
                    self.tmp_group_idxs.push(g.0);
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
                if !other.mask.get_unchecked(*i as usize) {
                    continue;
                }
                let better = if other.null_picks.get_unchecked(*i as usize) {
                    self.select_null(*g as usize)
                } else {
                    let v = other.values.get_unchecked(*i as usize);
                    self.select_value(*g as usize, |r, grp| r.select_combine(grp, v))
                };
                if better {
                    self.tmp_subset.push(*i);
                    self.tmp_group_idxs.push(*g);
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
            null_picks: core::mem::take(&mut self.evicted_null_picks).into_mut(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            evicted_null_picks: BitmapBuilder::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
            null_keys: self.null_keys,
            payload: self.payload.take_evictions(),
            tmp_group_idxs: Vec::new(),
            tmp_subset: Vec::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let mask = core::mem::take(&mut self.mask);
        drop(core::mem::take(&mut self.values));
        drop(core::mem::take(&mut self.null_picks));
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
