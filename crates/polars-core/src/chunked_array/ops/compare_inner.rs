//! Used to speed up TotalEq and TotalOrd of elements within an array.

use std::cmp::Ordering;

use crate::chunked_array::ChunkedArrayLayout;
use crate::prelude::*;
use crate::series::implementations::null::NullChunked;

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NonNull<T>(pub T);

impl<T: TotalEq> TotalEq for NonNull<T> {
    fn tot_eq(&self, other: &Self) -> bool {
        self.0.tot_eq(&other.0)
    }
}

pub trait NullOrderCmp {
    fn null_order_cmp(&self, other: &Self, nulls_last: bool) -> Ordering;
}

impl<T: TotalOrd> NullOrderCmp for Option<T> {
    fn null_order_cmp(&self, other: &Self, nulls_last: bool) -> Ordering {
        match (self, other) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => {
                if nulls_last {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            },
            (Some(_), None) => {
                if nulls_last {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            },
            (Some(l), Some(r)) => l.tot_cmp(r),
        }
    }
}

impl<T: TotalOrd> NullOrderCmp for NonNull<T> {
    fn null_order_cmp(&self, other: &Self, _nulls_last: bool) -> Ordering {
        self.0.tot_cmp(&other.0)
    }
}

trait GetInner {
    type Item;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item;
}

impl<'a, T: PolarsDataType> GetInner for &'a ChunkedArray<T> {
    type Item = Option<T::Physical<'a>>;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        ChunkedArray::get_unchecked(self, idx)
    }
}

impl<'a, T: StaticArray> GetInner for &'a T {
    type Item = Option<T::ValueT<'a>>;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        <T as StaticArray>::get_unchecked(self, idx)
    }
}

impl<'a, T: PolarsDataType> GetInner for NonNull<&'a ChunkedArray<T>> {
    type Item = NonNull<T::Physical<'a>>;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        NonNull(self.0.value_unchecked(idx))
    }
}

impl<'a, T: StaticArray> GetInner for NonNull<&'a T> {
    type Item = NonNull<T::ValueT<'a>>;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        NonNull(self.0.value_unchecked(idx))
    }
}

pub trait TotalEqInner: Send + Sync {
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool;
}

pub trait TotalOrdInner: Send + Sync {
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn cmp_element_unchecked(
        &self,
        idx_a: usize,
        idx_b: usize,
        nulls_last: bool,
    ) -> Ordering;
}

impl<T> TotalEqInner for T
where
    T: GetInner + Send + Sync,
    T::Item: TotalEq,
{
    #[inline]
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
        self.get_unchecked(idx_a).tot_eq(&self.get_unchecked(idx_b))
    }
}

impl TotalEqInner for &NullChunked {
    unsafe fn eq_element_unchecked(&self, _idx_a: usize, _idx_b: usize) -> bool {
        true
    }
}

/// Create a type that implements TotalEqInner.
pub(crate) trait IntoTotalEqInner<'a> {
    /// Create a type that implements `TakeRandom`.
    fn into_total_eq_inner(self) -> Box<dyn TotalEqInner + 'a>;
}

impl<'a> IntoTotalEqInner<'a> for &'a NullChunked {
    fn into_total_eq_inner(self) -> Box<dyn TotalEqInner + 'a> {
        Box::new(self)
    }
}

/// We use a trait object because we want to call this from Series and cannot use a typed enum.
impl<'a, T> IntoTotalEqInner<'a> for &'a ChunkedArray<T>
where
    T: PolarsDataType,
    T::Physical<'a>: TotalEq,
{
    fn into_total_eq_inner(self) -> Box<dyn TotalEqInner + 'a> {
        match self.layout() {
            ChunkedArrayLayout::SingleNoNull(arr) => Box::new(NonNull(arr)),
            ChunkedArrayLayout::Single(arr) => Box::new(arr),
            ChunkedArrayLayout::MultiNoNull(ca) => Box::new(NonNull(ca)),
            ChunkedArrayLayout::Multi(ca) => Box::new(ca),
        }
    }
}

impl<T> TotalOrdInner for T
where
    T: GetInner + Send + Sync,
    T::Item: NullOrderCmp,
{
    #[inline]
    unsafe fn cmp_element_unchecked(
        &self,
        idx_a: usize,
        idx_b: usize,
        nulls_last: bool,
    ) -> Ordering {
        let a = self.get_unchecked(idx_a);
        let b = self.get_unchecked(idx_b);
        a.null_order_cmp(&b, nulls_last)
    }
}

/// Create a type that implements TotalOrdInner.
pub(crate) trait IntoTotalOrdInner<'a> {
    /// Create a type that implements `TakeRandom`.
    fn into_total_ord_inner(self) -> Box<dyn TotalOrdInner + 'a>;
}

impl<'a, T> IntoTotalOrdInner<'a> for &'a ChunkedArray<T>
where
    T: PolarsDataType,
    T::Physical<'a>: TotalOrd,
{
    fn into_total_ord_inner(self) -> Box<dyn TotalOrdInner + 'a> {
        match self.layout() {
            ChunkedArrayLayout::SingleNoNull(arr) => Box::new(NonNull(arr)),
            ChunkedArrayLayout::Single(arr) => Box::new(arr),
            ChunkedArrayLayout::MultiNoNull(ca) => Box::new(NonNull(ca)),
            ChunkedArrayLayout::Multi(ca) => Box::new(ca),
        }
    }
}

#[cfg(feature = "dtype-categorical")]
struct LocalCategorical<'a> {
    rev_map: &'a Utf8ViewArray,
    cats: &'a UInt32Chunked,
}

#[cfg(feature = "dtype-categorical")]
impl<'a> GetInner for LocalCategorical<'a> {
    type Item = Option<&'a str>;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        let cat = self.cats.get_unchecked(idx)?;
        Some(self.rev_map.value_unchecked(cat as usize))
    }
}

#[cfg(feature = "dtype-categorical")]
struct GlobalCategorical<'a> {
    p1: &'a PlHashMap<u32, u32>,
    p2: &'a Utf8ViewArray,
    cats: &'a UInt32Chunked,
}

#[cfg(feature = "dtype-categorical")]
impl<'a> GetInner for GlobalCategorical<'a> {
    type Item = Option<&'a str>;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        let cat = self.cats.get_unchecked(idx)?;
        let idx = self.p1.get(&cat).unwrap();
        Some(self.p2.value_unchecked(*idx as usize))
    }
}

#[cfg(feature = "dtype-categorical")]
impl<'a> IntoTotalOrdInner<'a> for &'a CategoricalChunked {
    fn into_total_ord_inner(self) -> Box<dyn TotalOrdInner + 'a> {
        let cats = self.physical();
        match &**self.get_rev_map() {
            RevMapping::Global(p1, p2, _) => Box::new(GlobalCategorical { p1, p2, cats }),
            RevMapping::Local(rev_map, _) => Box::new(LocalCategorical { rev_map, cats }),
        }
    }
}
