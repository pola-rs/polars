#![allow(unsafe_op_in_unsafe_fn)]
//! Used to speed up TotalEq and TotalOrd of elements within an array.

use std::cmp::Ordering;

use polars_utils::nulls::IsNull;
use polars_utils::sort::reorder_cmp;
use polars_utils::total_ord::TotalOrdWrap;

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

impl<T: TotalOrd> TotalOrd for NonNull<T> {
    fn tot_cmp(&self, other: &Self) -> Ordering {
        self.0.tot_cmp(&other.0)
    }
}

impl<T> IsNull for NonNull<T> {
    const HAS_NULLS: bool = false;
    type Inner = T;

    fn is_null(&self) -> bool {
        false
    }
    fn unwrap_inner(self) -> Self::Inner {
        self.0
    }
}

pub trait GetInner {
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

pub trait TotalOrdInner: Send + Sync {
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn cmp_element_unchecked(
        &self,
        idx_a: usize,
        idx_b: usize,
        descending: bool,
        nulls_last: bool,
    ) -> Ordering;
}

impl<T> TotalOrdInner for T
where
    T: GetInner + Send + Sync,
    T::Item: TotalOrd + IsNull,
{
    #[inline]
    unsafe fn cmp_element_unchecked(
        &self,
        idx_a: usize,
        idx_b: usize,
        descending: bool,
        nulls_last: bool,
    ) -> Ordering {
        let a = self.get_unchecked(idx_a);
        let b = self.get_unchecked(idx_b);
        reorder_cmp(&TotalOrdWrap(a), &TotalOrdWrap(b), descending, nulls_last)
    }
}

impl TotalOrdInner for &NullChunked {
    #[inline]
    unsafe fn cmp_element_unchecked(
        &self,
        _idx_a: usize,
        _idx_b: usize,
        _descending: bool,
        _nulls_last: bool,
    ) -> Ordering {
        Ordering::Equal
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

impl<'a> IntoTotalOrdInner<'a> for &'a NullChunked {
    fn into_total_ord_inner(self) -> Box<dyn TotalOrdInner + 'a> {
        Box::new(self)
    }
}

#[cfg(feature = "dtype-categorical")]
struct LexicalCategorical<'a, T: PolarsCategoricalType> {
    mapping: &'a CategoricalMapping,
    cats: &'a ChunkedArray<T::PolarsPhysical>,
}

#[cfg(feature = "dtype-categorical")]
impl<'a, T: PolarsCategoricalType> GetInner for LexicalCategorical<'a, T> {
    type Item = Option<&'a str>;
    unsafe fn get_unchecked(&self, idx: usize) -> Self::Item {
        let cat = self.cats.get_unchecked(idx)?;
        Some(self.mapping.cat_to_str_unchecked(cat.as_cat()))
    }
}

#[cfg(feature = "dtype-categorical")]
impl<'a, T: PolarsCategoricalType> IntoTotalOrdInner<'a> for &'a CategoricalChunked<T> {
    fn into_total_ord_inner(self) -> Box<dyn TotalOrdInner + 'a> {
        if self.uses_lexical_ordering() {
            Box::new(LexicalCategorical::<T> {
                mapping: self.get_mapping(),
                cats: &self.phys,
            })
        } else {
            self.phys.into_total_ord_inner()
        }
    }
}
