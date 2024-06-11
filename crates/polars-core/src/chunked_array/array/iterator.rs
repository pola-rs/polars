use std::ptr::NonNull;

use super::*;
use crate::chunked_array::list::iterator::AmortizedListIter;
use crate::series::amortized_iter::{unstable_series_container_and_ptr, AmortSeries, ArrayBox};

impl ArrayChunked {
    /// This is an iterator over a [`ArrayChunked`] that save allocations.
    /// A Series is:
    ///     1. [`Arc<ChunkedArray>`]
    ///     ChunkedArray is:
    ///         2. Vec< 3. ArrayRef>
    ///
    /// The [`ArrayRef`] we indicated with 3. will be updated during iteration.
    /// The Series will be pinned in memory, saving an allocation for
    /// 1. Arc<..>
    /// 2. Vec<...>
    ///
    /// # Warning
    /// Though memory safe in the sense that it will not read unowned memory, UB, or memory leaks
    /// this function still needs precautions. The returned should never be cloned or taken longer
    /// than a single iteration, as every call on `next` of the iterator will change the contents of
    /// that Series.
    ///
    /// # Safety
    /// The lifetime of [AmortSeries] is bound to the iterator. Keeping it alive
    /// longer than the iterator is UB.
    pub fn amortized_iter(&self) -> AmortizedListIter<impl Iterator<Item = Option<ArrayBox>> + '_> {
        self.amortized_iter_with_name("")
    }

    /// This is an iterator over a [`ArrayChunked`] that save allocations.
    /// A Series is:
    ///     1. [`Arc<ChunkedArray>`]
    ///     ChunkedArray is:
    ///         2. Vec< 3. ArrayRef>
    ///
    /// The ArrayRef we indicated with 3. will be updated during iteration.
    /// The Series will be pinned in memory, saving an allocation for
    /// 1. Arc<..>
    /// 2. Vec<...>
    ///
    /// If the returned `AmortSeries` is cloned, the local copy will be replaced and a new container
    /// will be set.
    pub fn amortized_iter_with_name(
        &self,
        name: &str,
    ) -> AmortizedListIter<impl Iterator<Item = Option<ArrayBox>> + '_> {
        // we create the series container from the inner array
        // so that the container has the proper dtype.
        let arr = self.downcast_iter().next().unwrap();
        let inner_values = arr.values();

        let inner_dtype = self.inner_dtype();
        let iter_dtype = match inner_dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => inner_dtype.to_physical(),
            // TODO: figure out how to deal with physical/logical distinction
            // physical primitives like time, date etc. work
            // physical nested need more
            _ => inner_dtype.clone(),
        };

        // SAFETY:
        // inner type passed as physical type
        let (s, ptr) =
            unsafe { unstable_series_container_and_ptr(name, inner_values.clone(), &iter_dtype) };

        // SAFETY: `ptr` belongs to the `Series`.
        unsafe {
            AmortizedListIter::new(
                self.len(),
                s,
                NonNull::new(ptr).unwrap(),
                self.downcast_iter().flat_map(|arr| arr.iter()),
                inner_dtype.clone(),
            )
        }
    }

    pub fn try_apply_amortized_to_list<F>(&self, mut f: F) -> PolarsResult<ListChunked>
    where
        F: FnMut(AmortSeries) -> PolarsResult<Series>,
    {
        if self.is_empty() {
            return Ok(Series::new_empty(
                self.name(),
                &DataType::List(Box::new(self.inner_dtype().clone())),
            )
            .list()
            .unwrap()
            .clone());
        }
        let mut fast_explode = self.null_count() == 0;
        let mut ca: ListChunked = {
            self.amortized_iter()
                .map(|opt_v| {
                    opt_v
                        .map(|v| {
                            let out = f(v);
                            if let Ok(out) = &out {
                                if out.is_empty() {
                                    fast_explode = false
                                }
                            };
                            out
                        })
                        .transpose()
                })
                .collect::<PolarsResult<_>>()?
        };
        ca.rename(self.name());
        if fast_explode {
            ca.set_fast_explode();
        }
        Ok(ca)
    }

    /// Apply a closure `F` to each array.
    ///
    /// # Safety
    /// Return series of `F` must has the same dtype and number of elements as input.
    #[must_use]
    pub unsafe fn apply_amortized_same_type<F>(&self, mut f: F) -> Self
    where
        F: FnMut(AmortSeries) -> Series,
    {
        if self.is_empty() {
            return self.clone();
        }
        self.amortized_iter()
            .map(|opt_v| {
                opt_v.map(|v| {
                    let out = f(v);
                    to_arr(&out)
                })
            })
            .collect_ca_with_dtype(self.name(), self.dtype().clone())
    }

    /// Try apply a closure `F` to each array.
    ///
    /// # Safety
    /// Return series of `F` must has the same dtype and number of elements as input if it is Ok.
    pub unsafe fn try_apply_amortized_same_type<F>(&self, mut f: F) -> PolarsResult<Self>
    where
        F: FnMut(AmortSeries) -> PolarsResult<Series>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        self.amortized_iter()
            .map(|opt_v| {
                opt_v
                    .map(|v| {
                        let out = f(v)?;
                        Ok(to_arr(&out))
                    })
                    .transpose()
            })
            .try_collect_ca_with_dtype(self.name(), self.dtype().clone())
    }

    /// Zip with a `ChunkedArray` then apply a binary function `F` elementwise.
    ///
    /// # Safety
    //  Return series of `F` must has the same dtype and number of elements as input series.
    #[must_use]
    pub unsafe fn zip_and_apply_amortized_same_type<'a, T, F>(
        &'a self,
        ca: &'a ChunkedArray<T>,
        mut f: F,
    ) -> Self
    where
        T: PolarsDataType,
        F: FnMut(Option<AmortSeries>, Option<T::Physical<'a>>) -> Option<Series>,
    {
        if self.is_empty() {
            return self.clone();
        }
        self.amortized_iter()
            .zip(ca.iter())
            .map(|(opt_s, opt_v)| {
                let out = f(opt_s, opt_v);
                out.map(|s| to_arr(&s))
            })
            .collect_ca_with_dtype(self.name(), self.dtype().clone())
    }

    /// Apply a closure `F` elementwise.
    #[must_use]
    pub fn apply_amortized_generic<F, K, V>(&self, f: F) -> ChunkedArray<V>
    where
        V: PolarsDataType,
        F: FnMut(Option<AmortSeries>) -> Option<K> + Copy,
        V::Array: ArrayFromIter<Option<K>>,
    {
        {
            self.amortized_iter().map(f).collect_ca(self.name())
        }
    }

    /// Try apply a closure `F` elementwise.
    pub fn try_apply_amortized_generic<F, K, V>(&self, f: F) -> PolarsResult<ChunkedArray<V>>
    where
        V: PolarsDataType,
        F: FnMut(Option<AmortSeries>) -> PolarsResult<Option<K>> + Copy,
        V::Array: ArrayFromIter<Option<K>>,
    {
        {
            self.amortized_iter().map(f).try_collect_ca(self.name())
        }
    }

    pub fn for_each_amortized<F>(&self, f: F)
    where
        F: FnMut(Option<AmortSeries>),
    {
        {
            self.amortized_iter().for_each(f)
        }
    }
}

fn to_arr(s: &Series) -> ArrayRef {
    if s.chunks().len() > 1 {
        let s = s.rechunk();
        s.chunks()[0].clone()
    } else {
        s.chunks()[0].clone()
    }
}
