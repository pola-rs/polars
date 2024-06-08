use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use polars_utils::unwrap::UnwrapUncheckedRelease;

use crate::prelude::*;
use crate::series::amortized_iter::{unstable_series_container_and_ptr, AmortSeries, ArrayBox};

pub struct AmortizedListIter<'a, I: Iterator<Item = Option<ArrayBox>>> {
    len: usize,
    series_container: Rc<Series>,
    inner: NonNull<ArrayRef>,
    lifetime: PhantomData<&'a ArrayRef>,
    iter: I,
    // used only if feature="dtype-struct"
    #[allow(dead_code)]
    inner_dtype: DataType,
}

impl<'a, I: Iterator<Item = Option<ArrayBox>>> AmortizedListIter<'a, I> {
    pub(crate) unsafe fn new(
        len: usize,
        series_container: Series,
        inner: NonNull<ArrayRef>,
        iter: I,
        inner_dtype: DataType,
    ) -> Self {
        Self {
            len,
            series_container: Rc::new(series_container),
            inner,
            lifetime: PhantomData,
            iter,
            inner_dtype,
        }
    }
}

impl<'a, I: Iterator<Item = Option<ArrayBox>>> Iterator for AmortizedListIter<'a, I> {
    type Item = Option<AmortSeries>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|opt_val| {
            opt_val.map(|array_ref| {
                #[cfg(feature = "dtype-struct")]
                // structs arrays are bound to the series not to the arrayref
                // so we must get a hold to the new array
                if matches!(self.inner_dtype, DataType::Struct(_)) {
                    // SAFETY:
                    // dtype is known
                    unsafe {
                        let s = Series::from_chunks_and_dtype_unchecked(
                            "",
                            vec![array_ref],
                            &self.inner_dtype.to_physical(),
                        )
                        .cast_unchecked(&self.inner_dtype)
                        .unwrap();
                        let inner = Rc::make_mut(&mut self.series_container);
                        *inner = s;

                        return AmortSeries::new(self.series_container.clone());
                    }
                }
                // The series is cloned, we make a new container.
                if Arc::strong_count(&self.series_container.0) > 1
                    || Rc::strong_count(&self.series_container) > 1
                {
                    let (s, ptr) = unsafe {
                        unstable_series_container_and_ptr(
                            self.series_container.name(),
                            array_ref,
                            self.series_container.dtype(),
                        )
                    };
                    self.series_container = Rc::new(s);
                    self.inner = NonNull::new(ptr).unwrap();
                } else {
                    // SAFETY: we checked the RC above;
                    let series_mut = unsafe {
                        Rc::get_mut(&mut self.series_container).unwrap_unchecked_release()
                    };
                    // update the inner state
                    unsafe { *self.inner.as_mut() = array_ref };

                    // last iteration could have set the sorted flag (e.g. in compute_len)
                    series_mut.clear_flags();
                    // make sure that the length is correct
                    series_mut._get_inner_mut().compute_len();
                }

                // SAFETY:
                // inner belongs to Series.
                unsafe {
                    AmortSeries::new_with_chunk(self.series_container.clone(), self.inner.as_ref())
                }
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

// # Safety
// we correctly implemented size_hint
unsafe impl<'a, I: Iterator<Item = Option<ArrayBox>>> TrustedLen for AmortizedListIter<'a, I> {}

impl ListChunked {
    /// This is an iterator over a [`ListChunked`] that saves allocations.
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
    pub fn amortized_iter(&self) -> AmortizedListIter<impl Iterator<Item = Option<ArrayBox>> + '_> {
        self.amortized_iter_with_name("")
    }

    /// See `amortized_iter`.
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

        // SAFETY: ptr belongs the the Series..
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

    /// Apply a closure `F` elementwise.
    #[must_use]
    pub fn apply_amortized_generic<F, K, V>(&self, f: F) -> ChunkedArray<V>
    where
        V: PolarsDataType,
        F: FnMut(Option<AmortSeries>) -> Option<K> + Copy,
        V::Array: ArrayFromIter<Option<K>>,
    {
        // TODO! make an amortized iter that does not flatten
        self.amortized_iter().map(f).collect_ca(self.name())
    }

    pub fn try_apply_amortized_generic<F, K, V>(&self, f: F) -> PolarsResult<ChunkedArray<V>>
    where
        V: PolarsDataType,
        F: FnMut(Option<AmortSeries>) -> PolarsResult<Option<K>> + Copy,
        V::Array: ArrayFromIter<Option<K>>,
    {
        // TODO! make an amortized iter that does not flatten
        self.amortized_iter().map(f).try_collect_ca(self.name())
    }

    pub fn for_each_amortized<F>(&self, f: F)
    where
        F: FnMut(Option<AmortSeries>),
    {
        self.amortized_iter().for_each(f)
    }

    /// Zip with a `ChunkedArray` then apply a binary function `F` elementwise.
    #[must_use]
    pub fn zip_and_apply_amortized<'a, T, I, F>(&'a self, ca: &'a ChunkedArray<T>, mut f: F) -> Self
    where
        T: PolarsDataType,
        &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
        I: TrustedLen<Item = Option<T::Physical<'a>>>,
        F: FnMut(Option<AmortSeries>, Option<T::Physical<'a>>) -> Option<Series>,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = self.null_count() == 0;
        let mut out: ListChunked = {
            self.amortized_iter()
                .zip(ca)
                .map(|(opt_s, opt_v)| {
                    let out = f(opt_s, opt_v);
                    match out {
                        Some(out) => {
                            fast_explode &= !out.is_empty();
                            Some(out)
                        },
                        None => {
                            fast_explode = false;
                            out
                        },
                    }
                })
                .collect_trusted()
        };

        out.rename(self.name());
        if fast_explode {
            out.set_fast_explode();
        }
        out
    }

    #[must_use]
    pub fn binary_zip_and_apply_amortized<'a, T, U, F>(
        &'a self,
        ca1: &'a ChunkedArray<T>,
        ca2: &'a ChunkedArray<U>,
        mut f: F,
    ) -> Self
    where
        T: PolarsDataType,
        U: PolarsDataType,
        F: FnMut(
            Option<AmortSeries>,
            Option<T::Physical<'a>>,
            Option<U::Physical<'a>>,
        ) -> Option<Series>,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = self.null_count() == 0;
        let mut out: ListChunked = {
            self.amortized_iter()
                .zip(ca1.iter())
                .zip(ca2.iter())
                .map(|((opt_s, opt_u), opt_v)| {
                    let out = f(opt_s, opt_u, opt_v);
                    match out {
                        Some(out) => {
                            fast_explode &= !out.is_empty();
                            Some(out)
                        },
                        None => {
                            fast_explode = false;
                            out
                        },
                    }
                })
                .collect_trusted()
        };

        out.rename(self.name());
        if fast_explode {
            out.set_fast_explode();
        }
        out
    }

    pub fn try_zip_and_apply_amortized<'a, T, I, F>(
        &'a self,
        ca: &'a ChunkedArray<T>,
        mut f: F,
    ) -> PolarsResult<Self>
    where
        T: PolarsDataType,
        &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
        I: TrustedLen<Item = Option<T::Physical<'a>>>,
        F: FnMut(Option<AmortSeries>, Option<T::Physical<'a>>) -> PolarsResult<Option<Series>>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let mut fast_explode = self.null_count() == 0;
        let mut out: ListChunked = {
            self.amortized_iter()
                .zip(ca)
                .map(|(opt_s, opt_v)| {
                    let out = f(opt_s, opt_v)?;
                    match out {
                        Some(out) => {
                            fast_explode &= !out.is_empty();
                            Ok(Some(out))
                        },
                        None => {
                            fast_explode = false;
                            Ok(out)
                        },
                    }
                })
                .collect::<PolarsResult<_>>()?
        };

        out.rename(self.name());
        if fast_explode {
            out.set_fast_explode();
        }
        Ok(out)
    }

    /// Apply a closure `F` elementwise.
    #[must_use]
    pub fn apply_amortized<F>(&self, mut f: F) -> Self
    where
        F: FnMut(AmortSeries) -> Series,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = self.null_count() == 0;
        let mut ca: ListChunked = {
            self.amortized_iter()
                .map(|opt_v| {
                    opt_v.map(|v| {
                        let out = f(v);
                        if out.is_empty() {
                            fast_explode = false;
                        }
                        out
                    })
                })
                .collect_trusted()
        };

        ca.rename(self.name());
        if fast_explode {
            ca.set_fast_explode();
        }
        ca
    }

    pub fn try_apply_amortized<F>(&self, mut f: F) -> PolarsResult<Self>
    where
        F: FnMut(AmortSeries) -> PolarsResult<Series>,
    {
        if self.is_empty() {
            return Ok(self.clone());
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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_iter_list() {
        let mut builder = get_list_builder(&DataType::Int32, 10, 10, "").unwrap();
        builder.append_series(&Series::new("", &[1, 2, 3])).unwrap();
        builder.append_series(&Series::new("", &[3, 2, 1])).unwrap();
        builder.append_series(&Series::new("", &[1, 1])).unwrap();
        let ca = builder.finish();

        ca.amortized_iter().zip(&ca).for_each(|(s1, s2)| {
            assert!(s1.unwrap().as_ref().equals(&s2.unwrap()));
        })
    }
}
