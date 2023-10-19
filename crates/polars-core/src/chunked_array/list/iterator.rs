use std::marker::PhantomData;
use std::pin::Pin;
use std::ptr::NonNull;

use crate::prelude::*;
use crate::series::unstable::{ArrayBox, UnstableSeries};
use crate::utils::CustomIterTools;

pub struct AmortizedListIter<'a, I: Iterator<Item = Option<ArrayBox>>> {
    len: usize,
    series_container: Pin<Box<Series>>,
    inner: NonNull<ArrayRef>,
    lifetime: PhantomData<&'a ArrayRef>,
    iter: I,
    // used only if feature="dtype-struct"
    #[allow(dead_code)]
    inner_dtype: DataType,
}

impl<'a, I: Iterator<Item = Option<ArrayBox>>> AmortizedListIter<'a, I> {
    pub(crate) fn new(
        len: usize,
        series_container: Pin<Box<Series>>,
        inner: NonNull<ArrayRef>,
        iter: I,
        inner_dtype: DataType,
    ) -> Self {
        Self {
            len,
            series_container,
            inner,
            lifetime: PhantomData,
            iter,
            inner_dtype,
        }
    }
}

impl<'a, I: Iterator<Item = Option<ArrayBox>>> Iterator for AmortizedListIter<'a, I> {
    type Item = Option<UnstableSeries<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|opt_val| {
            opt_val.map(|array_ref| {
                #[cfg(feature = "dtype-struct")]
                // structs arrays are bound to the series not to the arrayref
                // so we must get a hold to the new array
                if matches!(self.inner_dtype, DataType::Struct(_)) {
                    // Safety
                    // dtype is known
                    unsafe {
                        let mut s = Series::from_chunks_and_dtype_unchecked(
                            "",
                            vec![array_ref],
                            &self.inner_dtype.to_physical(),
                        )
                        .cast_unchecked(&self.inner_dtype)
                        .unwrap();
                        // swap the new series with the container
                        std::mem::swap(&mut *self.series_container, &mut s);
                        // return a reference to the container
                        // this lifetime is now bound to 'a
                        return UnstableSeries::new(
                            &mut *(&mut *self.series_container as *mut Series),
                        );
                    }
                }

                // update the inner state
                unsafe { *self.inner.as_mut() = array_ref };

                // last iteration could have set the sorted flag (e.g. in compute_len)
                self.series_container.clear_settings();
                // make sure that the length is correct
                self.series_container._get_inner_mut().compute_len();

                // Safety
                // we cannot control the lifetime of an iterators `next` method.
                // but as long as self is alive the reference to the series container is valid
                let refer = &mut *self.series_container;
                unsafe {
                    let s = std::mem::transmute::<&mut Series, &'a mut Series>(refer);
                    UnstableSeries::new_with_chunk(s, self.inner.as_ref())
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
    /// This is an iterator over a [`ListChunked`] that save allocations.
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
    /// # Warning
    /// Though memory safe in the sense that it will not read unowned memory, UB, or memory leaks
    /// this function still needs precautions. The returned should never be cloned or taken longer
    /// than a single iteration, as every call on `next` of the iterator will change the contents of
    /// that Series.
    ///
    /// # Safety
    /// The lifetime of [UnstableSeries] is bound to the iterator. Keeping it alive
    /// longer than the iterator is UB.
    pub unsafe fn amortized_iter(
        &self,
    ) -> AmortizedListIter<impl Iterator<Item = Option<ArrayBox>> + '_> {
        self.amortized_iter_with_name("")
    }

    /// # Safety
    /// The lifetime of [UnstableSeries] is bound to the iterator. Keeping it alive
    /// longer than the iterator is UB.
    pub unsafe fn amortized_iter_with_name(
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

        // Safety:
        // inner type passed as physical type
        let series_container = unsafe {
            let mut s = Series::from_chunks_and_dtype_unchecked(
                name,
                vec![inner_values.clone()],
                &iter_dtype,
            );
            s.clear_settings();
            Box::pin(s)
        };

        let ptr = series_container.array_ref(0) as *const ArrayRef as *mut ArrayRef;

        AmortizedListIter::new(
            self.len(),
            series_container,
            NonNull::new(ptr).unwrap(),
            self.downcast_iter().flat_map(|arr| arr.iter()),
            inner_dtype,
        )
    }

    /// Apply a closure `F` elementwise.
    #[must_use]
    pub fn apply_amortized_generic<'a, F, K, V>(&'a self, f: F) -> ChunkedArray<V>
    where
        V: PolarsDataType,
        F: FnMut(Option<UnstableSeries<'a>>) -> Option<K> + Copy,
        V::Array: ArrayFromIter<Option<K>>,
    {
        // TODO! make an amortized iter that does not flatten
        // SAFETY: unstable series never lives longer than the iterator.
        unsafe { self.amortized_iter().map(f).collect_ca(self.name()) }
    }

    pub fn for_each_amortized<'a, F>(&'a self, f: F)
    where
        F: FnMut(Option<UnstableSeries<'a>>),
    {
        // SAFETY: unstable series never lives longer than the iterator.
        unsafe { self.amortized_iter().for_each(f) }
    }

    /// Zip with a `ChunkedArray` then apply a binary function `F` elementwise.
    #[must_use]
    pub fn zip_and_apply_amortized<'a, T, I, F>(&'a self, ca: &'a ChunkedArray<T>, mut f: F) -> Self
    where
        T: PolarsDataType,
        &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
        I: TrustedLen<Item = Option<T::Physical<'a>>>,
        F: FnMut(Option<UnstableSeries<'a>>, Option<T::Physical<'a>>) -> Option<Series>,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = self.null_count() == 0;
        // SAFETY: unstable series never lives longer than the iterator.
        let mut out: ListChunked = unsafe {
            self.amortized_iter()
                .zip(ca)
                .map(|(opt_s, opt_v)| {
                    let out = f(opt_s, opt_v);
                    match out {
                        Some(out) if out.is_empty() => {
                            fast_explode = false;
                            Some(out)
                        },
                        None => {
                            fast_explode = false;
                            out
                        },
                        _ => out,
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
        F: FnMut(
            Option<UnstableSeries<'a>>,
            Option<T::Physical<'a>>,
        ) -> PolarsResult<Option<Series>>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let mut fast_explode = self.null_count() == 0;
        // SAFETY: unstable series never lives longer than the iterator.
        let mut out: ListChunked = unsafe {
            self.amortized_iter()
                .zip(ca)
                .map(|(opt_s, opt_v)| {
                    let out = f(opt_s, opt_v)?;
                    match out {
                        Some(out) if out.is_empty() => {
                            fast_explode = false;
                            Ok(Some(out))
                        },
                        None => {
                            fast_explode = false;
                            Ok(out)
                        },
                        _ => Ok(out),
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
    pub fn apply_amortized<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(UnstableSeries<'a>) -> Series,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = self.null_count() == 0;
        // SAFETY: unstable series never lives longer than the iterator.
        let mut ca: ListChunked = unsafe {
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

    pub fn try_apply_amortized<'a, F>(&'a self, mut f: F) -> PolarsResult<Self>
    where
        F: FnMut(UnstableSeries<'a>) -> PolarsResult<Series>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let mut fast_explode = self.null_count() == 0;
        // SAFETY: unstable series never lives longer than the iterator.
        let mut ca: ListChunked = unsafe {
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

        // SAFETY: unstable series never lives longer than the iterator.
        unsafe {
            ca.amortized_iter().zip(&ca).for_each(|(s1, s2)| {
                assert!(s1.unwrap().as_ref().series_equal(&s2.unwrap()));
            })
        };
    }
}
