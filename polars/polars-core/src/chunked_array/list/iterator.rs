use crate::prelude::*;
use crate::series::unstable::{ArrayBox, UnstableSeries};
use crate::utils::CustomIterTools;
use arrow::array::ArrayRef;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::pin::Pin;
use std::ptr::NonNull;

#[cfg(feature = "private")]
pub struct AmortizedListIter<'a, I: Iterator<Item = Option<ArrayBox>>> {
    series_container: Pin<Box<Series>>,
    inner: NonNull<ArrayRef>,
    lifetime: PhantomData<&'a ArrayRef>,
    iter: I,
}

impl<'a, I: Iterator<Item = Option<ArrayBox>>> Iterator for AmortizedListIter<'a, I> {
    type Item = Option<UnstableSeries<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|opt_val| {
            opt_val.map(|array_ref| {
                unsafe { *self.inner.as_mut() = array_ref.into() };
                // Safety
                // we cannot control the lifetime of an iterators `next` method.
                // but as long as self is alive the reference to the series container is valid
                let refer = &*self.series_container;
                unsafe {
                    let s = std::mem::transmute::<&Series, &'a Series>(refer);
                    UnstableSeries::new_with_chunk(s, self.inner.as_ref())
                }
            })
        })
    }
}

#[cfg(feature = "private")]
unsafe impl<'a, I: Iterator<Item = Option<ArrayBox>>> TrustedLen for AmortizedListIter<'a, I> {}

impl ListChunked {
    /// This is an iterator over a ListChunked that save allocations.
    /// A Series is:
    ///     1. Arc<ChunkedArray>
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
    #[cfg(feature = "private")]
    pub fn amortized_iter(&self) -> AmortizedListIter<impl Iterator<Item = Option<ArrayBox>> + '_> {
        // we create the series container from the inner array
        // so that the container has the proper dtype.
        let arr = self.downcast_iter().next().unwrap();
        let inner_values = arr.values();
        let series_container = Box::pin(Series::try_from(("", inner_values.clone())).unwrap());

        let ptr = &series_container.chunks()[0] as *const ArrayRef as *mut ArrayRef;

        AmortizedListIter {
            series_container,
            inner: NonNull::new(ptr).unwrap(),
            lifetime: PhantomData,
            iter: self
                .downcast_iter()
                .map(|arr| arr.iter())
                .flatten()
                .trust_my_length(self.len()),
        }
    }

    /// Apply a closure `F` elementwise.
    #[cfg(feature = "private")]
    pub fn apply_amortized<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(UnstableSeries<'a>) -> Series,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = true;
        let mut ca: ListChunked = self
            .amortized_iter()
            .map(|opt_v| {
                opt_v.map(|v| {
                    let out = f(v);
                    if out.is_empty() {
                        fast_explode = false;
                    }
                    out
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        if fast_explode {
            ca.set_fast_explode();
        }
        ca
    }

    pub fn try_apply_amortized<'a, F>(&'a self, mut f: F) -> Result<Self>
    where
        F: FnMut(UnstableSeries<'a>) -> Result<Series>,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let mut fast_explode = true;
        let mut ca: ListChunked = self
            .amortized_iter()
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
            .collect::<Result<_>>()?;
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
        let mut builder = get_list_builder(&DataType::Int32, 10, 10, "");
        builder.append_series(&Series::new("", &[1, 2, 3]));
        builder.append_series(&Series::new("", &[3, 2, 1]));
        builder.append_series(&Series::new("", &[1, 1]));
        let ca = builder.finish();

        ca.amortized_iter()
            .zip(ca.into_iter())
            .for_each(|(s1, s2)| {
                assert!(s1.unwrap().as_ref().series_equal(&s2.unwrap()));
            });
    }
}
