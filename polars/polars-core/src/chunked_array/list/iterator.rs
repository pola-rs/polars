use crate::prelude::*;
use crate::utils::CustomIterTools;
use arrow::array::ArrayRef;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::pin::Pin;

/// A wrapper type that should make it a bit more clear that we should not clone Series
#[derive(Debug)]
#[cfg(feature = "private")]
pub struct UnsafeSeries<'a>(&'a Series);

/// We don't implement Deref so that the caller is aware of converting to Series
impl AsRef<Series> for UnsafeSeries<'_> {
    fn as_ref(&self) -> &Series {
        self.0
    }
}

type ArrayBox = Box<dyn Array>;

impl UnsafeSeries<'_> {
    pub fn clone(&self) {
        panic!("don't clone this type, use deep_clone")
    }

    pub fn deep_clone(&self) -> Series {
        let array_ref = self.0.chunks()[0].clone();
        Series::try_from((self.0.name(), array_ref)).unwrap()
    }
}

#[cfg(feature = "private")]
pub struct AmortizedListIter<'a, I: Iterator<Item = Option<ArrayBox>>> {
    series_container: Pin<Box<Series>>,
    inner: *mut ArrayRef,
    lifetime: PhantomData<&'a ArrayRef>,
    iter: I,
}

impl<'a, I: Iterator<Item = Option<ArrayBox>>> Iterator for AmortizedListIter<'a, I> {
    type Item = Option<UnsafeSeries<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|opt_val| {
            opt_val.map(|array_ref| {
                unsafe { *self.inner = array_ref.into() };
                // Safety
                // we cannot control the lifetime of an iterators `next` method.
                // but as long as self is alive the reference to the series container is valid
                let refer = &*self.series_container;
                UnsafeSeries(unsafe { std::mem::transmute::<&Series, &'a Series>(refer) })
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
        let series_container = if self.is_empty() {
            // in case of no data, the actual Series does not matter
            Box::pin(Series::new("", &[true]))
        } else {
            Box::pin(self.get(0).unwrap())
        };

        let ptr = &series_container.chunks()[0] as *const ArrayRef as *mut ArrayRef;

        AmortizedListIter {
            series_container,
            inner: ptr,
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
    pub fn apply_amortized<'a, F>(&'a self, f: F) -> Self
    where
        F: Fn(UnsafeSeries<'a>) -> Series + Copy,
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

        if fast_explode {
            ca.set_fast_explode();
        }
        ca
    }

    pub fn try_apply_amortized<'a, F>(&'a self, f: F) -> Result<Self>
    where
        F: Fn(UnsafeSeries<'a>) -> Result<Series> + Copy,
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
