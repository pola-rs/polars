//! Special list utility methods
use crate::prelude::*;
use crate::utils::CustomIterTools;
use arrow::array::ArrayRef;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;

/// A wrapper type that should make it a bit more clear that we should not clone T
pub(crate) struct NoClone<T>(T);

impl<T> Deref for NoClone<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(crate) struct AmortizedListIter<'a, I: Iterator<Item = Option<ArrayRef>>> {
    series_container: Pin<Box<Series>>,
    inner: *mut ArrayRef,
    lifetime: PhantomData<&'a ArrayRef>,
    iter: I,
}

impl<'a, I: Iterator<Item = Option<ArrayRef>>> Iterator for AmortizedListIter<'a, I> {
    type Item = Option<NoClone<&'a Series>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|opt_val| {
            opt_val.map(|array_ref| {
                unsafe { *self.inner = array_ref };
                // Safety
                // we cannot control the lifetime of an iterators `next` method.
                // but as long as self is alive the reference to the series container is valid
                let refer = &*self.series_container;
                NoClone(unsafe { std::mem::transmute::<&Series, &'a Series>(refer) })
            })
        })
    }
}

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
    #[allow(dead_code)]
    pub(crate) fn amortized_iter(
        &self,
    ) -> AmortizedListIter<impl Iterator<Item = Option<ArrayRef>> + '_> {
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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;
    use std::mem::ManuallyDrop;
    use std::ops::DerefMut;

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
                assert!(s1.unwrap().series_equal(&s2.unwrap()));
            });
    }
}
