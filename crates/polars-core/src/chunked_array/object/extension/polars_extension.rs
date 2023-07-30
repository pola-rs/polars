use std::mem::ManuallyDrop;

use arrow::array::FixedSizeBinaryArray;

use super::*;
use crate::prelude::*;

pub struct PolarsExtension {
    array: Option<FixedSizeBinaryArray>,
}

impl PolarsExtension {
    /// This is very expensive
    pub(crate) unsafe fn arr_to_av(arr: &FixedSizeBinaryArray, i: usize) -> AnyValue {
        let arr = arr.slice_typed_unchecked(i, 1);
        let pe = Self::new(arr);
        let pe = ManuallyDrop::new(pe);
        pe.get_series("").get(0).unwrap().into_static().unwrap()
    }

    pub(crate) unsafe fn new(array: FixedSizeBinaryArray) -> Self {
        Self { array: Some(array) }
    }

    /// Take the Array hold by `[PolarsExtension]` and forget polars extension,
    /// so that drop is not called
    pub(crate) fn take_and_forget(self) -> FixedSizeBinaryArray {
        let mut md = ManuallyDrop::new(self);
        md.array.take().unwrap()
    }

    /// Apply a function with the sentinel, without the sentinels drop being called
    unsafe fn with_sentinel<T, F: FnOnce(&mut ExtensionSentinel) -> T>(&self, fun: F) -> T {
        let mut sentinel = self.get_sentinel();
        let out = fun(&mut sentinel);
        std::mem::forget(sentinel);
        out
    }

    /// Load the sentinel from the heap.
    /// be very careful, this dereference a raw pointer on the heap,
    unsafe fn get_sentinel(&self) -> Box<ExtensionSentinel> {
        if let ArrowDataType::Extension(_, _, Some(metadata)) =
            self.array.as_ref().unwrap().data_type()
        {
            let mut iter = metadata.split(';');

            let pid = iter.next().unwrap().parse::<u128>().unwrap();
            let ptr = iter.next().unwrap().parse::<usize>().unwrap();
            if pid == *PROCESS_ID {
                Box::from_raw(ptr as *const ExtensionSentinel as *mut ExtensionSentinel)
            } else {
                panic!("pid did not mach process id")
            }
        } else {
            panic!("should have metadata in extension type")
        }
    }

    /// Calls the heap allocated function in the `[ExtensionSentinel]` that knows
    /// how to convert the `[FixedSizeBinaryArray]` to a `Series` of type `[ObjectChunked<T>]`
    pub(crate) unsafe fn get_series(&self, name: &str) -> Series {
        self.with_sentinel(|sent| {
            (sent.to_series_fn.as_ref().unwrap())(self.array.as_ref().unwrap(), name)
        })
    }

    // heap allocates a function that converts the binary array to a Series of `[ObjectChunked<T>]`
    // the `name` will be the `name` of the output `Series` when this function is called (later).
    pub(crate) unsafe fn set_to_series_fn<T: PolarsObject>(&mut self) {
        let f = Box::new(move |arr: &FixedSizeBinaryArray, name: &str| {
            let iter = arr.iter().map(|opt| {
                opt.map(|bytes| {
                    let t = std::ptr::read_unaligned(bytes.as_ptr() as *const T);

                    let ret = t.clone();
                    std::mem::forget(t);
                    ret
                })
            });

            let ca = ObjectChunked::<T>::from_iter_options(name, iter);
            ca.into_series()
        });
        self.with_sentinel(move |sent| {
            sent.to_series_fn = Some(f);
        });
    }
}

impl Drop for PolarsExtension {
    fn drop(&mut self) {
        // implicitly drop by taking ownership
        unsafe { self.get_sentinel() };
    }
}
