use super::*;
use crate::prelude::*;
use arrow::array::FixedSizeBinaryArray;
use std::mem::ManuallyDrop;

pub struct PolarsExtension {
    array: Option<FixedSizeBinaryArray>,
}

impl PolarsExtension {
    pub(crate) fn new(array: FixedSizeBinaryArray) -> Self {
        Self { array: Some(array) }
    }

    pub(crate) fn take_and_forget(mut self) -> FixedSizeBinaryArray {
        let mut md = ManuallyDrop::new(self);
        md.array.take().unwrap()
    }

    fn get_sentinel(&self) -> Box<ExtensionSentinel> {
        if let ArrowDataType::Extension(_, _, Some(metadata)) =
            self.array.as_ref().unwrap().data_type()
        {
            let mut iter = metadata.split(';');

            let pid = iter.next().unwrap().parse::<u128>().unwrap();
            let ptr = iter.next().unwrap().parse::<usize>().unwrap();
            if pid == *PROCESS_ID {
                let et = unsafe {
                    Box::from_raw(ptr as *const ExtensionSentinel as *mut ExtensionSentinel)
                };
                et
            } else {
                panic!("pid did not mach process id")
            }
        } else {
            panic!("should have metadata in extension type")
        }
    }

    pub(crate) fn get_series(&self) -> Series {
        let sent = self.get_sentinel();
        let s = (sent.to_series_fn.as_ref().unwrap())(&self.array.as_ref().unwrap());
        std::mem::forget(sent);
        s
    }

    // heap allocates a function that converts the binary array to a Series of `[ObjectChunked<T>]`
    pub(crate) fn set_to_series_fn<T: PolarsObject>(&mut self, name: &str) {
        let name = name.to_string();
        let f = Box::new(move |arr: &FixedSizeBinaryArray| {
            let iter = arr.iter().map(|opt| {
                opt.map(|bytes| {
                    let t = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const T) };

                    let ret = t.clone();
                    std::mem::forget(t);
                    ret
                })
            });

            let ca = ObjectChunked::<T>::new_from_opt_iter(&name, iter);
            ca.into_series()
        });
        let mut et = self.get_sentinel();
        et.to_series_fn = Some(f);
        // forget because drop will be triggered later
        std::mem::forget(et);
    }
}

impl Drop for PolarsExtension {
    fn drop(&mut self) {
        // implicitly drop by taking ownership
        self.get_sentinel();
    }
}
