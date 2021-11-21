use arrow::array::FixedSizeBinaryArray;
use std::mem::ManuallyDrop;
use crate::prelude::*;
use super::*;

pub struct PolarsExtension {
    array: Option<FixedSizeBinaryArray>,
}

impl PolarsExtension {
    pub(crate) fn new(array: FixedSizeBinaryArray) -> Self {
        Self {
            array: Some(array)
        }

    }
    pub(crate) fn take_and_forget(mut self) -> FixedSizeBinaryArray {
        let mut md = ManuallyDrop::new(self);
        md.array.take().unwrap()
    }

    pub(crate) fn set_to_series_fn(mut self) {
        
        selfr

    }
}

impl Drop for PolarsExtension {
    fn drop(&mut self) {
        if let ArrowDataType::Extension(_, _, metadata) = self.array.as_ref().unwrap().data_type() {
            let metadata = metadata.as_ref().expect("should have metadata");
            let mut iter = metadata.split(';');

            let pid = iter.next().unwrap().parse::<u128>().unwrap();
            let ptr = iter.next().unwrap().parse::<usize>().unwrap();
            if pid == *PROCESS_ID {
                // implicitly drop by taking ownership
                let _et = unsafe {
                    Box::from_raw(ptr as *const ExtensionSentinel as *mut ExtensionSentinel)
                };
            }
        }
    }
}

