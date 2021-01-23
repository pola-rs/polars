use crate::prelude::*;
use crate::utils::NoNull;

impl BooleanChunked {
    pub fn arg_true(&self) -> UInt32Chunked {
        // the allocation is probably cheaper as the filter is super fast
        let ca: NoNull<UInt32Chunked> = (0u32..self.len() as u32).collect();
        ca.into_inner().filter(self).unwrap()
    }
}
