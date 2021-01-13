use crate::prelude::*;
use crate::utils::Xob;

impl BooleanChunked {
    pub fn arg_true(&self) -> UInt32Chunked {
        // the allocation is probably cheaper as the filter is super fast
        let ca: Xob<UInt32Chunked> = (0u32..self.len() as u32).collect();
        ca.into_inner().filter(self).unwrap()
    }
}
