use crate::prelude::*;
use crate::utils::Xob;

impl BooleanChunked {
    pub fn arg_true(&self) -> UInt32Chunked {
        let ca: Xob<_> = if self.null_count() == 0 {
            self.into_no_null_iter()
                .enumerate()
                .filter_map(|(idx, valid)| match valid {
                    true => Some(idx as u32),
                    false => None,
                })
                .collect()
        } else {
            self.into_iter()
                .enumerate()
                .filter_map(|(idx, opt_valid)| match opt_valid {
                    Some(true) => Some(idx as u32),
                    _ => None,
                })
                .collect()
        };
        ca.into_inner()
    }
}
