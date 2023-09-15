use crate::prelude::*;

impl Series {
    /// Traverse and collect every nth element in a new array.
    pub fn take_every(&self, n: usize) -> Series {
        let idx = (0..self.len() as IdxSize)
            .step_by(n)
            .collect_ca("");
        // SAFETY: we stay in-bounds.
        unsafe { self.take_unchecked(&idx) }
    }
}
