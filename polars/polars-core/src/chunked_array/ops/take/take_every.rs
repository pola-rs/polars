use crate::prelude::*;

impl Series {
    /// Traverse and collect every nth element in a new array.
    pub fn take_every(&self, n: usize) -> Series {
        let mut idx = (0..self.len()).step_by(n);

        // safety: we are in bounds
        unsafe { self.take_iter_unchecked(&mut idx) }
    }
}
