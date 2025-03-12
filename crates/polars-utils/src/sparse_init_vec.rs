use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

pub struct SparseInitVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,

    num_init: AtomicUsize,
    init_mask: Vec<AtomicU8>,
}

unsafe impl<T: Send> Send for SparseInitVec<T> {}
unsafe impl<T: Send> Sync for SparseInitVec<T> {}

impl<T> SparseInitVec<T> {
    pub fn with_capacity(len: usize) -> Self {
        let init_mask = (0..len.div_ceil(8)).map(|_| AtomicU8::new(0)).collect();
        let mut storage = Vec::with_capacity(len);
        let cap = storage.capacity();
        let ptr = storage.as_mut_ptr();
        core::mem::forget(storage);
        Self {
            len,
            cap,
            ptr,
            num_init: AtomicUsize::new(0),
            init_mask,
        }
    }

    pub fn try_set(&self, idx: usize, value: T) -> Result<(), T> {
        unsafe {
            if idx >= self.len {
                return Err(value);
            }

            // SAFETY: we use Relaxed orderings as we only ever read data back in methods that take
            // self mutably or owned, already implying synchronization.
            let init_mask_byte = self.init_mask.get_unchecked(idx / 8);
            let bit_mask = 1 << (idx % 8);
            if init_mask_byte.fetch_or(bit_mask, Ordering::Relaxed) & bit_mask != 0 {
                return Err(value);
            }

            self.ptr.add(idx).write(value);
            self.num_init.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    pub fn try_assume_init(mut self) -> Result<Vec<T>, Self> {
        unsafe {
            if *self.num_init.get_mut() == self.len {
                let ret = Vec::from_raw_parts(self.ptr, self.len, self.cap);
                drop(core::mem::take(&mut self.init_mask));
                core::mem::forget(self);
                Ok(ret)
            } else {
                Err(self)
            }
        }
    }
}

impl<T> Drop for SparseInitVec<T> {
    fn drop(&mut self) {
        unsafe {
            // Make sure storage gets dropped even if element drop panics.
            let _storage = Vec::from_raw_parts(self.ptr, 0, self.cap);

            for idx in 0..self.len {
                let init_mask_byte = self.init_mask.get_unchecked_mut(idx / 8);
                let bit_mask = 1 << (idx % 8);
                if *init_mask_byte.get_mut() & bit_mask != 0 {
                    self.ptr.add(idx).drop_in_place();
                }
            }
        }
    }
}
