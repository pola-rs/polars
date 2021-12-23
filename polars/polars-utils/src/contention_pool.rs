use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct LowContentionPool<T> {
    stack: Vec<Mutex<T>>,
    size: AtomicUsize,
}

impl<T: Default> LowContentionPool<T> {
    pub fn new(size: usize) -> Self {
        let mut stack = Vec::with_capacity(size);
        for _ in 0..size {
            stack.push(Mutex::new(T::default()))
        }
        let size = AtomicUsize::new(size);

        Self { stack, size }
    }

    pub fn get(&self) -> T {
        let size = self.size.fetch_sub(1, Ordering::AcqRel);
        // implementation error if this fails
        assert!(size <= self.stack.len());
        let mut locked = self.stack[size - 1].lock();
        std::mem::take(&mut *locked)
    }

    pub fn set(&self, value: T) {
        let size = self.size.fetch_add(1, Ordering::AcqRel);
        // implementation error if this fails
        assert!(size <= self.stack.len());
        let mut locked = self.stack[size - 1].lock();
        *locked = value;
    }
}
