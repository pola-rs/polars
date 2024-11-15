use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

const INCOMPLETE: usize = 0x0;
const RUNNING: usize = 0x1;
const COMPLETE: usize = 0x2;

unsafe impl<T: Sync + Send> Sync for CacheCell<T> {}
unsafe impl<T: Send> Send for CacheCell<T> {}

pub struct CacheCell<T> {
    state: AtomicUsize,
    value: UnsafeCell<Option<T>>,
}

impl<T> Default for CacheCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CacheCell<T> {
    pub const fn new() -> CacheCell<T> {
        CacheCell {
            value: UnsafeCell::new(None),
            state: AtomicUsize::new(INCOMPLETE),
        }
    }

    /// unblocking version of OnceCell::get_or_try_init
    pub fn get_or_try_init<F, E>(&self, f: F) -> Option<Result<&T, E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if let Some(value) = self.get() {
            return Some(Ok(value));
        }

        match self.try_initialize(f) {
            Some(Err(e)) => return Some(Err(e)),
            None => return None,
            _ => {},
        }

        debug_assert!(self.is_initialized());
        Some(Ok(unsafe { self.get_unchecked() }))
    }

    fn get(&self) -> Option<&T> {
        if self.is_initialized() {
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    fn is_initialized(&self) -> bool {
        self.state.load(Ordering::Acquire) == COMPLETE
    }

    unsafe fn get_unchecked(&self) -> &T {
        debug_assert!(self.is_initialized());
        let slot = &*self.value.get();
        slot.as_ref().unwrap_unchecked()
    }

    fn try_initialize<F, E>(&self, f: F) -> Option<Result<(), E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let curr_state = self.state.load(Ordering::Acquire);

        match curr_state {
            COMPLETE => Some(Ok(())),
            INCOMPLETE => {
                let exchange = self.state.compare_exchange(
                    curr_state,
                    RUNNING,
                    Ordering::Acquire,
                    Ordering::Acquire,
                );

                if exchange.is_err() {
                    return None;
                }

                let ret = match f() {
                    Ok(value) => {
                        let slot: *mut Option<T> = self.value.get();
                        unsafe { *slot = Some(value) };
                        self.state.store(COMPLETE, Ordering::Release);
                        Ok(())
                    },
                    Err(e) => {
                        self.state.store(INCOMPLETE, Ordering::Release);
                        Err(e)
                    },
                };

                Some(ret)
            },
            RUNNING => None,
            _ => panic!("invalid state"),
        }
    }
}
