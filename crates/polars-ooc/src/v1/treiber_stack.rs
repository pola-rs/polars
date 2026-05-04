use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

struct Node<T> {
    value: T,
    next: *mut Node<T>,
}

/// Lock-free Treiber stack used for spill-entry tracking.
///
/// - [`push`](Self::push): CAS onto head.
/// - [`scan`](Self::scan): non-destructive walk from head, call `f` on each
///   element. Used for spill collection.
/// - [`clear`](Self::clear): swap head → null, free nodes.
///
/// Size: 8 bytes (one `AtomicPtr`).
pub(crate) struct TreiberStack<T> {
    head: AtomicPtr<Node<T>>,
}

unsafe impl<T: Send> Send for TreiberStack<T> {}
unsafe impl<T: Send> Sync for TreiberStack<T> {}

impl<T> Default for TreiberStack<T> {
    fn default() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

impl<T> TreiberStack<T> {
    /// Push a value onto the stack (LIFO).
    pub(crate) fn push(&self, value: T) {
        let node = Box::into_raw(Box::new(Node {
            value,
            next: ptr::null_mut(),
        }));
        loop {
            let old_head = self.head.load(Ordering::Acquire);
            unsafe { (*node).next = old_head };
            if self
                .head
                .compare_exchange_weak(old_head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Walk the chain without modifying it. Call `f` on each element.
    ///
    /// Concurrent pushes are safe — they add to the head, while scan
    /// walks from a snapshot of the head. Newly pushed nodes are not
    /// visited by the current scan.
    pub(crate) fn scan<F: FnMut(&T)>(&self, mut f: F) {
        let mut current = self.head.load(Ordering::Acquire);
        while !current.is_null() {
            let node = unsafe { &*current };
            f(&node.value);
            current = node.next;
        }
    }

    /// Atomically take and drop the entire chain. Must be called at
    /// exclusive boundaries (no concurrent [`scan`](Self::scan)).
    pub(crate) fn clear(&self) {
        let head = self.head.swap(ptr::null_mut(), Ordering::AcqRel);
        let mut current = head;
        while !current.is_null() {
            let node = unsafe { Box::from_raw(current) };
            current = node.next;
        }
    }
}

impl<T> Drop for TreiberStack<T> {
    fn drop(&mut self) {
        let mut current = self.head.load(Ordering::Relaxed);
        while !current.is_null() {
            let node = unsafe { Box::from_raw(current) };
            current = node.next;
        }
    }
}
