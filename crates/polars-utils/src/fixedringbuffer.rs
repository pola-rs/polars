/// A ring-buffer with a size determined at creation-time
///
/// This makes it perfectly suited for buffers that produce and consume at different speeds.
pub struct FixedRingBuffer<T> {
    start: usize,
    length: usize,
    buffer: *mut T,
    /// The wanted fixed capacity in the buffer
    capacity: usize,

    /// The actually allocated capacity, this should not be used for any calculations and it purely
    /// used for the deallocation.
    _buffer_capacity: usize,
}

#[inline(always)]
const fn wrapping_add(x: usize, n: usize, capacity: usize) -> usize {
    assert!(n <= capacity);

    let sub = if capacity - n <= x { capacity } else { 0 };

    x.wrapping_add(n).wrapping_sub(sub)
}

impl<T> FixedRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let buffer = Vec::with_capacity(capacity);

        Self {
            start: 0,
            length: 0,

            _buffer_capacity: buffer.capacity(),
            buffer: buffer.leak() as *mut [T] as *mut T,
            capacity,
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.length
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    pub const fn remaining_capacity(&self) -> usize {
        self.capacity - self.len()
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len() == self.capacity
    }

    /// Get a reference to all elements in the form of two slices.
    ///
    /// These are in the listed in the order of being pushed into the buffer.
    #[inline]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        // SAFETY: Only pick the part that is actually defined
        if self.capacity - self.length > self.start {
            (
                unsafe {
                    std::slice::from_raw_parts(self.buffer.wrapping_add(self.start), self.length)
                },
                &[],
            )
        } else {
            (
                unsafe {
                    std::slice::from_raw_parts(
                        self.buffer.wrapping_add(self.start),
                        self.capacity - self.start,
                    )
                },
                unsafe {
                    std::slice::from_raw_parts(
                        self.buffer,
                        wrapping_add(self.start, self.length, self.capacity),
                    )
                },
            )
        }
    }

    /// Pop an item at the front of the [`FixedRingBuffer`]
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        // SAFETY: This value is never read again
        let item = unsafe { self.buffer.wrapping_add(self.start).read() };
        self.start = wrapping_add(self.start, 1, self.capacity);
        self.length -= 1;
        Some(item)
    }

    /// Push an item into the [`FixedRingBuffer`]
    ///
    /// Returns `None` if there is no more space
    #[inline]
    pub fn push(&mut self, value: T) -> Option<()> {
        if self.is_full() {
            return None;
        }

        let offset = wrapping_add(self.start, self.len(), self.capacity);

        unsafe { self.buffer.wrapping_add(offset).write(value) };
        self.length += 1;

        Some(())
    }
}

impl<T: Copy> FixedRingBuffer<T> {
    /// Add at most `num` items of `value` into the [`FixedRingBuffer`]
    ///
    /// This returns the amount of items actually added.
    pub fn fill_repeat(&mut self, value: T, num: usize) -> usize {
        if num == 0 || self.is_full() {
            return 0;
        }

        let num = usize::min(num, self.remaining_capacity());

        let start = wrapping_add(self.start, self.len(), self.capacity);
        let end = wrapping_add(start, num, self.capacity);

        if start < end {
            unsafe { std::slice::from_raw_parts_mut(self.buffer.wrapping_add(start), num) }
                .fill(value);
        } else {
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.buffer.wrapping_add(start),
                    self.capacity - start,
                )
            }
            .fill(value);

            if end != 0 {
                unsafe { std::slice::from_raw_parts_mut(self.buffer, end) }.fill(value);
            }
        }

        self.length += num;

        num
    }
}

impl<T> Drop for FixedRingBuffer<T> {
    fn drop(&mut self) {
        for i in 0..self.length {
            let offset = wrapping_add(self.start, i, self.capacity);
            unsafe { self.buffer.wrapping_add(offset).read() };
        }

        unsafe { Vec::from_raw_parts(self.buffer, 0, self._buffer_capacity) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut frb = FixedRingBuffer::new(256);

        assert!(frb.pop_front().is_none());

        frb.push(1).unwrap();
        frb.push(3).unwrap();

        assert_eq!(frb.pop_front(), Some(1));
        assert_eq!(frb.pop_front(), Some(3));
        assert_eq!(frb.pop_front(), None);

        assert!(!frb.is_full());
        assert_eq!(frb.fill_repeat(42, 300), 256);
        assert!(frb.is_full());

        for _ in 0..256 {
            assert_eq!(frb.pop_front(), Some(42));
            assert!(!frb.is_full());
        }
        assert_eq!(frb.pop_front(), None);
    }

    #[test]
    fn boxed() {
        let mut frb = FixedRingBuffer::new(256);

        assert!(frb.pop_front().is_none());

        frb.push(Box::new(1)).unwrap();
        frb.push(Box::new(3)).unwrap();

        assert_eq!(frb.pop_front(), Some(Box::new(1)));
        assert_eq!(frb.pop_front(), Some(Box::new(3)));
        assert_eq!(frb.pop_front(), None);
    }
}
