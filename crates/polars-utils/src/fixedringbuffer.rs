pub struct FixedRingBuffer<T> {
    start: usize,
    length: usize,
    buffer: *mut T,

    buffer_capacity: usize,

    /// I know that `buffer` also stores a capacity, but these are different. I promise.
    capacity: usize,
}

impl<T> FixedRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let buffer = Vec::with_capacity(capacity);

        Self {
            start: 0,
            length: 0,

            buffer_capacity: buffer.capacity(),
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

    #[inline(always)]
    const fn wrapping_add(&self, x: usize, n: usize) -> usize {
        assert!(n <= self.capacity);

        x.wrapping_add(n).wrapping_sub(if self.capacity - n <= x {
            self.capacity
        } else {
            0
        })
    }

    // SAFETY: This value is never read again
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let item = unsafe { self.buffer.wrapping_add(self.start).read() };
        self.start = self.wrapping_add(self.start, 1);
        self.length -= 1;
        Some(item)
    }

    pub fn push(&mut self, value: T) -> Option<()> {
        if self.is_full() {
            return None;
        }

        let offset = self.wrapping_add(self.start, self.len());

        unsafe { self.buffer.wrapping_add(offset).write(value) };
        self.length += 1;

        Some(())
    }
}

impl<T: Copy> FixedRingBuffer<T> {
    pub fn fill_repeat(&mut self, value: T, num: usize) -> usize {
        if num == 0 || self.is_full() {
            return 0;
        }

        let num = usize::min(num, self.remaining_capacity());

        let start = self.wrapping_add(self.start, self.len());
        let end = self.wrapping_add(start, num);

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
            let offset = self.wrapping_add(self.start, i);
            unsafe { self.buffer.wrapping_add(offset).read() };
        }

        unsafe { Vec::from_raw_parts(self.buffer, 0, self.buffer_capacity) };
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
