use polars_utils::IdxSize;

// Tracks the sum of consecutive values in a dynamically sized array where the values can be written
// in any order.
pub struct ConsecutiveCountState {
    counts: Box<[IdxSize]>,
    next_index: usize,
    sum: IdxSize,
}

impl ConsecutiveCountState {
    pub fn new(len: usize) -> Self {
        Self {
            counts: vec![IdxSize::MAX; len].into_boxed_slice(),
            next_index: 0,
            sum: 0,
        }
    }

    /// Sum of all consecutive counts.
    pub fn sum(&self) -> IdxSize {
        self.sum
    }

    /// Write count at index.
    pub fn write(&mut self, index: usize, count: IdxSize) {
        debug_assert!(
            self.counts[index] == IdxSize::MAX,
            "second write to same index"
        );
        debug_assert!(count != IdxSize::MAX, "count can not be IdxSize::MAX");

        self.counts[index] = count;

        // Update sum and next index.
        while self.next_index < self.counts.len() {
            let count = self.counts[self.next_index];
            if count == IdxSize::MAX {
                break;
            }
            self.sum += count;
            self.next_index += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn counts(&self) -> impl Iterator<Item = Option<IdxSize>> + '_ {
        self.counts
            .iter()
            .map(|&count| (count != IdxSize::MAX).then_some(count))
    }
}
