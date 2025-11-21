use std::ops::Range;

/// Reverses indexing direction
pub struct IdxMapper {
    total_len: usize,
    reverse: bool,
}

impl IdxMapper {
    pub fn new(total_len: usize, reverse: bool) -> Self {
        Self { total_len, reverse }
    }
}

impl IdxMapper {
    /// # Panics
    /// `range.end <= self.total_len`
    #[inline]
    pub fn map_range(&self, range: Range<usize>) -> Range<usize> {
        if self.reverse {
            // len: 5
            // array: [0 1 2 3 4]
            // slice: [    2 3  ]
            // in:  1..3 (right-to-left)
            // out: 2..4
            map_range::<true>(self.total_len, range)
        } else {
            range
        }
    }
}

/// # Safety
/// `range.end <= total_len`
#[inline]
pub fn map_range<const REVERSE: bool>(total_len: usize, range: Range<usize>) -> Range<usize> {
    assert!(range.end <= total_len);
    if REVERSE {
        total_len - range.end..total_len - range.start
    } else {
        range
    }
}

#[cfg(test)]
mod tests {
    use super::IdxMapper;

    #[test]
    fn test_idx_map_roundtrip() {
        let map = IdxMapper::new(100, true);

        assert_eq!(map.map_range(map.map_range(5..77)), 5..77);
    }
}
