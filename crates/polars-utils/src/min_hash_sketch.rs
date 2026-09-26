const NUM_ENTRIES: usize = 256;

#[derive(Clone, Copy, Debug)]
struct Entry {
    tag: u32,
    count: u32,
}

/// A sample of distinct keys chosen by hash, along with how often each key
/// occurred. Whether a key is sampled does not depend on how often it occurs.
#[derive(Clone)]
pub struct MinHashSketch {
    entries: Box<[Entry; NUM_ENTRIES]>,
}

impl Default for MinHashSketch {
    fn default() -> Self {
        Self::new()
    }
}

impl MinHashSketch {
    pub fn new() -> Self {
        let empty = Entry {
            tag: u32::MAX,
            count: 0,
        };
        Self {
            entries: vec![empty; NUM_ENTRIES].try_into().unwrap(),
        }
    }

    /// Add a new hash to the sketch.
    #[inline]
    pub fn insert(&mut self, h: u64) {
        const IDX_ODD: u64 = 0x85b0fc582ca15d65; // Randomly chosen.
        const TAG_ODD: u64 = 0x4a0da407caf4e9ad; // Randomly chosen.
        let idx = (h.wrapping_mul(IDX_ODD) >> (64 - NUM_ENTRIES.trailing_zeros())) as usize;
        let tag = (h.wrapping_mul(TAG_ODD) >> 32) as u32;
        let entry = &mut self.entries[idx];
        if tag < entry.tag {
            entry.tag = tag;
            entry.count = 0;
        }
        entry.count += (tag == entry.tag) as u32;
    }

    pub fn combine(&mut self, other: &MinHashSketch) {
        for (entry, other) in self.entries.iter_mut().zip(other.entries.iter()) {
            if other.tag < entry.tag {
                *entry = *other;
            } else if other.tag == entry.tag {
                entry.count += other.count;
            }
        }
    }

    /// Estimates the number of distinct keys as a fraction of the number of
    /// rows, weighting each distinct key equally.
    pub fn key_ratio(&self) -> Option<f64> {
        let mut nonempty = 0u64;
        let mut total = 0u64;
        for entry in self.entries.iter() {
            nonempty += (entry.count > 0) as u64;
            total += entry.count as u64;
        }
        (nonempty > 0).then(|| nonempty as f64 / total as f64)
    }
}
