//! Split-block bloom filter, the layout of parquet's bloom filters: 32-byte
//! blocks, and a key sets one bit in each of the eight words of one block.
//! See <https://github.com/apache/parquet-format/blob/master/BloomFilter.md>.

const SALT: [u32; 8] = [
    1203114875, 1150766481, 2284105051, 2729912477, 1884591559, 770785867, 2667333959, 1550580529,
];

const BLOCK_BYTES: usize = 32;

#[inline]
fn block_index(hash: u64, num_blocks: usize) -> usize {
    (((hash >> 32) * num_blocks as u64) >> 32) as usize
}

#[inline]
fn block_mask(hash: u64) -> [u32; 8] {
    let key = hash as u32;
    std::array::from_fn(|i| 1u32 << (key.wrapping_mul(SALT[i]) >> 27))
}

#[inline]
fn load_block(bitset: &[u8]) -> [u32; 8] {
    let chunks = bitset.as_chunks::<4>().0;
    std::array::from_fn(|i| u32::from_le_bytes(chunks[i]))
}

#[inline]
fn store_block(block: [u32; 8], bitset: &mut [u8]) {
    let chunks = bitset.as_chunks_mut::<4>().0;
    for (i, x) in block.iter().enumerate() {
        chunks[i] = x.to_le_bytes();
    }
}

#[inline]
fn block_contains(block: [u32; 8], mask: [u32; 8]) -> bool {
    let mut found = true;
    for i in 0..8 {
        found &= block[i] & mask[i] != 0;
    }
    found
}

/// Whether `hash` is in the filter held by `bitset`, a whole number of blocks.
pub fn is_in_set(bitset: &[u8], hash: u64) -> bool {
    let b = block_index(hash, bitset.len() / BLOCK_BYTES);
    let block = load_block(&bitset[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES]);
    block_contains(block, block_mask(hash))
}

/// Add `hash` to the filter held by `bitset`, a whole number of blocks.
pub fn insert(bitset: &mut [u8], hash: u64) {
    let b = block_index(hash, bitset.len() / BLOCK_BYTES);
    let slice = &mut bitset[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
    let mut block = load_block(slice);
    let mask = block_mask(hash);
    for i in 0..8 {
        block[i] |= mask[i];
    }
    store_block(block, slice);
}

/// A split-block bloom filter over 64-bit hashes.
#[derive(Clone)]
pub struct SplitBlockBloom {
    blocks: Vec<[u32; 8]>,
}

impl SplitBlockBloom {
    fn num_blocks_for(num_keys: usize, bits_per_key: usize) -> usize {
        num_keys
            .saturating_mul(bits_per_key)
            .div_ceil(BLOCK_BYTES * 8)
            .max(1)
            .next_power_of_two()
    }

    /// The bytes `with_capacity` allocates.
    pub fn size_for(num_keys: usize, bits_per_key: usize) -> usize {
        Self::num_blocks_for(num_keys, bits_per_key).saturating_mul(BLOCK_BYTES)
    }

    /// A filter sized for `num_keys` keys at `bits_per_key` bits each, rounded
    /// up to a power of two blocks.
    pub fn with_capacity(num_keys: usize, bits_per_key: usize) -> Self {
        Self {
            blocks: vec![[0; 8]; Self::num_blocks_for(num_keys, bits_per_key)],
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * BLOCK_BYTES
    }

    /// The number of bits the filter holds.
    pub fn num_bits(&self) -> usize {
        self.size_bytes() * 8
    }

    #[inline]
    pub fn insert(&mut self, hash: u64) {
        let b = block_index(hash, self.blocks.len());
        let block = &mut self.blocks[b];
        let mask = block_mask(hash);
        for i in 0..8 {
            block[i] |= mask[i];
        }
    }

    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        let block = self.blocks[block_index(hash, self.blocks.len())];
        block_contains(block, block_mask(hash))
    }

    /// Add every key of `other`, a filter of the same size.
    pub fn union_with(&mut self, other: &Self) {
        assert_eq!(self.blocks.len(), other.blocks.len());
        for (a, b) in self.blocks.iter_mut().zip(&other.blocks) {
            for i in 0..8 {
                a[i] |= b[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_matches_bitset() {
        let mut bloom = SplitBlockBloom::with_capacity(1000, 8);
        let mut bitset = vec![0u8; bloom.size_bytes()];
        let hashes: Vec<u64> = (0..1000u64)
            .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15))
            .collect();
        for &h in &hashes {
            bloom.insert(h);
            insert(&mut bitset, h);
        }
        for h in hashes
            .iter()
            .copied()
            .chain((1000..2000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)))
        {
            assert_eq!(bloom.contains(h), is_in_set(&bitset, h));
        }
        assert!(hashes.iter().all(|&h| bloom.contains(h)));
    }

    #[test]
    fn union() {
        let mut a = SplitBlockBloom::with_capacity(100, 8);
        let mut b = a.clone();
        a.insert(1);
        b.insert(2);
        a.union_with(&b);
        assert!(a.contains(1) && a.contains(2));
    }
}
