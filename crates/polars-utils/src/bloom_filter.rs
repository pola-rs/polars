//! Split-block bloom filter, the layout of parquet's bloom filters: 32-byte
//! blocks, and a key sets one bit in each of the eight words of one block.
//! See <https://github.com/apache/parquet-format/blob/master/BloomFilter.md>.

const SALT: [u32; 8] = [
    1203114875, 1150766481, 2284105051, 2729912477, 1884591559, 770785867, 2667333959, 1550580529,
];

const BLOCK_BYTES: usize = 32;

/// Always below `num_blocks`.
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
fn load_block(bytes: &[u8; BLOCK_BYTES]) -> [u32; 8] {
    // SAFETY: the block is 8 words of 4 bytes.
    let words: [u32; 8] = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast()) };
    words.map(u32::from_le)
}

#[inline]
fn store_block(block: [u32; 8], bytes: &mut [u8; BLOCK_BYTES]) {
    // SAFETY: the block is 8 words of 4 bytes.
    unsafe { std::ptr::write_unaligned(bytes.as_mut_ptr().cast(), block.map(u32::to_le)) }
}

/// The byte offset of the block of `hash` in `bitset`, which holds at least
/// one block.
#[inline]
fn block_offset(bitset: &[u8], hash: u64) -> usize {
    let num_blocks = bitset.len() / BLOCK_BYTES;
    assert!(num_blocks > 0);
    block_index(hash, num_blocks) * BLOCK_BYTES
}

#[inline]
fn block_bytes(bitset: &[u8], hash: u64) -> &[u8; BLOCK_BYTES] {
    let offset = block_offset(bitset, hash);
    // SAFETY: `offset + BLOCK_BYTES <= bitset.len()`.
    unsafe { &*(bitset.as_ptr().add(offset) as *const [u8; BLOCK_BYTES]) }
}

#[inline]
fn block_bytes_mut(bitset: &mut [u8], hash: u64) -> &mut [u8; BLOCK_BYTES] {
    let offset = block_offset(bitset, hash);
    // SAFETY: `offset + BLOCK_BYTES <= bitset.len()`.
    unsafe { &mut *(bitset.as_mut_ptr().add(offset) as *mut [u8; BLOCK_BYTES]) }
}

#[inline]
fn block_contains(block: [u32; 8], mask: [u32; 8]) -> bool {
    let mut found = true;
    for i in 0..8 {
        found &= block[i] & mask[i] != 0;
    }
    found
}

/// Whether `hash` is in the filter held by `bitset`, at least one block.
pub fn is_in_set(bitset: &[u8], hash: u64) -> bool {
    let block = load_block(block_bytes(bitset, hash));
    block_contains(block, block_mask(hash))
}

/// Add `hash` to the filter held by `bitset`, at least one block.
pub fn insert(bitset: &mut [u8], hash: u64) {
    let bytes = block_bytes_mut(bitset, hash);
    let mut block = load_block(bytes);
    let mask = block_mask(hash);
    for i in 0..8 {
        block[i] |= mask[i];
    }
    store_block(block, bytes);
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
        // SAFETY: `b` is below the number of blocks.
        let block = unsafe { self.blocks.get_unchecked_mut(b) };
        let mask = block_mask(hash);
        for i in 0..8 {
            block[i] |= mask[i];
        }
    }

    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        let b = block_index(hash, self.blocks.len());
        // SAFETY: `b` is below the number of blocks.
        let block = unsafe { *self.blocks.get_unchecked(b) };
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
    #[should_panic]
    fn is_in_set_needs_a_block() {
        is_in_set(&[], 0);
    }

    #[test]
    #[should_panic]
    fn insert_needs_a_block() {
        insert(&mut [0u8; 31], 0);
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
