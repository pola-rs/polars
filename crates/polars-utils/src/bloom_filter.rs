//! Split-block bloom filter, the layout of parquet's bloom filters: 32-byte
//! blocks, and a key sets one bit in each of the eight words of one block.
//! See <https://github.com/apache/parquet-format/blob/master/BloomFilter.md>.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::Zeroable;
use bytemuck::allocation::zeroed_vec;

use crate::mem::prefetch::prefetch_write;

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

/// Eight words of a filter. Aligned so that a block never spans two cache
/// lines.
#[derive(Clone, Copy, Zeroable)]
#[repr(C, align(32))]
struct Block([u32; 8]);

/// A split-block bloom filter over 64-bit hashes.
#[derive(Clone)]
pub struct SplitBlockBloom {
    blocks: Vec<Block>,
}

impl SplitBlockBloom {
    fn num_blocks_for(num_keys: usize, bits_per_key: usize) -> usize {
        num_keys
            .saturating_mul(bits_per_key)
            .div_ceil(BLOCK_BYTES * 8)
            .max(1)
            .next_power_of_two()
    }

    /// The bytes [`AtomicSplitBlockBloom::with_capacity`] allocates.
    pub fn size_for(num_keys: usize, bits_per_key: usize) -> usize {
        Self::num_blocks_for(num_keys, bits_per_key).saturating_mul(BLOCK_BYTES)
    }

    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * BLOCK_BYTES
    }

    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        let b = block_index(hash, self.blocks.len());
        // SAFETY: `b` is below the number of blocks.
        let block = unsafe { self.blocks.get_unchecked(b).0 };
        block_contains(block, block_mask(hash))
    }
}

/// Hashes are inserted in groups of this many.
const INSERT_GROUP: usize = 64;
/// Filters with fewer blocks than this stay in the cache and aren't prefetched.
const MIN_PREFETCH_BLOCKS: usize = 1 << 15;

/// The words of a [`Block`], two at a time, so fewer atomic operations are
/// needed per key.
#[repr(C, align(32))]
struct AtomicBlock([AtomicU64; 4]);

const _: () = assert!(
    size_of::<AtomicBlock>() == size_of::<Block>()
        && align_of::<AtomicBlock>() == align_of::<Block>()
);

/// A [`SplitBlockBloom`] that many threads insert into at once.
pub struct AtomicSplitBlockBloom {
    blocks: Vec<AtomicBlock>,
}

impl AtomicSplitBlockBloom {
    /// A filter sized for `num_keys` keys at `bits_per_key` bits each, rounded
    /// up to a power of two blocks. Its pages are only touched once a key is
    /// inserted in them.
    pub fn with_capacity(num_keys: usize, bits_per_key: usize) -> Self {
        let blocks: Vec<Block> =
            zeroed_vec(SplitBlockBloom::num_blocks_for(num_keys, bits_per_key));
        // SAFETY: the blocks have the same size and alignment, and any bits
        // are valid in both.
        Self {
            blocks: unsafe { cast_vec(blocks) },
        }
    }

    /// The number of bits the filter holds.
    pub fn num_bits(&self) -> usize {
        self.blocks.len() * BLOCK_BYTES * 8
    }

    /// Insert every hash of `hashes`. The masks of a group of hashes are
    /// computed together, which vectorizes, and on a large filter the blocks
    /// of the group are fetched before any is written, so that their cache
    /// misses overlap.
    pub fn insert_many(&self, hashes: &[u64]) {
        let prefetch = self.blocks.len() >= MIN_PREFETCH_BLOCKS;
        let mut masks = [[0u32; 8]; INSERT_GROUP];
        let mut idxs = [0usize; INSERT_GROUP];
        for group in hashes.chunks(INSERT_GROUP) {
            for ((hash, mask), b) in group.iter().zip(&mut masks).zip(&mut idxs) {
                *mask = block_mask(*hash);
                *b = block_index(*hash, self.blocks.len());
            }
            let idxs = &idxs[..group.len()];
            if prefetch {
                for b in idxs {
                    // SAFETY: `b` is below the number of blocks.
                    let block = unsafe { self.blocks.get_unchecked(*b) };
                    prefetch_write((block as *const AtomicBlock).cast());
                }
            }
            for (mask, b) in masks.iter().zip(idxs) {
                // SAFETY: `b` is below the number of blocks.
                let block = unsafe { &self.blocks.get_unchecked(*b).0 };
                let mask: [u64; 4] = bytemuck::cast(*mask);
                let words: [u64; 4] = std::array::from_fn(|i| block[i].load(Ordering::Relaxed));
                let mut missing = 0;
                for i in 0..4 {
                    missing |= mask[i] & !words[i];
                }
                // A block that has the bits of the key already is only read,
                // so the blocks of keys seen before stay shared between
                // threads.
                if missing != 0 {
                    for i in 0..4 {
                        if mask[i] & !words[i] != 0 {
                            block[i].fetch_or(mask[i], Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }

    /// The filter, once no thread inserts anymore.
    pub fn into_inner(self) -> SplitBlockBloom {
        // SAFETY: as in `with_capacity`.
        SplitBlockBloom {
            blocks: unsafe { cast_vec(self.blocks) },
        }
    }
}

/// The allocation of `v` as a `Vec<U>`.
///
/// # Safety
/// `U` must have the size and alignment of `T`, and every value of `T` must be
/// a valid `U`.
unsafe fn cast_vec<T, U>(v: Vec<T>) -> Vec<U> {
    let mut v = ManuallyDrop::new(v);
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast(), v.len(), v.capacity()) }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn spread_hashes(range: std::ops::Range<u64>) -> Vec<u64> {
        range.map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect()
    }

    fn words(bloom: &SplitBlockBloom) -> Vec<[u32; 8]> {
        bloom.blocks.iter().map(|b| b.0).collect()
    }

    /// The words of the filter of `hashes` that the bitset functions build.
    fn bitset_words(hashes: &[u64], size_bytes: usize) -> Vec<[u32; 8]> {
        let mut bitset = vec![0u8; size_bytes];
        hashes.iter().for_each(|h| insert(&mut bitset, *h));
        bitset
            .chunks_exact(BLOCK_BYTES)
            .map(|b| load_block(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn matches_bitset() {
        // Below and above the size that is prefetched, in batches that do and
        // don't fill a group.
        for keys in [1000, 1 << 20] {
            let hashes = spread_hashes(0..keys as u64);
            for batch in [7, 100, keys] {
                let atomic = AtomicSplitBlockBloom::with_capacity(keys, 8);
                hashes.chunks(batch).for_each(|c| atomic.insert_many(c));
                assert_eq!(atomic.num_bits(), SplitBlockBloom::size_for(keys, 8) * 8);
                let bloom = atomic.into_inner();
                assert!(words(&bloom) == bitset_words(&hashes, bloom.size_bytes()));
            }
        }
    }

    #[test]
    fn contains_like_bitset() {
        let hashes = spread_hashes(0..1000);
        let atomic = AtomicSplitBlockBloom::with_capacity(1000, 8);
        atomic.insert_many(&hashes);
        let bloom = atomic.into_inner();
        let mut bitset = vec![0u8; bloom.size_bytes()];
        hashes.iter().for_each(|h| insert(&mut bitset, *h));
        assert!(hashes.iter().all(|h| bloom.contains(*h)));
        for h in spread_hashes(1000..2000) {
            assert_eq!(bloom.contains(h), is_in_set(&bitset, h));
        }
    }

    #[test]
    fn insert_from_several_threads() {
        let hashes = spread_hashes(0..100_000);
        let shared = AtomicSplitBlockBloom::with_capacity(hashes.len(), 8);
        std::thread::scope(|s| {
            for part in hashes.chunks(10_000) {
                let shared = &shared;
                s.spawn(move || shared.insert_many(part));
            }
        });
        let bloom = shared.into_inner();
        assert!(words(&bloom) == bitset_words(&hashes, bloom.size_bytes()));
    }
}
