/// Magic numbers taken from https://github.com/apache/parquet-format/blob/master/BloomFilter.md
const SALT: [u32; 8] = [
    1203114875, 1150766481, 2284105051, 2729912477, 1884591559, 770785867, 2667333959, 1550580529,
];

/// Size of one split-block bloom filter block in bytes.
pub const BLOCK_SIZE: usize = 32;

#[inline]
fn block(bitset: &[u8], block_index: usize) -> &[u8] {
    &bitset[block_index * BLOCK_SIZE..(block_index + 1) * BLOCK_SIZE]
}

#[inline]
fn block_mut(bitset: &mut [u8], block_index: usize) -> &mut [u8] {
    &mut bitset[block_index * BLOCK_SIZE..(block_index + 1) * BLOCK_SIZE]
}

/// Block index for `hash` from its upper 32 bits. `bitset_len` must be a multiple of [`BLOCK_SIZE`].
#[inline]
pub fn hash_to_block_index(hash: u64, bitset_len: usize) -> usize {
    let number_of_blocks = bitset_len as u64 / BLOCK_SIZE as u64;
    let low_hash = hash >> 32;
    let block_index = ((low_hash * number_of_blocks) >> 32) as u32;
    block_index as usize
}

/// Eight single-bit masks (one per `u32` word), derived from the hash's lower 32 bits.
fn new_mask(x: u32) -> [u32; 8] {
    let mut a = [0u32; 8];
    for i in 0..8 {
        let mask = x.wrapping_mul(SALT[i]);
        let mask = mask >> 27;
        let mask = 0x1 << mask;
        a[i] = mask;
    }
    a
}

/// Loads a block from the bitset to the stack.
#[inline]
fn load_block(bitset: &[u8]) -> [u32; 8] {
    let mut a = [0u32; 8];
    let bitset = bitset.chunks_exact(4).take(8);
    for (a, chunk) in a.iter_mut().zip(bitset) {
        *a = u32::from_le_bytes(chunk.try_into().unwrap())
    }
    a
}

/// Assigns a block from the stack to `bitset`.
#[inline]
fn unload_block(block: [u32; 8], bitset: &mut [u8]) {
    let bitset = bitset.chunks_exact_mut(4).take(8);
    for (a, chunk) in block.iter().zip(bitset) {
        let a = a.to_le_bytes();
        chunk[0] = a[0];
        chunk[1] = a[1];
        chunk[2] = a[2];
        chunk[3] = a[3];
    }
}

/// Probe one [`BLOCK_SIZE`]-byte bloom block for `hash`.
///
/// `false` if all eight required bits are absent; `true` otherwise.
pub fn is_maybe_in_block(block: &[u8], hash: u64) -> bool {
    debug_assert_eq!(block.len(), BLOCK_SIZE);
    let key = hash as u32;
    let mask = new_mask(key);
    let block_mask = load_block(block);

    for i in 0..8 {
        if mask[i] & block_mask[i] == 0 {
            return false;
        }
    }
    true
}

/// Probe a bloom filter bitset for `hash`.
///
/// In Parquet this bitset is the filter for one column chunk. `false` means
/// definitely not present; `true` means may be present (or inconclusive).
pub fn is_maybe_in_bitset(bitset: &[u8], hash: u64) -> bool {
    let block_index = hash_to_block_index(hash, bitset.len());
    is_maybe_in_block(block(bitset, block_index), hash)
}

/// Inserts a new hash: set the eight bit positions for `hash` in the bloom bitset.
pub fn insert(bitset: &mut [u8], hash: u64) {
    let block_index = hash_to_block_index(hash, bitset.len());
    let mask = new_mask(hash as u32);
    let block = block_mut(bitset, block_index);
    let mut block_mask = load_block(block);
    for i in 0..8 {
        block_mask[i] |= mask[i];
    }
    unload_block(block_mask, block);
}
