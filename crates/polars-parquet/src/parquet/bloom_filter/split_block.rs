/// magic numbers taken from https://github.com/apache/parquet-format/blob/master/BloomFilter.md
const SALT: [u32; 8] = [
    1203114875, 1150766481, 2284105051, 2729912477, 1884591559, 770785867, 2667333959, 1550580529,
];

fn hash_to_block_index(hash: u64, len: usize) -> usize {
    let number_of_blocks = len as u64 / 32;
    let low_hash = hash >> 32;
    let block_index = ((low_hash * number_of_blocks) >> 32) as u32;
    block_index as usize
}

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

/// loads a block from the bitset to the stack
#[inline]
fn load_block(bitset: &[u8]) -> [u32; 8] {
    let chunks = bitset.as_chunks::<4>().0;
    std::array::from_fn(|i| u32::from_le_bytes(chunks[i]))
}

/// assigns a block from the stack to `bitset`
#[inline]
fn store_block(block: [u32; 8], bitset: &mut [u8]) {
    let chunks = bitset.as_chunks_mut::<4>().0;
    for (i, x) in block.iter().enumerate() {
        chunks[i] = x.to_le_bytes();
    }
}

/// Returns whether the `hash` is in the set
pub fn is_in_set(bitset: &[u8], hash: u64) -> bool {
    let block_index = hash_to_block_index(hash, bitset.len());
    let key = hash as u32;

    let mask = new_mask(key);
    let slice = &bitset[block_index * 32..(block_index + 1) * 32];
    let block_mask = load_block(slice);

    for i in 0..8 {
        if mask[i] & block_mask[i] == 0 {
            return false;
        }
    }
    true
}

/// Inserts a new hash to the set
pub fn insert(bitset: &mut [u8], hash: u64) {
    let block_index = hash_to_block_index(hash, bitset.len());
    let key = hash as u32;

    let mask = new_mask(key);
    let slice = &bitset[block_index * 32..(block_index + 1) * 32];
    let mut block_mask = load_block(slice);

    for i in 0..8 {
        block_mask[i] |= mask[i];

        let mut_slice = &mut bitset[block_index * 32..(block_index + 1) * 32];
        store_block(block_mask, mut_slice)
    }
}
