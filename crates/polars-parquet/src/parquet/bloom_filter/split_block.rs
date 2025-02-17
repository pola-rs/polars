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
    let mut a = [0u32; 8];
    let bitset = bitset.chunks_exact(4).take(8);
    for (a, chunk) in a.iter_mut().zip(bitset) {
        *a = u32::from_le_bytes(chunk.try_into().unwrap())
    }
    a
}

/// assigns a block from the stack to `bitset`
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
        unload_block(block_mask, mut_slice)
    }
}
