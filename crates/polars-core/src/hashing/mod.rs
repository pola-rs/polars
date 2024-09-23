mod identity;
pub(crate) mod vector_hasher;

use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
pub use identity::*;
pub use vector_hasher::*;

use crate::prelude::*;

// hash combine from c++' boost lib
#[inline]
pub fn _boost_hash_combine(l: u64, r: u64) -> u64 {
    l ^ r.wrapping_add(0x9e3779b9u64.wrapping_add(l << 6).wrapping_add(r >> 2))
}

// We must strike a balance between cache
// Overallocation seems a lot more expensive than resizing so we start reasonable small.
pub const _HASHMAP_INIT_SIZE: usize = 512;

/// Utility function used as comparison function in the hashmap.
/// The rationale is that equality is an AND operation and therefore its probability of success
/// declines rapidly with the number of keys. Instead of first copying an entire row from both
/// sides and then do the comparison, we do the comparison value by value catching early failures
/// eagerly.
///
/// # Safety
/// Doesn't check any bounds
#[inline]
pub(crate) unsafe fn compare_df_rows(keys: &DataFrame, idx_a: usize, idx_b: usize) -> bool {
    for s in keys.get_columns() {
        if !s.equal_element(idx_a, idx_b, s) {
            return false;
        }
    }
    true
}

/// Populate a multiple key hashmap with row indexes.
///
/// Instead of the keys (which could be very large), the row indexes are stored.
/// To check if a row is equal the original DataFrame is also passed as ref.
/// When a hash collision occurs the indexes are ptrs to the rows and the rows are compared
/// on equality.
pub fn populate_multiple_key_hashmap<V, H, F, G>(
    hash_tbl: &mut HashMap<IdxHash, V, H>,
    // row index
    idx: IdxSize,
    // hash
    original_h: u64,
    // keys of the hash table (will not be inserted, the indexes will be used)
    // the keys are needed for the equality check
    keys: &DataFrame,
    // value to insert
    vacant_fn: G,
    // function that gets a mutable ref to the occupied value in the hash table
    mut occupied_fn: F,
) where
    G: Fn() -> V,
    F: FnMut(&mut V),
    H: BuildHasher,
{
    let entry = hash_tbl
        .raw_entry_mut()
        // uses the idx to probe rows in the original DataFrame with keys
        // to check equality to find an entry
        // this does not invalidate the hashmap as this equality function is not used
        // during rehashing/resize (then the keys are already known to be unique).
        // Only during insertion and probing an equality function is needed
        .from_hash(original_h, |idx_hash| {
            // first check the hash values
            // before we incur a cache miss
            idx_hash.hash == original_h && {
                let key_idx = idx_hash.idx;
                // SAFETY:
                // indices in a group_by operation are always in bounds.
                unsafe { compare_df_rows(keys, key_idx as usize, idx as usize) }
            }
        });
    match entry {
        RawEntryMut::Vacant(entry) => {
            entry.insert_hashed_nocheck(original_h, IdxHash::new(idx, original_h), vacant_fn());
        },
        RawEntryMut::Occupied(mut entry) => {
            let (_k, v) = entry.get_key_value_mut();
            occupied_fn(v);
        },
    }
}
