use ahash::RandomState;
use hashbrown::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

pub(crate) struct IdHasher {
    hash: u64,
}

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("IdHasher should only be used for u64 keys")
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }
}

impl Default for IdHasher {
    fn default() -> Self {
        IdHasher { hash: 0 }
    }
}

pub(crate) type IdBuildHasher = BuildHasherDefault<IdHasher>;

pub(crate) fn prepare_hashed_relation<T>(
    b: impl Iterator<Item = T>,
) -> HashMap<T, Vec<usize>, RandomState>
where
    T: Hash + Eq,
{
    let random_state = RandomState::default();

    let hashes_nd_keys = b
        .map(|val| {
            let mut hasher = random_state.build_hasher();
            val.hash(&mut hasher);
            (hasher.finish(), val)
        })
        .collect::<Vec<_>>();

    let mut hash_tbl: HashMap<T, Vec<usize>, RandomState> =
        HashMap::with_capacity_and_hasher(hashes_nd_keys.len(), random_state);

    hashes_nd_keys
        .into_iter()
        .enumerate()
        .for_each(|(idx, (h, t))| {
            hash_tbl
                .raw_entry_mut()
                // uses the key to check equality to find and entry
                .from_key_hashed_nocheck(h, &t)
                // if entry is found modify it
                .and_modify(|_k, v| {
                    v.push(idx);
                })
                // otherwise we insert both the key and new Vec without hashing
                .or_insert_with(|| (t, vec![idx]));
        });
    hash_tbl
}
