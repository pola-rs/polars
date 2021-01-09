use ahash::RandomState;
use crossbeam::thread;
use hashbrown::HashMap;
use itertools::Itertools;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

// Read more:
//  https://www.cockroachlabs.com/blog/vectorized-hash-joiner/
//  http://myeyesareblind.com/2017/02/06/Combine-hash-values/

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

fn finish_table_from_key_hashes<T>(
    hashes_nd_keys: Vec<(u64, T)>,
    mut hash_tbl: HashMap<T, Vec<usize>, RandomState>,
    offset: usize,
) -> HashMap<T, Vec<usize>, RandomState>
where
    T: Hash + Eq,
{
    hashes_nd_keys
        .into_iter()
        .enumerate()
        .for_each(|(idx, (h, t))| {
            let idx = idx + offset;
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

    let hash_tbl: HashMap<T, Vec<usize>, RandomState> =
        HashMap::with_capacity_and_hasher(hashes_nd_keys.len(), random_state);

    finish_table_from_key_hashes(hashes_nd_keys, hash_tbl, 0)
}

pub(crate) fn create_hash_threaded_vectorized<I, T>(
    iters: Vec<I>,
) -> (Vec<Vec<(u64, T)>>, RandomState)
where
    I: Iterator<Item = T> + Send,
    T: Send + Hash + Eq,
{
    let random_state = RandomState::default();
    let n_threads = iters.len();

    let hashes = thread::scope(|s| {
        let handles = iters
            .into_iter()
            .map(|iter| {
                // joinhandles
                s.spawn(|_| {
                    // create hashes and keys
                    iter.map(|val| {
                        let mut hasher = random_state.build_hasher();
                        val.hash(&mut hasher);
                        (hasher.finish(), val)
                    })
                    .collect_vec()
                })
            })
            .collect_vec();

        let mut results = Vec::with_capacity(n_threads);
        for h in handles {
            let mut v = h.join().unwrap();
            v.shrink_to_fit();
            results.push(v);
        }
        results
    })
    .unwrap();
    (hashes, random_state)
}

pub(crate) fn prepare_hashed_relation_threaded<T, I>(
    iters: Vec<I>,
) -> HashMap<T, Vec<usize>, RandomState>
where
    I: Iterator<Item = T> + Send,
    T: Send + Hash + Eq,
{
    let (hashes_and_keys, random_state) = create_hash_threaded_vectorized(iters);
    let size = hashes_and_keys.iter().fold(0, |acc, v| acc + v.len());

    let mut hash_tbl: HashMap<T, Vec<usize>, RandomState> =
        HashMap::with_capacity_and_hasher(size / 3, random_state);

    let mut offset = 0;
    for hashes_and_keys in hashes_and_keys {
        let len = hashes_and_keys.len();
        hash_tbl = finish_table_from_key_hashes(hashes_and_keys, hash_tbl, offset);
        offset += len;
    }

    hash_tbl
}
