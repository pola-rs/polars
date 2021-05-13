use crate::datatypes::UInt64Chunked;
use crate::prelude::*;
use crate::POOL;
use ahash::RandomState;
use arrow::array::ArrayRef;
use hashbrown::{hash_map::RawEntryMut, HashMap};
use itertools::Itertools;
use rayon::prelude::*;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

// Read more:
//  https://www.cockroachlabs.com/blog/vectorized-hash-joiner/
//  http://myeyesareblind.com/2017/02/06/Combine-hash-values/

pub trait VecHash {
    /// Compute the hash for all values in the array.
    ///
    /// This currently only works with the AHash RandomState hasher builder.
    fn vec_hash(&self, _random_state: RandomState) -> UInt64Chunked {
        unimplemented!()
    }
}

pub trait VecHashId {
    /// Compute the hash id by interpreting the bits as 64 bits.
    ///
    /// Watch out for [accidental quadratic behavior](https://accidentallyquadratic.tumblr.com/post/153545455987/rust-hash-iteration-reinsertion)
    fn vec_hash_id(&self) -> UInt64Chunked {
        unimplemented!()
    }
}

/// A random u64 used for the None case
/// the other values hash `T` instead of `Option<T>`
const RANDOM_U64: u64 = 4352984574;

impl VecHashId for UInt64Chunked {
    fn vec_hash_id(&self) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| match opt_v {
            None => RANDOM_U64,
            Some(v) => v,
        })
    }
}

impl VecHashId for UInt32Chunked {
    fn vec_hash_id(&self) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| match opt_v {
            None => RANDOM_U64,
            Some(v) => v as u64,
        })
    }
}

impl VecHashId for Int32Chunked {
    fn vec_hash_id(&self) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| match opt_v {
            None => RANDOM_U64,
            Some(v) => (unsafe { std::mem::transmute::<i32, u32>(v) }) as u64,
        })
    }
}

impl VecHashId for Int64Chunked {
    fn vec_hash_id(&self) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| match opt_v {
            None => RANDOM_U64,
            Some(v) => unsafe { std::mem::transmute::<i64, u64>(v) },
        })
    }
}

impl Series {
    pub fn vec_hash_id(&self) -> UInt64Chunked {
        use DataType::*;
        match self.dtype() {
            UInt64 => self.u64().unwrap().vec_hash_id(),
            Int64 => self.i64().unwrap().vec_hash_id(),
            Int32 => self.i32().unwrap().vec_hash_id(),
            UInt32 => self.u32().unwrap().vec_hash_id(),
            _ => unimplemented!(),
        }
    }
}

impl<T> VecHash for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash,
{
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        // Note that we don't use the no null branch! This can break in unexpected ways.
        // for instance with threading we split an array in n_threads, this may lead to
        // splits that have no nulls and splits that have nulls. Then one array is hashed with
        // Option<T> and the other array with T.
        // Meaning that they cannot be compared. By always hashing on Option<T> the random_state is
        // the only deterministic seed.
        self.branch_apply_cast_numeric_no_null(|opt_v| {
            let mut hasher = random_state.build_hasher();
            opt_v.hash(&mut hasher);
            hasher.finish()
        })
    }
}

impl VecHash for Utf8Chunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| {
            let mut hasher = random_state.build_hasher();
            opt_v.hash(&mut hasher);
            hasher.finish()
        })
    }
}

impl VecHash for BooleanChunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| {
            let mut hasher = random_state.build_hasher();
            opt_v.hash(&mut hasher);
            hasher.finish()
        })
    }
}

impl VecHash for Float32Chunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| {
            let opt_v = opt_v.map(|v| v.to_bits());
            let mut hasher = random_state.build_hasher();
            opt_v.hash(&mut hasher);
            hasher.finish()
        })
    }
}
impl VecHash for Float64Chunked {
    fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
        self.branch_apply_cast_numeric_no_null(|opt_v| {
            let opt_v = opt_v.map(|v| v.to_bits());
            let mut hasher = random_state.build_hasher();
            opt_v.hash(&mut hasher);
            hasher.finish()
        })
    }
}

impl VecHash for ListChunked {}

pub struct IdHasher {
    hash: u64,
}

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("IdHasher should only be used for integer keys <= 64 bit precision")
    }

    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64)
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    fn write_i32(&mut self, i: i32) {
        // Safety:
        // same number of bits
        unsafe { self.write_u32(std::mem::transmute::<i32, u32>(i)) }
    }

    fn write_i64(&mut self, i: i64) {
        // Safety:
        // same number of bits
        unsafe { self.write_u64(std::mem::transmute::<i64, u64>(i)) }
    }
}

impl Default for IdHasher {
    fn default() -> Self {
        IdHasher { hash: 0 }
    }
}

pub type IdBuildHasher = BuildHasherDefault<IdHasher>;

#[derive(Debug)]
pub(crate) struct IdxHash {
    // idx in row of Series, DataFrame
    pub(crate) idx: u32,
    // precomputed hash of T
    hash: u64,
}

impl Hash for IdxHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

impl IdxHash {
    #[inline]
    pub(crate) fn new(idx: u32, hash: u64) -> Self {
        IdxHash { idx, hash }
    }
}

/// Check if a hash should be processed in that thread.
#[inline]
pub(crate) fn this_thread(h: u64, thread_no: u64, n_threads: u64) -> bool {
    (h + thread_no) % n_threads == 0
}

fn finish_table_from_key_hashes<T>(
    hashes_nd_keys: Vec<(u64, T)>,
    mut hash_tbl: HashMap<T, Vec<u32>, RandomState>,
    offset: usize,
) -> HashMap<T, Vec<u32>, RandomState>
where
    T: Hash + Eq,
{
    hashes_nd_keys
        .into_iter()
        .enumerate()
        .for_each(|(idx, (h, t))| {
            let idx = (idx + offset) as u32;

            let entry = hash_tbl
                .raw_entry_mut()
                // uses the key to check equality to find and entry
                .from_key_hashed_nocheck(h, &t);

            match entry {
                RawEntryMut::Vacant(entry) => {
                    entry.insert_hashed_nocheck(h, t, vec![idx]);
                }
                RawEntryMut::Occupied(mut entry) => {
                    let (_k, v) = entry.get_key_value_mut();
                    v.push(idx);
                }
            }
        });
    hash_tbl
}

pub(crate) fn prepare_hashed_relation<T>(
    b: impl Iterator<Item = T>,
) -> HashMap<T, Vec<u32>, RandomState>
where
    T: Hash + Eq,
{
    let build_hasher = RandomState::default();

    let hashes_nd_keys = b
        .map(|val| {
            let mut hasher = build_hasher.build_hasher();
            val.hash(&mut hasher);
            (hasher.finish(), val)
        })
        .collect::<Vec<_>>();

    let hash_tbl: HashMap<T, Vec<u32>, RandomState> =
        HashMap::with_capacity_and_hasher(hashes_nd_keys.len(), build_hasher);

    finish_table_from_key_hashes(hashes_nd_keys, hash_tbl, 0)
}

pub(crate) fn prepare_hashed_relation_threaded<T, I>(
    iters: Vec<I>,
) -> Vec<HashMap<T, Vec<u32>, RandomState>>
where
    I: Iterator<Item = T> + Send,
    T: Send + Hash + Eq + Sync + Copy,
{
    let n_threads = iters.len();
    let (hashes_and_keys, build_hasher) = create_hash_and_keys_threaded_vectorized(iters, None);
    let size = hashes_and_keys.iter().fold(0, |acc, v| acc + v.len());

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let build_hasher = build_hasher.clone();
            let hashes_and_keys = &hashes_and_keys;
            let thread_no = thread_no as u64;
            let mut hash_tbl: HashMap<T, Vec<u32>, RandomState> =
                HashMap::with_capacity_and_hasher(size / (5 * n_threads), build_hasher);

            let n_threads = n_threads as u64;
            let mut offset = 0;
            for hashes_and_keys in hashes_and_keys {
                let len = hashes_and_keys.len();
                hashes_and_keys
                    .iter()
                    .enumerate()
                    .for_each(|(idx, (h, k))| {
                        let idx = idx as u32;
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if this_thread(*h, thread_no, n_threads) {
                            let idx = idx + offset;
                            let entry = hash_tbl
                                .raw_entry_mut()
                                // uses the key to check equality to find and entry
                                .from_key_hashed_nocheck(*h, &k);

                            match entry {
                                RawEntryMut::Vacant(entry) => {
                                    entry.insert_hashed_nocheck(*h, *k, vec![idx]);
                                }
                                RawEntryMut::Occupied(mut entry) => {
                                    let (_k, v) = entry.get_key_value_mut();
                                    v.push(idx);
                                }
                            }
                        }
                    });

                offset += len as u32;
            }
            hash_tbl
        })
    })
    .collect()
}

pub(crate) fn create_hash_and_keys_threaded_vectorized<I, T>(
    iters: Vec<I>,
    build_hasher: Option<RandomState>,
) -> (Vec<Vec<(u64, T)>>, RandomState)
where
    I: IntoIterator<Item = T> + Send,
    T: Send + Hash + Eq,
{
    let build_hasher = build_hasher.unwrap_or_default();
    let hashes = POOL.install(|| {
        iters
            .into_par_iter()
            .map(|iter| {
                // create hashes and keys
                iter.into_iter()
                    .map(|val| {
                        let mut hasher = build_hasher.build_hasher();
                        val.hash(&mut hasher);
                        (hasher.finish(), val)
                    })
                    .collect_vec()
            })
            .collect()
    });
    (hashes, build_hasher)
}

// Combines two hashes into one hash
// http://myeyesareblind.com/2017/02/06/Combine-hash-values/
fn combine_hashes(l: u64, r: u64) -> u64 {
    let hash = (17 * 37u64).wrapping_add(l);
    hash.wrapping_mul(37).wrapping_add(r)
}

pub(crate) fn df_rows_to_hashes_threaded(
    keys: &[DataFrame],
    hasher_builder: Option<RandomState>,
) -> (Vec<UInt64Chunked>, RandomState) {
    let hasher_builder = hasher_builder.unwrap_or_default();

    let hashes = POOL.install(|| {
        keys.into_par_iter()
            .map(|df| {
                let hb = hasher_builder.clone();
                let (ca, _) = df_rows_to_hashes(df, Some(hb));
                ca
            })
            .collect()
    });
    (hashes, hasher_builder)
}

pub(crate) fn df_rows_to_hashes(
    keys: &DataFrame,
    build_hasher: Option<RandomState>,
) -> (UInt64Chunked, RandomState) {
    let build_hasher = build_hasher.unwrap_or_default();
    let hashes: Vec<_> = keys
        .columns
        .iter()
        .map(|s| {
            let h = s.vec_hash(build_hasher.clone());
            // if this fails we have unexpected groupby results.
            debug_assert_eq!(h.null_count(), 0);
            h
        })
        .collect();

    let mut iter = hashes.into_iter();
    let first = iter.next().unwrap();

    // take the columns of hashes and create one hash from them.
    // All columns have the same no. of chunks so we can take the fast path.
    (
        iter.fold(first, |acc, s| {
            let chunks = acc
                .data_views()
                .zip(s.data_views())
                .map(|(array_left, array_right)| {
                    let av: AlignedVec<_> = array_left
                        .iter()
                        .zip(array_right)
                        .map(|(&l, &r)| combine_hashes(l, r))
                        .collect();
                    Arc::new(av.into_primitive_array::<UInt64Type>(None)) as ArrayRef
                })
                .collect();
            UInt64Chunked::new_from_chunks("", chunks)
        }),
        build_hasher,
    )
}
