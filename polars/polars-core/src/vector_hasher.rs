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

// Used to to get a u64 from the hashing keys
pub(crate) trait AsU64 {
    #[allow(clippy::wrong_self_convention)]
    fn as_u64(self) -> u64;
}

impl AsU64 for u32 {
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl AsU64 for u64 {
    #[inline]
    fn as_u64(self) -> u64 {
        self
    }
}

impl AsU64 for Option<u32> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v as u64,
            // just a number
            None => u64::MAX,
        }
    }
}

impl AsU64 for Option<u64> {
    #[inline]
    fn as_u64(self) -> u64 {
        self.unwrap_or(u64::MAX)
    }
}

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

    #[inline]
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
/// Contains an idx of a row in a DataFrame and the precomputed hash of that row.
/// That hash still needs to be used to create another hash to be able to resize hashmaps without
/// accidental quadratic behavior. So do not use an Identity function!
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

/// Contains a ptr to the string slice an the precomputed hash of that string.
/// During rehashes, we will rehash the hash instead of the string, that makes rehashing
/// cheap and allows cache coherent small hash tables.
#[derive(Eq, Copy, Clone)]
pub(crate) struct StrHash<'a> {
    str: Option<&'a str>,
    hash: u64,
}

impl<'a> Hash for StrHash<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

impl<'a> StrHash<'a> {
    pub(crate) fn new(s: Option<&'a str>, hash: u64) -> Self {
        Self { str: s, hash }
    }
}

impl<'a> PartialEq for StrHash<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.str == other.str
    }
}

impl<'a> AsU64 for StrHash<'a> {
    #[inline]
    fn as_u64(self) -> u64 {
        self.hash
    }
}

/// Check if a hash should be processed in that thread.
#[inline]
pub(crate) fn this_thread(h: u64, thread_no: u64, n_threads: u64) -> bool {
    // n % 2^i = n & (2^i - 1)
    // we only accept thread pools that are 2^i
    (h + thread_no) & (n_threads - 1) == 0
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

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let build_hasher = build_hasher.clone();
            let hashes_and_keys = &hashes_and_keys;
            let thread_no = thread_no as u64;
            let mut hash_tbl: HashMap<T, Vec<u32>, RandomState> =
                HashMap::with_hasher(build_hasher);

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

// hash combine from c++' boost lib
fn boost_hash_combine(l: u64, r: u64) -> u64 {
    l ^ r.wrapping_add(0x9e3779b9u64.wrapping_add(l << 6).wrapping_add(r >> 2))
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

    let n_chunks = hashes[0].chunks().len();
    let mut av = AlignedVec::with_capacity_aligned(keys.height());

    // two code paths, one has one layer of indirection less.
    if n_chunks == 1 {
        let chunks: Vec<&[u64]> = hashes
            .iter()
            .map(|ca| {
                ca.downcast_iter()
                    .map(|arr| arr.values())
                    .collect::<Vec<_>>()[0]
            })
            .collect();
        unsafe {
            let chunk_len = chunks.get_unchecked(0).len();

            // over chunk length in column direction
            for idx in 0..chunk_len {
                let hslice = chunks.get_unchecked(0);
                let mut h = *hslice.get_unchecked(idx);

                // in row direction
                for column_i in 1..hashes.len() {
                    let hslice = chunks.get_unchecked(column_i);
                    let h_ = *hslice.get_unchecked(idx);
                    h = boost_hash_combine(h, h_)
                }

                av.push(h);
            }
        }
    // path with more indirection
    } else {
        let chunks: Vec<Vec<&[u64]>> = hashes
            .iter()
            .map(|ca| {
                ca.downcast_iter()
                    .map(|arr| arr.values())
                    .collect::<Vec<_>>()
            })
            .collect();
        unsafe {
            for chunk_i in 0..n_chunks {
                let chunk_len = chunks.get_unchecked(0).get_unchecked(chunk_i).len();

                // over chunk length in column direction
                for idx in 0..chunk_len {
                    let hslice = chunks.get_unchecked(0).get_unchecked(chunk_i);
                    let mut h = *hslice.get_unchecked(idx);

                    // in row direction
                    for column_i in 1..hashes.len() {
                        let hslice = chunks.get_unchecked(column_i).get_unchecked(chunk_i);
                        let h_ = *hslice.get_unchecked(idx);
                        h = boost_hash_combine(h, h_)
                    }

                    av.push(h);
                }
            }
        }
    }

    let chunks = vec![Arc::new(av.into_primitive_array::<UInt64Type>(None)) as ArrayRef];
    (UInt64Chunked::new_from_chunks("", chunks), build_hasher)
}
