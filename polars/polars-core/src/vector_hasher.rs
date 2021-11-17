use crate::datatypes::UInt64Chunked;
use crate::prelude::*;
use crate::utils::arrow::array::Array;
use crate::POOL;
use ahash::{CallHasher, RandomState};
use arrow::bitmap::utils::get_bit_unchecked;
use hashbrown::{hash_map::RawEntryMut, HashMap};
use polars_arrow::utils::CustomIterTools;
use rayon::prelude::*;
use std::convert::TryInto;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

// Read more:
//  https://www.cockroachlabs.com/blog/vectorized-hash-joiner/
//  http://myeyesareblind.com/2017/02/06/Combine-hash-values/

pub trait VecHash {
    /// Compute the hash for all values in the array.
    ///
    /// This currently only works with the AHash RandomState hasher builder.
    fn vec_hash(&self, _random_state: RandomState) -> AlignedVec<u64> {
        unimplemented!()
    }

    fn vec_hash_combine(&self, _random_state: RandomState, _hashes: &mut [u64]) {
        unimplemented!()
    }
}

pub(crate) fn get_null_hash_value(random_state: RandomState) -> u64 {
    // we just start with a large prime number and hash that twice
    // to get a constant hash value for null/None
    let mut hasher = random_state.build_hasher();
    3188347919usize.hash(&mut hasher);
    let first = hasher.finish();
    let mut hasher = random_state.build_hasher();
    first.hash(&mut hasher);
    hasher.finish()
}

impl<T> VecHash for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash + CallHasher,
{
    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        // Note that we don't use the no null branch! This can break in unexpected ways.
        // for instance with threading we split an array in n_threads, this may lead to
        // splits that have no nulls and splits that have nulls. Then one array is hashed with
        // Option<T> and the other array with T.
        // Meaning that they cannot be compared. By always hashing on Option<T> the random_state is
        // the only deterministic seed.
        let mut av = AlignedVec::with_capacity(self.len());

        self.downcast_iter().for_each(|arr| {
            av.extend(
                arr.values()
                    .as_slice()
                    .iter()
                    .map(|v| T::Native::get_hash(v, &random_state)),
            );
        });

        let null_h = get_null_hash_value(random_state);
        let hashes = av.as_mut_slice();

        let mut offset = 0;
        self.downcast_iter().for_each(|arr| {
            if let Some(validity) = arr.validity() {
                let slice = validity.as_slice().0;
                (0..validity.len())
                    .map(|i| unsafe { get_bit_unchecked(slice, i) })
                    .zip(&mut hashes[offset..])
                    .for_each(|(valid, h)| {
                        if !valid {
                            *h = null_h;
                        }
                    })
            }
            offset += arr.len();
        });

        av
    }

    fn vec_hash_combine(&self, random_state: RandomState, hashes: &mut [u64]) {
        let null_h = get_null_hash_value(random_state.clone());

        let mut offset = 0;
        self.downcast_iter().for_each(|arr| {
            match arr.null_count() {
                0 => arr
                    .values()
                    .as_slice()
                    .iter()
                    .zip(&mut hashes[offset..])
                    .for_each(|(v, h)| {
                        let l = T::Native::get_hash(v, &random_state);
                        *h = boost_hash_combine(l, *h)
                    }),
                _ => arr
                    .iter()
                    .zip(&mut hashes[offset..])
                    .for_each(|(opt_v, h)| match opt_v {
                        Some(v) => {
                            let l = T::Native::get_hash(v, &random_state);
                            *h = boost_hash_combine(l, *h)
                        }
                        None => {
                            *h = boost_hash_combine(null_h, *h);
                        }
                    }),
            }
            offset += arr.len();
        });
    }
}

impl VecHash for Utf8Chunked {
    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        let null_h = get_null_hash_value(random_state.clone());
        let mut av = AlignedVec::with_capacity(self.len());
        self.downcast_iter().for_each(|arr| {
            av.extend(arr.into_iter().map(|opt_v| match opt_v {
                Some(v) => str::get_hash(v, &random_state),
                None => null_h,
            }))
        });
        av
    }

    fn vec_hash_combine(&self, random_state: RandomState, hashes: &mut [u64]) {
        let null_h = get_null_hash_value(random_state.clone());
        self.apply_to_slice(
            |opt_v, h| {
                let l = match opt_v {
                    Some(v) => str::get_hash(v, &random_state),
                    None => null_h,
                };
                boost_hash_combine(l, *h)
            },
            hashes,
        )
    }
}

impl VecHash for BooleanChunked {
    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        let mut av = AlignedVec::with_capacity(self.len());
        self.downcast_iter().for_each(|arr| {
            av.extend(arr.into_iter().map(|opt_v| {
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                hasher.finish()
            }))
        });
        av
    }

    fn vec_hash_combine(&self, random_state: RandomState, hashes: &mut [u64]) {
        self.apply_to_slice(
            |opt_v, h| {
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                boost_hash_combine(hasher.finish(), *h)
            },
            hashes,
        )
    }
}

impl VecHash for Float32Chunked {
    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        self.bit_repr_small().vec_hash(random_state)
    }

    fn vec_hash_combine(&self, random_state: RandomState, hashes: &mut [u64]) {
        self.bit_repr_small().vec_hash_combine(random_state, hashes)
    }
}
impl VecHash for Float64Chunked {
    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        self.bit_repr_large().vec_hash(random_state)
    }
    fn vec_hash_combine(&self, random_state: RandomState, hashes: &mut [u64]) {
        self.bit_repr_large().vec_hash_combine(random_state, hashes)
    }
}

impl VecHash for ListChunked {}

#[cfg(feature = "object")]
impl<T> VecHash for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        // Note that we don't use the no null branch! This can break in unexpected ways.
        // for instance with threading we split an array in n_threads, this may lead to
        // splits that have no nulls and splits that have nulls. Then one array is hashed with
        // Option<T> and the other array with T.
        // Meaning that they cannot be compared. By always hashing on Option<T> the random_state is
        // the only deterministic seed.
        let mut av = AlignedVec::with_capacity(self.len());

        self.downcast_iter().for_each(|arr| {
            av.extend(arr.into_iter().map(|opt_v| {
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                hasher.finish()
            }))
        });
        av
    }

    fn vec_hash_combine(&self, random_state: RandomState, hashes: &mut [u64]) {
        self.apply_to_slice(
            |opt_v, h| {
                let mut hasher = random_state.build_hasher();
                opt_v.hash(&mut hasher);
                boost_hash_combine(hasher.finish(), *h)
            },
            hashes,
        )
    }
}

// Used to to get a u64 from the hashing keys
// We need to modify the hashing algorithm to use the hash for this and only compute the hash once.
pub(crate) trait AsU64 {
    #[allow(clippy::wrong_self_convention)]
    fn as_u64(self) -> u64;
}

impl AsU64 for u32 {
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl AsU64 for u64 {
    fn as_u64(self) -> u64 {
        self
    }
}

impl AsU64 for Option<u32> {
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v as u64,
            // just a number safe from overflow
            None => u64::MAX >> 2,
        }
    }
}

impl AsU64 for Option<u64> {
    fn as_u64(self) -> u64 {
        self.unwrap_or(u64::MAX >> 2)
    }
}

impl AsU64 for [u8; 9] {
    fn as_u64(self) -> u64 {
        // the last byte includes the null information.
        // that one is skipped. Worst thing that could happen is unbalanced partition.
        u64::from_ne_bytes(self[..8].try_into().unwrap())
    }
}
const BUILD_HASHER: RandomState = RandomState::with_seeds(0, 0, 0, 0);
impl AsU64 for [u8; 17] {
    fn as_u64(self) -> u64 {
        <[u8]>::get_hash(&self, &BUILD_HASHER)
    }
}

impl AsU64 for [u8; 13] {
    fn as_u64(self) -> u64 {
        <[u8]>::get_hash(&self, &BUILD_HASHER)
    }
}

#[derive(Default)]
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
    fn as_u64(self) -> u64 {
        self.hash
    }
}

#[inline]
/// For partitions that are a power of 2 we can use a bitshift instead of a modulo.
pub(crate) fn this_partition(h: u64, thread_no: u64, n_partitions: u64) -> bool {
    // n % 2^i = n & (2^i - 1)
    (h.wrapping_add(thread_no)) & n_partitions.wrapping_sub(1) == 0
}

pub(crate) fn prepare_hashed_relation_threaded<T, I>(
    iters: Vec<I>,
) -> Vec<HashMap<T, (bool, Vec<u32>), RandomState>>
where
    I: Iterator<Item = T> + Send + TrustedLen,
    T: Send + Hash + Eq + Sync + Copy,
{
    let n_partitions = iters.len();
    assert!(n_partitions.is_power_of_two());

    let (hashes_and_keys, build_hasher) = create_hash_and_keys_threaded_vectorized(iters, None);

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|partition_no| {
            let build_hasher = build_hasher.clone();
            let hashes_and_keys = &hashes_and_keys;
            let partition_no = partition_no as u64;
            let mut hash_tbl: HashMap<T, (bool, Vec<u32>), RandomState> =
                HashMap::with_hasher(build_hasher);

            let n_threads = n_partitions as u64;
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
                        if this_partition(*h, partition_no, n_threads) {
                            let idx = idx + offset;
                            let entry = hash_tbl
                                .raw_entry_mut()
                                // uses the key to check equality to find and entry
                                .from_key_hashed_nocheck(*h, k);

                            match entry {
                                RawEntryMut::Vacant(entry) => {
                                    entry.insert_hashed_nocheck(*h, *k, (false, vec![idx]));
                                }
                                RawEntryMut::Occupied(mut entry) => {
                                    let (_k, v) = entry.get_key_value_mut();
                                    v.1.push(idx);
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
    I::IntoIter: TrustedLen,
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
                    .collect_trusted::<Vec<_>>()
            })
            .collect()
    });
    (hashes, build_hasher)
}

// hash combine from c++' boost lib
#[inline]
pub(crate) fn boost_hash_combine(l: u64, r: u64) -> u64 {
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

    let mut iter = keys.iter();
    let first = iter.next().expect("at least one key");
    let mut hashes = first.vec_hash(build_hasher.clone());
    let hslice = hashes.as_mut_slice();

    for keys in iter {
        keys.vec_hash_combine(build_hasher.clone(), hslice);
    }

    let chunks = vec![Arc::new(PrimitiveArray::from_data(
        ArrowDataType::UInt64,
        hashes.into(),
        None,
    )) as ArrayRef];
    (UInt64Chunked::new_from_chunks("", chunks), build_hasher)
}
