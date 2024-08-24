use arrow::bitmap::utils::get_bit_unchecked;
use polars_utils::total_ord::{ToTotalOrd, TotalHash};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::*;
use crate::prelude::*;
use crate::series::implementations::null::NullChunked;
use crate::POOL;

// See: https://github.com/tkaitchuck/aHash/blob/f9acd508bd89e7c5b2877a9510098100f9018d64/src/operations.rs#L4
const MULTIPLE: u64 = 6364136223846793005;

// Read more:
//  https://www.cockroachlabs.com/blog/vectorized-hash-joiner/
//  http://myeyesareblind.com/2017/02/06/Combine-hash-values/

pub trait VecHash {
    /// Compute the hash for all values in the array.
    fn vec_hash(&self, _random_state: PlRandomState, _buf: &mut Vec<u64>) -> PolarsResult<()> {
        polars_bail!(un_impl = vec_hash);
    }

    fn vec_hash_combine(
        &self,
        _random_state: PlRandomState,
        _hashes: &mut [u64],
    ) -> PolarsResult<()> {
        polars_bail!(un_impl = vec_hash_combine);
    }
}

pub(crate) const fn folded_multiply(s: u64, by: u64) -> u64 {
    let result = (s as u128).wrapping_mul(by as u128);
    ((result & 0xffff_ffff_ffff_ffff) as u64) ^ ((result >> 64) as u64)
}

pub(crate) fn get_null_hash_value(random_state: &PlRandomState) -> u64 {
    // we just start with a large prime number and hash that twice
    // to get a constant hash value for null/None
    let first = random_state.hash_one(3188347919usize);
    random_state.hash_one(first)
}

fn insert_null_hash(chunks: &[ArrayRef], random_state: PlRandomState, buf: &mut Vec<u64>) {
    let null_h = get_null_hash_value(&random_state);
    let hashes = buf.as_mut_slice();

    let mut offset = 0;
    chunks.iter().for_each(|arr| {
        if arr.null_count() > 0 {
            let validity = arr.validity().unwrap();
            let (slice, byte_offset, _) = validity.as_slice();
            (0..validity.len())
                .map(|i| unsafe { get_bit_unchecked(slice, i + byte_offset) })
                .zip(&mut hashes[offset..])
                .for_each(|(valid, h)| {
                    *h = [null_h, *h][valid as usize];
                })
        }
        offset += arr.len();
    });
}

fn numeric_vec_hash<T>(ca: &ChunkedArray<T>, random_state: PlRandomState, buf: &mut Vec<u64>)
where
    T: PolarsNumericType,
    T::Native: TotalHash + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash,
{
    // Note that we don't use the no null branch! This can break in unexpected ways.
    // for instance with threading we split an array in n_threads, this may lead to
    // splits that have no nulls and splits that have nulls. Then one array is hashed with
    // Option<T> and the other array with T.
    // Meaning that they cannot be compared. By always hashing on Option<T> the random_state is
    // the only deterministic seed.
    buf.clear();
    buf.reserve(ca.len());

    #[allow(unused_unsafe)]
    #[allow(clippy::useless_transmute)]
    ca.downcast_iter().for_each(|arr| {
        buf.extend(
            arr.values()
                .as_slice()
                .iter()
                .copied()
                .map(|v| random_state.hash_one(v.to_total_ord())),
        );
    });
    insert_null_hash(&ca.chunks, random_state, buf)
}

fn numeric_vec_hash_combine<T>(
    ca: &ChunkedArray<T>,
    random_state: PlRandomState,
    hashes: &mut [u64],
) where
    T: PolarsNumericType,
    T::Native: TotalHash + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash,
{
    let null_h = get_null_hash_value(&random_state);

    let mut offset = 0;
    ca.downcast_iter().for_each(|arr| {
        match arr.null_count() {
            0 => arr
                .values()
                .as_slice()
                .iter()
                .zip(&mut hashes[offset..])
                .for_each(|(v, h)| {
                    // Inlined from ahash. This ensures we combine with the previous state.
                    *h = folded_multiply(
                        // Be careful not to xor the hash directly with the existing hash,
                        // it would lead to 0-hashes for 2 columns containing equal values.
                        random_state.hash_one(v.to_total_ord()) ^ folded_multiply(*h, MULTIPLE),
                        MULTIPLE,
                    );
                }),
            _ => {
                let validity = arr.validity().unwrap();
                let (slice, byte_offset, _) = validity.as_slice();
                (0..validity.len())
                    .map(|i| unsafe { get_bit_unchecked(slice, i + byte_offset) })
                    .zip(&mut hashes[offset..])
                    .zip(arr.values().as_slice())
                    .for_each(|((valid, h), l)| {
                        let lh = random_state.hash_one(l.to_total_ord());
                        let to_hash = [null_h, lh][valid as usize];
                        *h = folded_multiply(to_hash ^ folded_multiply(*h, MULTIPLE), MULTIPLE);
                    });
            },
        }
        offset += arr.len();
    });
}

macro_rules! vec_hash_numeric {
    ($ca:ident) => {
        impl VecHash for $ca {
            fn vec_hash(
                &self,
                random_state: PlRandomState,
                buf: &mut Vec<u64>,
            ) -> PolarsResult<()> {
                numeric_vec_hash(self, random_state, buf);
                Ok(())
            }

            fn vec_hash_combine(
                &self,
                random_state: PlRandomState,
                hashes: &mut [u64],
            ) -> PolarsResult<()> {
                numeric_vec_hash_combine(self, random_state, hashes);
                Ok(())
            }
        }
    };
}

vec_hash_numeric!(Int64Chunked);
vec_hash_numeric!(Int32Chunked);
vec_hash_numeric!(Int16Chunked);
vec_hash_numeric!(Int8Chunked);
vec_hash_numeric!(UInt64Chunked);
vec_hash_numeric!(UInt32Chunked);
vec_hash_numeric!(UInt16Chunked);
vec_hash_numeric!(UInt8Chunked);
vec_hash_numeric!(Float64Chunked);
vec_hash_numeric!(Float32Chunked);
#[cfg(feature = "dtype-decimal")]
vec_hash_numeric!(Int128Chunked);

impl VecHash for StringChunked {
    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.as_binary().vec_hash(random_state, buf)?;
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        self.as_binary().vec_hash_combine(random_state, hashes)?;
        Ok(())
    }
}

// used in polars-pipe
pub fn _hash_binary_array(arr: &BinaryArray<i64>, random_state: PlRandomState, buf: &mut Vec<u64>) {
    let null_h = get_null_hash_value(&random_state);
    if arr.null_count() == 0 {
        // use the null_hash as seed to get a hash determined by `random_state` that is passed
        buf.extend(arr.values_iter().map(|v| xxh3_64_with_seed(v, null_h)))
    } else {
        buf.extend(arr.into_iter().map(|opt_v| match opt_v {
            Some(v) => xxh3_64_with_seed(v, null_h),
            None => null_h,
        }))
    }
}

fn hash_binview_array(arr: &BinaryViewArray, random_state: PlRandomState, buf: &mut Vec<u64>) {
    let null_h = get_null_hash_value(&random_state);
    if arr.null_count() == 0 {
        // use the null_hash as seed to get a hash determined by `random_state` that is passed
        buf.extend(arr.values_iter().map(|v| xxh3_64_with_seed(v, null_h)))
    } else {
        buf.extend(arr.into_iter().map(|opt_v| match opt_v {
            Some(v) => xxh3_64_with_seed(v, null_h),
            None => null_h,
        }))
    }
}

impl VecHash for BinaryChunked {
    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        buf.clear();
        buf.reserve(self.len());
        self.downcast_iter()
            .for_each(|arr| hash_binview_array(arr, random_state.clone(), buf));
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        let null_h = get_null_hash_value(&random_state);

        let mut offset = 0;
        self.downcast_iter().for_each(|arr| {
            match arr.null_count() {
                0 => arr
                    .values_iter()
                    .zip(&mut hashes[offset..])
                    .for_each(|(v, h)| {
                        let l = xxh3_64_with_seed(v, null_h);
                        *h = _boost_hash_combine(l, *h)
                    }),
                _ => {
                    let validity = arr.validity().unwrap();
                    let (slice, byte_offset, _) = validity.as_slice();
                    (0..validity.len())
                        .map(|i| unsafe { get_bit_unchecked(slice, i + byte_offset) })
                        .zip(&mut hashes[offset..])
                        .zip(arr.values_iter())
                        .for_each(|((valid, h), l)| {
                            let l = if valid {
                                xxh3_64_with_seed(l, null_h)
                            } else {
                                null_h
                            };
                            *h = _boost_hash_combine(l, *h)
                        });
                },
            }
            offset += arr.len();
        });
        Ok(())
    }
}

impl VecHash for BinaryOffsetChunked {
    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        buf.clear();
        buf.reserve(self.len());
        self.downcast_iter()
            .for_each(|arr| _hash_binary_array(arr, random_state.clone(), buf));
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        let null_h = get_null_hash_value(&random_state);

        let mut offset = 0;
        self.downcast_iter().for_each(|arr| {
            match arr.null_count() {
                0 => arr
                    .values_iter()
                    .zip(&mut hashes[offset..])
                    .for_each(|(v, h)| {
                        let l = xxh3_64_with_seed(v, null_h);
                        *h = _boost_hash_combine(l, *h)
                    }),
                _ => {
                    let validity = arr.validity().unwrap();
                    let (slice, byte_offset, _) = validity.as_slice();
                    (0..validity.len())
                        .map(|i| unsafe { get_bit_unchecked(slice, i + byte_offset) })
                        .zip(&mut hashes[offset..])
                        .zip(arr.values_iter())
                        .for_each(|((valid, h), l)| {
                            let l = if valid {
                                xxh3_64_with_seed(l, null_h)
                            } else {
                                null_h
                            };
                            *h = _boost_hash_combine(l, *h)
                        });
                },
            }
            offset += arr.len();
        });
        Ok(())
    }
}

impl VecHash for NullChunked {
    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        let null_h = get_null_hash_value(&random_state);
        buf.clear();
        buf.resize(self.len(), null_h);
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        let null_h = get_null_hash_value(&random_state);
        hashes
            .iter_mut()
            .for_each(|h| *h = _boost_hash_combine(null_h, *h));
        Ok(())
    }
}
impl VecHash for BooleanChunked {
    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        buf.clear();
        buf.reserve(self.len());
        let true_h = random_state.hash_one(true);
        let false_h = random_state.hash_one(false);
        let null_h = get_null_hash_value(&random_state);
        self.downcast_iter().for_each(|arr| {
            if arr.null_count() == 0 {
                buf.extend(arr.values_iter().map(|v| if v { true_h } else { false_h }))
            } else {
                buf.extend(arr.into_iter().map(|opt_v| match opt_v {
                    Some(true) => true_h,
                    Some(false) => false_h,
                    None => null_h,
                }))
            }
        });
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        let true_h = random_state.hash_one(true);
        let false_h = random_state.hash_one(false);
        let null_h = get_null_hash_value(&random_state);

        let mut offset = 0;
        self.downcast_iter().for_each(|arr| {
            match arr.null_count() {
                0 => arr
                    .values_iter()
                    .zip(&mut hashes[offset..])
                    .for_each(|(v, h)| {
                        let l = if v { true_h } else { false_h };
                        *h = _boost_hash_combine(l, *h)
                    }),
                _ => {
                    let validity = arr.validity().unwrap();
                    let (slice, byte_offset, _) = validity.as_slice();
                    (0..validity.len())
                        .map(|i| unsafe { get_bit_unchecked(slice, i + byte_offset) })
                        .zip(&mut hashes[offset..])
                        .zip(arr.values())
                        .for_each(|((valid, h), l)| {
                            let l = if valid {
                                if l {
                                    true_h
                                } else {
                                    false_h
                                }
                            } else {
                                null_h
                            };
                            *h = _boost_hash_combine(l, *h)
                        });
                },
            }
            offset += arr.len();
        });
        Ok(())
    }
}

#[cfg(feature = "object")]
impl<T> VecHash for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        // Note that we don't use the no null branch! This can break in unexpected ways.
        // for instance with threading we split an array in n_threads, this may lead to
        // splits that have no nulls and splits that have nulls. Then one array is hashed with
        // Option<T> and the other array with T.
        // Meaning that they cannot be compared. By always hashing on Option<T> the random_state is
        // the only deterministic seed.
        buf.clear();
        buf.reserve(self.len());

        self.downcast_iter()
            .for_each(|arr| buf.extend(arr.into_iter().map(|opt_v| random_state.hash_one(opt_v))));

        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        self.apply_to_slice(
            |opt_v, h| {
                let hashed = random_state.hash_one(opt_v);
                _boost_hash_combine(hashed, *h)
            },
            hashes,
        );
        Ok(())
    }
}

pub fn _df_rows_to_hashes_threaded_vertical(
    keys: &[DataFrame],
    hasher_builder: Option<PlRandomState>,
) -> PolarsResult<(Vec<UInt64Chunked>, PlRandomState)> {
    let hasher_builder = hasher_builder.unwrap_or_default();

    let hashes = POOL.install(|| {
        keys.into_par_iter()
            .map(|df| {
                let hb = hasher_builder.clone();
                let mut hashes = vec![];
                series_to_hashes(df.get_columns(), Some(hb), &mut hashes)?;
                Ok(UInt64Chunked::from_vec("", hashes))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    Ok((hashes, hasher_builder))
}

pub(crate) fn series_to_hashes(
    keys: &[Series],
    build_hasher: Option<PlRandomState>,
    hashes: &mut Vec<u64>,
) -> PolarsResult<PlRandomState> {
    let build_hasher = build_hasher.unwrap_or_default();

    let mut iter = keys.iter();
    let first = iter.next().expect("at least one key");
    first.vec_hash(build_hasher.clone(), hashes)?;

    for keys in iter {
        keys.vec_hash_combine(build_hasher.clone(), hashes)?;
    }

    Ok(build_hasher)
}
