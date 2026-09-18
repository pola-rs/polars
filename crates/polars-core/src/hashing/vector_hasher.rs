use std::hash::BuildHasher;

use polars_arrow::legacy::trusted_len::TrustedLenPush;
use polars_arrow::types::NativeType;
use polars_utils::aliases::PlSeedableRandomStateQuality;
use polars_utils::hashing::{_boost_hash_combine, folded_multiply};
use polars_utils::total_ord::{ToTotalOrd, TotalHash};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::*;
use crate::prelude::*;
use crate::runtime::RAYON;
use crate::series::implementations::null::NullChunked;

// See: https://github.com/tkaitchuck/aHash/blob/f9acd508bd89e7c5b2877a9510098100f9018d64/src/operations.rs#L4
const MULTIPLE: u64 = 6364136223846793005;

// Read more:
//  https://www.cockroachlabs.com/blog/vectorized-hash-joiner/
//  http://myeyesareblind.com/2017/02/06/Combine-hash-values/

pub trait VecHash {
    /// Compute the hash for all values in the array.
    fn vec_hash(
        &self,
        _random_state: PlSeedableRandomStateQuality,
        _buf: &mut Vec<u64>,
    ) -> PolarsResult<()>;

    fn vec_hash_combine(
        &self,
        _random_state: PlSeedableRandomStateQuality,
        _hashes: &mut [u64],
    ) -> PolarsResult<()>;
}

pub(crate) fn get_null_hash_value(random_state: &PlSeedableRandomStateQuality) -> u64 {
    // we just start with a large prime number and hash that twice
    // to get a constant hash value for null/None
    let first = random_state.hash_one(3188347919usize);
    random_state.hash_one(first)
}

fn insert_null_hash(
    chunks: &[PlArrayRef],
    random_state: PlSeedableRandomStateQuality,
    buf: &mut Vec<u64>,
) {
    let null_h = get_null_hash_value(&random_state);
    let hashes = buf.as_mut_slice();

    let mut offset = 0;
    chunks.iter().for_each(|arr| {
        if arr.null_count() > 0 {
            let validity = arr.validity().unwrap();

            match validity.flat_bitmap() {
                // A mask that holds one bit per element is walked by the bitmap's own iterator,
                // which shifts the bits off a word it is holding. Zipping drives it one `next`
                // at a time, which is what the representation has to be settled ahead of the
                // walk for: stepping the mask itself is cheap, matching how it is held is not.
                Some(bitmap) => bitmap
                    .iter()
                    .zip(&mut hashes[offset..])
                    .for_each(|(valid, h)| {
                        *h = [null_h, *h][valid as usize];
                    }),
                // A mask that says the same of every element says it without a bit per element,
                // and a chunk that has a null in it is a chunk that is null all the way through.
                None => {
                    debug_assert_eq!(arr.null_count(), arr.len());
                    if !validity.scalar_value().unwrap_or(false) {
                        hashes[offset..offset + arr.len()].fill(null_h);
                    }
                },
            }
        }
        offset += arr.len();
    });
}

fn numeric_vec_hash<T>(
    ca: &ChunkedArray<T>,
    random_state: PlSeedableRandomStateQuality,
    buf: &mut Vec<u64>,
) where
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
        // A chunk that repeats one value hashes it once, and that hash repeats in turn.
        if let Some(value) = arr.scalar_value_ignore_validity() {
            let hash = random_state.hash_one(value.to_total_ord());
            buf.extend(std::iter::repeat_n(hash, arr.len()));
            return;
        }

        buf.extend(
            arr.flat_values()
                .unwrap()
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
    random_state: PlSeedableRandomStateQuality,
    hashes: &mut [u64],
) where
    T: PolarsNumericType,
    T::Native: TotalHash + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash,
{
    let null_h = get_null_hash_value(&random_state);

    // Inlined from ahash. This ensures we combine with the previous state. Be careful not to xor
    // the hash directly with the existing hash, it would lead to 0-hashes for 2 columns
    // containing equal values.
    fn combine(h: &mut u64, to_hash: u64) {
        *h = folded_multiply(to_hash ^ folded_multiply(*h, MULTIPLE), MULTIPLE);
    }

    let mut offset = 0;
    ca.downcast_iter().for_each(|arr| {
        let hashes = &mut hashes[offset..offset + arr.len()];
        offset += arr.len();

        // Combining reads one hash per element out of the buffer either way. Which hash that is
        // depends on how the chunk holds its values and on what its mask says of them, and both
        // are settled here rather than asked per element: a chunk that repeats a single value
        // hashes it once, and a mask that says the same of every element says it once.
        match arr.null_count() {
            // No element is null, so each one combines with the hash of the value it holds.
            0 => match arr.flat_values() {
                Some(values) => values
                    .as_slice()
                    .iter()
                    .zip(hashes)
                    .for_each(|(v, h)| combine(h, random_state.hash_one(v.to_total_ord()))),
                None => {
                    let hash = random_state.hash_one(scalar_value(arr).to_total_ord());
                    hashes.iter_mut().for_each(|h| combine(h, hash));
                },
            },
            _ => {
                let validity = arr.validity().unwrap();

                let Some(bitmap) = validity.flat_bitmap() else {
                    // A mask that says the same of every element, in a chunk that has a null in
                    // it, says every one of them is null: the values are never read.
                    debug_assert_eq!(arr.null_count(), arr.len());
                    hashes.iter_mut().for_each(|h| combine(h, null_h));
                    return;
                };

                // The mask holds a bit per element and is walked by its own iterator, which
                // shifts the bits off a word it is holding rather than loading a byte apiece.
                match arr.flat_values() {
                    Some(values) => bitmap
                        .iter()
                        .zip(values.as_slice())
                        .zip(hashes.iter_mut())
                        .for_each(|((valid, v), h)| {
                            let to_hash =
                                [null_h, random_state.hash_one(v.to_total_ord())][valid as usize];
                            combine(h, to_hash);
                        }),
                    None => {
                        let hash = random_state.hash_one(scalar_value(arr).to_total_ord());
                        bitmap
                            .iter()
                            .zip(hashes.iter_mut())
                            .for_each(|(valid, h)| combine(h, [null_h, hash][valid as usize]));
                    },
                }
            },
        }
    });
}

/// The one value a chunk whose values are not flat repeats.
#[inline]
fn scalar_value<N: NativeType>(arr: &PlPrimitiveArray<N>) -> N {
    arr.scalar_value_ignore_validity()
        .expect("the values are not flat")
}

macro_rules! vec_hash_numeric {
    ($ca:ident) => {
        impl VecHash for $ca {
            fn vec_hash(
                &self,
                random_state: PlSeedableRandomStateQuality,
                buf: &mut Vec<u64>,
            ) -> PolarsResult<()> {
                numeric_vec_hash(self, random_state, buf);
                Ok(())
            }

            fn vec_hash_combine(
                &self,
                random_state: PlSeedableRandomStateQuality,
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
#[cfg(feature = "dtype-f16")]
vec_hash_numeric!(Float16Chunked);
#[cfg(feature = "dtype-u128")]
vec_hash_numeric!(UInt128Chunked);
#[cfg(any(feature = "dtype-decimal", feature = "dtype-i128"))]
vec_hash_numeric!(Int128Chunked);

impl VecHash for StringChunked {
    fn vec_hash(
        &self,
        random_state: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        self.as_binary().vec_hash(random_state, buf)?;
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlSeedableRandomStateQuality,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        self.as_binary().vec_hash_combine(random_state, hashes)?;
        Ok(())
    }
}

fn hash_binary_array(
    arr: &PlBinaryArray,
    random_state: PlSeedableRandomStateQuality,
    buf: &mut Vec<u64>,
) {
    let null_h = get_null_hash_value(&random_state);
    if arr.null_count() == 0 {
        // use the null_hash as seed to get a hash determined by `random_state` that is passed
        buf.extend(arr.values_iter().map(|v| xxh3_64_with_seed(v, null_h)))
    } else {
        buf.extend(arr.iter().map(|opt_v| match opt_v {
            Some(v) => xxh3_64_with_seed(v, null_h),
            None => null_h,
        }))
    }
}

fn hash_binview_array(
    arr: &PlBinaryViewArray,
    random_state: PlSeedableRandomStateQuality,
    buf: &mut Vec<u64>,
) {
    let null_h = get_null_hash_value(&random_state);
    if arr.null_count() == 0 {
        // use the null_hash as seed to get a hash determined by `random_state` that is passed
        buf.extend(arr.values_iter().map(|v| xxh3_64_with_seed(v, null_h)))
    } else {
        buf.extend(arr.iter().map(|opt_v| match opt_v {
            Some(v) => xxh3_64_with_seed(v, null_h),
            None => null_h,
        }))
    }
}

impl VecHash for BinaryChunked {
    fn vec_hash(
        &self,
        random_state: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        buf.clear();
        buf.reserve(self.len());
        self.downcast_iter()
            .for_each(|arr| hash_binview_array(arr, random_state.clone(), buf));
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlSeedableRandomStateQuality,
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
                    arr.validity()
                        .unwrap()
                        .iter()
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
    fn vec_hash(
        &self,
        random_state: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        buf.clear();
        buf.reserve(self.len());
        self.downcast_iter()
            .for_each(|arr| hash_binary_array(arr, random_state.clone(), buf));
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlSeedableRandomStateQuality,
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
                    arr.validity()
                        .unwrap()
                        .iter()
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
    fn vec_hash(
        &self,
        random_state: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        let null_h = get_null_hash_value(&random_state);
        buf.clear();
        buf.resize(self.len(), null_h);
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        random_state: PlSeedableRandomStateQuality,
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
    fn vec_hash(
        &self,
        random_state: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        buf.clear();
        buf.reserve(self.len());
        let true_h = random_state.hash_one(true);
        let false_h = random_state.hash_one(false);
        let null_h = get_null_hash_value(&random_state);
        self.downcast_iter().for_each(|arr| {
            if arr.null_count() == 0 {
                // A chunk holding a bit per element is read by the bitmap's own iterator, which
                // shifts the bits off a word it holds and knows how many are left: the values go
                // straight into the buffer. Its own `values_iter()` would answer each bit from a
                // reader it has to match the representation of once an element, and `extend`
                // would check the capacity as often.
                match arr.flat_values() {
                    Some(values) => buf.extend_trusted_len(
                        values.iter().map(|v| if v { true_h } else { false_h }),
                    ),
                    // Every element is the same one, so it is hashed once and that hash repeats.
                    None => {
                        let h = if arr.values().scalar_value().unwrap_or(false) {
                            true_h
                        } else {
                            false_h
                        };
                        buf.extend(std::iter::repeat_n(h, arr.len()))
                    },
                }
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
        random_state: PlSeedableRandomStateQuality,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        let true_h = random_state.hash_one(true);
        let false_h = random_state.hash_one(false);
        let null_h = get_null_hash_value(&random_state);

        let mut offset = 0;
        self.downcast_iter().for_each(|arr| {
            // See `vec_hash`: which of the three hashes each element combines with is settled by
            // how the chunk holds its values, and that is asked once here rather than per bit.
            match arr.null_count() {
                0 => match arr.flat_values() {
                    Some(values) => {
                        values
                            .iter()
                            .zip(&mut hashes[offset..])
                            .for_each(|(v, h)| {
                                let l = if v { true_h } else { false_h };
                                *h = _boost_hash_combine(l, *h)
                            })
                    },
                    None => {
                        let l = if arr.values().scalar_value().unwrap_or(false) {
                            true_h
                        } else {
                            false_h
                        };
                        hashes[offset..offset + arr.len()]
                            .iter_mut()
                            .for_each(|h| *h = _boost_hash_combine(l, *h));
                    },
                },
                _ => {
                    arr.validity()
                        .unwrap()
                        .iter()
                        .zip(&mut hashes[offset..])
                        .zip(arr.values())
                        .for_each(|((valid, h), l)| {
                            let l = if valid {
                                if l { true_h } else { false_h }
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
    fn vec_hash(
        &self,
        random_state: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
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
        random_state: PlSeedableRandomStateQuality,
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
    build_hasher: Option<PlSeedableRandomStateQuality>,
) -> PolarsResult<(Vec<UInt64Chunked>, PlSeedableRandomStateQuality)> {
    let build_hasher = build_hasher.unwrap_or_default();

    let hashes = RAYON.install(|| {
        keys.into_par_iter()
            .map(|df| {
                let hb = build_hasher.clone();
                let mut hashes = vec![];
                columns_to_hashes(df, Some(hb), &mut hashes)?;
                Ok(UInt64Chunked::from_vec(PlSmallStr::EMPTY, hashes))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    Ok((hashes, build_hasher))
}

pub fn columns_to_hashes(
    keys: &DataFrame,
    build_hasher: Option<PlSeedableRandomStateQuality>,
    hashes: &mut Vec<u64>,
) -> PolarsResult<PlSeedableRandomStateQuality> {
    let build_hasher = build_hasher.unwrap_or_default();

    if keys.width() == 0 {
        let null_h = get_null_hash_value(&build_hasher);

        for _ in 0..keys.height() {
            hashes.push(null_h);
        }

        return Ok(build_hasher);
    }

    let mut iter = keys.columns().iter();
    let first = iter.next().expect("at least one key");
    first.vec_hash(build_hasher.clone(), hashes)?;

    for keys in iter {
        keys.vec_hash_combine(build_hasher.clone(), hashes)?;
    }

    Ok(build_hasher)
}
