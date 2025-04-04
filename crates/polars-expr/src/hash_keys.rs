#![allow(unsafe_op_in_unsafe_fn)]
use std::hash::BuildHasher;

use arrow::array::{Array, BinaryArray, BinaryViewArray, PrimitiveArray, StaticArray, UInt64Array};
use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and_many;
use polars_core::error::polars_err;
use polars_core::frame::DataFrame;
use polars_core::prelude::row_encode::_get_rows_encoded_unordered;
use polars_core::prelude::{ChunkedArray, DataType, PlRandomState, PolarsDataType};
use polars_core::series::Series;
use polars_utils::IdxSize;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::hashing::HashPartitioner;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::{BuildHasherTotalExt, TotalHash};
use polars_utils::vec::PushUnchecked;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum HashKeysVariant {
    RowEncoded,
    Single,
    Binview,
}

pub fn hash_keys_variant_for_dtype(dt: &DataType) -> HashKeysVariant {
    match dt {
        dt if dt.is_primitive_numeric() | dt.is_temporal() => HashKeysVariant::Single,

        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => HashKeysVariant::Single,
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(_, _) => HashKeysVariant::Single,

        DataType::String | DataType::Binary => HashKeysVariant::Binview,

        // TODO: more efficient encoding for these.
        DataType::Boolean | DataType::Null => HashKeysVariant::RowEncoded,

        _ => HashKeysVariant::RowEncoded,
    }
}

macro_rules! downcast_single_key_ca {
    (
        $self:expr, | $ca:ident | $($body:tt)*
    ) => {{
        #[allow(unused_imports)]
        use polars_core::datatypes::DataType::*;
        match $self.dtype() {
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => { let $ca = $self.i8().unwrap(); $($body)* },
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => { let $ca = $self.i16().unwrap(); $($body)* },
            DataType::Int32 => { let $ca = $self.i32().unwrap(); $($body)* },
            DataType::Int64 => { let $ca = $self.i64().unwrap(); $($body)* },
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => { let $ca = $self.u8().unwrap(); $($body)* },
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => { let $ca = $self.u16().unwrap(); $($body)* },
            DataType::UInt32 => { let $ca = $self.u32().unwrap(); $($body)* },
            DataType::UInt64 => { let $ca = $self.u64().unwrap(); $($body)* },
            #[cfg(feature = "dtype-i128")]
            DataType::Int128 => { let $ca = $self.i128().unwrap(); $($body)* },
            DataType::Float32 => { let $ca = $self.f32().unwrap(); $($body)* },
            DataType::Float64 => { let $ca = $self.f64().unwrap(); $($body)* },

            #[cfg(feature = "dtype-date")]
            DataType::Date => { let $ca = $self.date().unwrap(); $($body)* },
            #[cfg(feature = "dtype-time")]
            DataType::Time => { let $ca = $self.time().unwrap(); $($body)* },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(..) => { let $ca = $self.datetime().unwrap(); $($body)* },
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(..) => { let $ca = $self.duration().unwrap(); $($body)* },

            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(..) => { let $ca = $self.decimal().unwrap(); $($body)* },
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(..) => { let $ca = $self.categorical().unwrap().physical(); $($body)* },

            _ => unreachable!(),
        }
    }}
}
pub(crate) use downcast_single_key_ca;

/// Represents a DataFrame plus a hash per row, intended for keys in grouping
/// or joining. The hashes may or may not actually be physically pre-computed,
/// this depends per type.
#[derive(Clone, Debug)]
pub enum HashKeys {
    RowEncoded(RowEncodedKeys),
    Binview(BinviewKeys),
    Single(SingleKeys),
}

impl HashKeys {
    pub fn from_df(
        df: &DataFrame,
        random_state: PlRandomState,
        null_is_valid: bool,
        force_row_encoding: bool,
    ) -> Self {
        let first_col_variant = hash_keys_variant_for_dtype(df[0].dtype());
        let use_row_encoding = force_row_encoding
            || df.width() > 1
            || first_col_variant == HashKeysVariant::RowEncoded;
        if use_row_encoding {
            let keys = df.get_columns();
            #[cfg(feature = "dtype-categorical")]
            for key in keys {
                if let DataType::Categorical(Some(rev_map), _) = key.dtype() {
                    assert!(
                        rev_map.is_active_global(),
                        "{}",
                        polars_err!(string_cache_mismatch)
                    );
                }
            }
            let mut keys_encoded = _get_rows_encoded_unordered(keys).unwrap().into_array();

            if !null_is_valid {
                let validities = keys
                    .iter()
                    .map(|c| c.as_materialized_series().rechunk_validity())
                    .collect_vec();
                let combined = combine_validities_and_many(&validities);
                keys_encoded.set_validity(combined);
            }

            // TODO: use vechash? Not supported yet for lists.
            // let mut hashes = Vec::with_capacity(df.height());
            // columns_to_hashes(df.get_columns(), Some(random_state), &mut hashes).unwrap();

            let hashes = keys_encoded
                .values_iter()
                .map(|k| random_state.hash_one(k))
                .collect();
            Self::RowEncoded(RowEncodedKeys {
                hashes: PrimitiveArray::from_vec(hashes),
                keys: keys_encoded,
            })
        } else if first_col_variant == HashKeysVariant::Binview {
            let keys = if let Ok(ca_str) = df[0].str() {
                ca_str.as_binary()
            } else {
                df[0].binary().unwrap().clone()
            };
            let keys = keys.rechunk().downcast_as_array().clone();

            let hashes = if keys.has_nulls() {
                keys.iter()
                    .map(|opt_k| opt_k.map(|k| random_state.hash_one(k)).unwrap_or(0))
                    .collect()
            } else {
                keys.values_iter()
                    .map(|k| random_state.hash_one(k))
                    .collect()
            };

            Self::Binview(BinviewKeys {
                hashes: PrimitiveArray::from_vec(hashes),
                keys,
                null_is_valid,
            })
        } else {
            Self::Single(SingleKeys {
                random_state,
                keys: df[0].as_materialized_series().rechunk(),
                null_is_valid,
            })
        }
    }

    pub fn len(&self) -> usize {
        match self {
            HashKeys::RowEncoded(s) => s.keys.len(),
            HashKeys::Single(s) => s.keys.len(),
            HashKeys::Binview(s) => s.keys.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn validity(&self) -> Option<&Bitmap> {
        match self {
            HashKeys::RowEncoded(s) => s.keys.validity(),
            HashKeys::Single(s) => s.keys.chunks()[0].validity(),
            HashKeys::Binview(s) => s.keys.validity(),
        }
    }

    pub fn null_is_valid(&self) -> bool {
        match self {
            HashKeys::RowEncoded(_) => false,
            HashKeys::Single(s) => s.null_is_valid,
            HashKeys::Binview(s) => s.null_is_valid,
        }
    }

    pub fn for_each_hash<F: FnMut(Option<u64>)>(&self, f: F) {
        match self {
            HashKeys::RowEncoded(s) => s.for_each_hash(f),
            HashKeys::Single(s) => s.for_each_hash(f),
            HashKeys::Binview(s) => s.for_each_hash(f),
        }
    }

    /// After this call partitions will be extended with the partition for each
    /// hash. Nulls are assigned IdxSize::MAX or a specific partition depending
    /// on whether partition_nulls is true.
    pub fn gen_partitions(
        &self,
        partitioner: &HashPartitioner,
        partitions: &mut Vec<IdxSize>,
        partition_nulls: bool,
    ) {
        unsafe {
            // Arbitrarily put nulls in partition 0.
            let null_p = if partition_nulls | self.null_is_valid() {
                0
            } else {
                IdxSize::MAX
            };
            partitions.reserve(self.len());
            self.for_each_hash(|opt_h| {
                partitions.push_unchecked(
                    opt_h
                        .map(|h| partitioner.hash_to_partition(h) as IdxSize)
                        .unwrap_or(null_p),
                );
            });
        }
    }

    /// After this call partition_idxs[p] will be extended with the indices of
    /// hashes that belong to partition p, and the cardinality sketches are
    /// updated accordingly.
    pub fn gen_idxs_per_partition(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
        partition_nulls: bool,
    ) {
        if sketches.is_empty() {
            self.gen_idxs_per_partition_impl::<false>(
                partitioner,
                partition_idxs,
                sketches,
                partition_nulls | self.null_is_valid(),
            );
        } else {
            self.gen_idxs_per_partition_impl::<true>(
                partitioner,
                partition_idxs,
                sketches,
                partition_nulls | self.null_is_valid(),
            );
        }
    }

    fn gen_idxs_per_partition_impl<const BUILD_SKETCHES: bool>(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
        partition_nulls: bool,
    ) {
        assert!(partition_idxs.len() == partitioner.num_partitions());
        assert!(!BUILD_SKETCHES || sketches.len() == partitioner.num_partitions());

        let mut idx = 0;
        self.for_each_hash(|opt_h| {
            if let Some(h) = opt_h {
                unsafe {
                    // SAFETY: we assured the number of partitions matches.
                    let p = partitioner.hash_to_partition(h);
                    partition_idxs.get_unchecked_mut(p).push(idx);
                    if BUILD_SKETCHES {
                        sketches.get_unchecked_mut(p).insert(h);
                    }
                }
            } else if partition_nulls {
                // Arbitrarily put nulls in partition 0.
                unsafe {
                    partition_idxs.get_unchecked_mut(0).push(idx);
                }
            }
            idx += 1;
        });
    }

    pub fn sketch_cardinality(&self, sketch: &mut CardinalitySketch) {
        self.for_each_hash(|opt_h| {
            sketch.insert(opt_h.unwrap_or(0));
        })
    }
}

#[derive(Clone, Debug)]
pub struct RowEncodedKeys {
    pub hashes: UInt64Array, // Always non-null, we use the validity of keys.
    pub keys: BinaryArray<i64>,
}

impl RowEncodedKeys {
    pub fn for_each_hash<F: FnMut(Option<u64>)>(&self, mut f: F) {
        if self.keys.has_nulls() {
            let validity = self.keys.validity().unwrap();
            for (hash, is_v) in self.hashes.values_iter().zip(validity) {
                if is_v { f(Some(*hash)) } else { f(None) }
            }
        } else {
            for hash in self.hashes.values_iter() {
                f(Some(*hash))
            }
        }
    }
}

/// Single keys without prehashing.
#[derive(Clone, Debug)]
pub struct SingleKeys {
    pub random_state: PlRandomState,
    pub keys: Series,
    pub null_is_valid: bool,
}

impl SingleKeys {
    pub fn for_each_hash<F: FnMut(Option<u64>)>(&self, f: F) {
        downcast_single_key_ca!(self.keys, |keys| {
            for_each_hash(keys, f, &self.random_state);
        })
    }
}

fn for_each_hash<T, F>(keys: &ChunkedArray<T>, mut f: F, random_state: &PlRandomState)
where
    T: PolarsDataType,
    for<'a> <T as PolarsDataType>::Physical<'a>: TotalHash,
    F: FnMut(Option<u64>),
{
    if keys.has_nulls() {
        for arr in keys.downcast_iter() {
            for opt_k in arr.iter() {
                f(opt_k.map(|k| random_state.tot_hash_one(k)))
            }
        }
    } else {
        for arr in keys.downcast_iter() {
            for k in arr.values_iter() {
                f(Some(random_state.tot_hash_one(k)))
            }
        }
    }
}

/// Pre-hashed binary view keys with prehashing.
#[derive(Clone, Debug)]
pub struct BinviewKeys {
    pub hashes: UInt64Array,
    pub keys: BinaryViewArray,
    pub null_is_valid: bool,
}

impl BinviewKeys {
    pub fn for_each_hash<F: FnMut(Option<u64>)>(&self, mut f: F) {
        if self.keys.has_nulls() {
            let hash_slice = self.hashes.values().as_slice();
            if let Some(validity) = self.keys.validity() {
                for (i, v) in validity.iter().enumerate() {
                    if v {
                        f(Some(unsafe { *hash_slice.get_unchecked(i) }));
                    } else {
                        f(None);
                    }
                }
            } else {
                for h in self.hashes.values_iter() {
                    f(Some(*h))
                }
            }
        } else {
            for h in self.hashes.values_iter() {
                f(Some(*h));
            }
        }
    }
}
