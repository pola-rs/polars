#![allow(unsafe_op_in_unsafe_fn)]
use std::hash::BuildHasher;

use arrow::array::{BinaryArray, PrimitiveArray, StaticArray, UInt64Array};
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

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum HashKeysVariant {
    RowEncoded,
    Single,
}

pub fn hash_keys_variant_for_dtype(dt: &DataType) -> HashKeysVariant {
    match dt {
        dt if dt.is_primitive_numeric() | dt.is_temporal() => HashKeysVariant::Single,

        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => HashKeysVariant::Single,
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(_, _) => HashKeysVariant::Single,

        // TODO: more efficient encoding for these.
        DataType::String | DataType::Binary | DataType::Boolean | DataType::Null => {
            HashKeysVariant::RowEncoded
        },

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
    Single(SingleKeys),
}

impl HashKeys {
    pub fn from_df(
        df: &DataFrame,
        random_state: PlRandomState,
        null_is_valid: bool,
        force_row_encoding: bool,
    ) -> Self {
        let use_row_encoding = force_row_encoding
            || df.width() > 1
            || hash_keys_variant_for_dtype(df[0].dtype()) == HashKeysVariant::RowEncoded;
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
        } else {
            Self::Single(SingleKeys {
                random_state,
                keys: df[0].as_materialized_series().clone(),
                null_is_valid,
            })
        }
    }

    pub fn len(&self) -> usize {
        match self {
            HashKeys::RowEncoded(s) => s.keys.len(),
            HashKeys::Single(s) => s.keys.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        match self {
            Self::RowEncoded(s) => s.gen_partitions(partitioner, partitions, partition_nulls),
            Self::Single(s) => s.gen_partitions(partitioner, partitions, partition_nulls),
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
            match self {
                Self::RowEncoded(s) => s.gen_idxs_per_partition::<false>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
                Self::Single(s) => s.gen_idxs_per_partition::<false>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
            }
        } else {
            match self {
                Self::RowEncoded(s) => s.gen_idxs_per_partition::<true>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
                Self::Single(s) => s.gen_idxs_per_partition::<true>(
                    partitioner,
                    partition_idxs,
                    sketches,
                    partition_nulls,
                ),
            }
        }
    }

    pub fn sketch_cardinality(&self, sketch: &mut CardinalitySketch) {
        match self {
            HashKeys::RowEncoded(s) => s.sketch_cardinality(sketch),
            HashKeys::Single(s) => s.sketch_cardinality(sketch),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RowEncodedKeys {
    pub hashes: UInt64Array,
    pub keys: BinaryArray<i64>,
}

impl RowEncodedKeys {
    pub fn gen_partitions(
        &self,
        partitioner: &HashPartitioner,
        partitions: &mut Vec<IdxSize>,
        partition_nulls: bool,
    ) {
        partitions.reserve(self.hashes.len());
        if let Some(validity) = self.keys.validity() {
            // Arbitrarily put nulls in partition 0.
            let null_p = if partition_nulls { 0 } else { IdxSize::MAX };
            partitions.extend(self.hashes.values_iter().zip(validity).map(|(h, is_v)| {
                if is_v {
                    partitioner.hash_to_partition(*h) as IdxSize
                } else {
                    null_p
                }
            }))
        } else {
            partitions.extend(
                self.hashes
                    .values_iter()
                    .map(|h| partitioner.hash_to_partition(*h) as IdxSize),
            )
        }
    }

    pub fn gen_idxs_per_partition<const BUILD_SKETCHES: bool>(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
        partition_nulls: bool,
    ) {
        assert!(partition_idxs.len() == partitioner.num_partitions());
        assert!(!BUILD_SKETCHES || sketches.len() == partitioner.num_partitions());

        if let Some(validity) = self.keys.validity() {
            for (i, (h, is_v)) in self.hashes.values_iter().zip(validity).enumerate() {
                if is_v {
                    unsafe {
                        // SAFETY: we assured the number of partitions matches.
                        let p = partitioner.hash_to_partition(*h);
                        partition_idxs.get_unchecked_mut(p).push(i as IdxSize);
                        if BUILD_SKETCHES {
                            sketches.get_unchecked_mut(p).insert(*h);
                        }
                    }
                } else if partition_nulls {
                    // Arbitrarily put nulls in partition 0.
                    unsafe {
                        partition_idxs.get_unchecked_mut(0).push(i as IdxSize);
                    }
                }
            }
        } else {
            for (i, h) in self.hashes.values_iter().enumerate() {
                unsafe {
                    // SAFETY: we assured the number of partitions matches.
                    let p = partitioner.hash_to_partition(*h);
                    partition_idxs.get_unchecked_mut(p).push(i as IdxSize);
                    if BUILD_SKETCHES {
                        sketches.get_unchecked_mut(p).insert(*h);
                    }
                }
            }
        }
    }

    pub fn sketch_cardinality(&self, sketch: &mut CardinalitySketch) {
        if let Some(validity) = self.keys.validity() {
            for (h, is_v) in self.hashes.values_iter().zip(validity) {
                if is_v {
                    sketch.insert(*h);
                }
            }
        } else {
            for h in self.hashes.values_iter() {
                sketch.insert(*h);
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
    #[allow(clippy::ptr_arg)] // Remove when implemented.
    pub fn gen_partitions(
        &self,
        partitioner: &HashPartitioner,
        partitions: &mut Vec<IdxSize>,
        partition_nulls: bool,
    ) {
        downcast_single_key_ca!(self.keys, |keys| {
            gen_partitions(
                keys,
                &self.random_state,
                partitioner,
                partitions,
                partition_nulls | self.null_is_valid,
            );
        });
    }

    pub fn gen_idxs_per_partition<const BUILD_SKETCHES: bool>(
        &self,
        partitioner: &HashPartitioner,
        partition_idxs: &mut [Vec<IdxSize>],
        sketches: &mut [CardinalitySketch],
        partition_nulls: bool,
    ) {
        downcast_single_key_ca!(self.keys, |keys| {
            gen_idxs_per_partition::<_, BUILD_SKETCHES>(
                keys,
                &self.random_state,
                partitioner,
                partition_idxs,
                sketches,
                partition_nulls | self.null_is_valid,
            );
        });
    }

    pub fn sketch_cardinality(&self, sketch: &mut CardinalitySketch) {
        downcast_single_key_ca!(self.keys, |keys| {
            sketch_cardinality(keys, &self.random_state, sketch);
        });
    }
}

fn gen_partitions<T>(
    ca: &ChunkedArray<T>,
    random_state: &PlRandomState,
    partitioner: &HashPartitioner,
    partitions: &mut Vec<IdxSize>,
    partition_nulls: bool,
) where
    T: PolarsDataType,
    for<'a> <T as PolarsDataType>::Physical<'a>: TotalHash,
{
    partitions.reserve(ca.len());
    if ca.has_nulls() {
        // Arbitrarily put nulls in partition 0.
        let null_p = if partition_nulls { 0 } else { IdxSize::MAX };
        for arr in ca.downcast_iter() {
            partitions.extend(arr.iter().map(|opt_k| {
                if let Some(k) = opt_k {
                    let h = random_state.tot_hash_one(k);
                    partitioner.hash_to_partition(h) as IdxSize
                } else {
                    null_p
                }
            }))
        }
    } else {
        for arr in ca.downcast_iter() {
            partitions.extend(arr.values_iter().map(|k| {
                let h = random_state.tot_hash_one(k);
                partitioner.hash_to_partition(h) as IdxSize
            }));
        }
    }
}

fn gen_idxs_per_partition<T, const BUILD_SKETCHES: bool>(
    ca: &ChunkedArray<T>,
    random_state: &PlRandomState,
    partitioner: &HashPartitioner,
    partition_idxs: &mut [Vec<IdxSize>],
    sketches: &mut [CardinalitySketch],
    partition_nulls: bool,
) where
    T: PolarsDataType,
    for<'a> <T as PolarsDataType>::Physical<'a>: TotalHash,
{
    assert!(partition_idxs.len() == partitioner.num_partitions());
    assert!(!BUILD_SKETCHES || sketches.len() == partitioner.num_partitions());

    let mut idx = 0;
    if ca.has_nulls() {
        for arr in ca.downcast_iter() {
            for opt_k in arr.iter() {
                if let Some(k) = opt_k {
                    unsafe {
                        // SAFETY: we assured the number of partitions matches.
                        let h = random_state.tot_hash_one(k);
                        let p = partitioner.hash_to_partition(h);
                        partition_idxs.get_unchecked_mut(p).push(idx as IdxSize);
                        if BUILD_SKETCHES {
                            sketches.get_unchecked_mut(p).insert(h);
                        }
                    }
                } else if partition_nulls {
                    // Arbitrarily put nulls in partition 0.
                    unsafe {
                        partition_idxs.get_unchecked_mut(0).push(idx as IdxSize);
                    }
                }

                idx += 1;
            }
        }
    } else {
        for arr in ca.downcast_iter() {
            for k in arr.values_iter() {
                unsafe {
                    // SAFETY: we assured the number of partitions matches.
                    let h = random_state.tot_hash_one(k);
                    let p = partitioner.hash_to_partition(h);
                    partition_idxs.get_unchecked_mut(p).push(idx as IdxSize);
                    if BUILD_SKETCHES {
                        sketches.get_unchecked_mut(p).insert(h);
                    }
                }

                idx += 1;
            }
        }
    }
}

fn sketch_cardinality<T>(
    ca: &ChunkedArray<T>,
    random_state: &PlRandomState,
    sketch: &mut CardinalitySketch,
) where
    T: PolarsDataType,
    for<'a> <T as PolarsDataType>::Physical<'a>: TotalHash,
{
    if ca.has_nulls() {
        for arr in ca.downcast_iter() {
            for k in arr.iter().flatten() {
                sketch.insert(random_state.tot_hash_one(k));
            }
        }
    } else {
        for arr in ca.downcast_iter() {
            for k in arr.values_iter() {
                sketch.insert(random_state.tot_hash_one(k));
            }
        }
    }
}
