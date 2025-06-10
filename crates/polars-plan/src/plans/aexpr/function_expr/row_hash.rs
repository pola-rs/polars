use std::hash::BuildHasher;

use polars_utils::aliases::{
    PlFixedStateQuality, PlSeedableRandomStateQuality, SeedableFromU64SeedExt,
};

use super::*;

pub(super) fn row_hash(c: &Column, k0: u64, k1: u64, k2: u64, k3: u64) -> PolarsResult<Column> {
    // TODO: don't expose all these seeds.
    let seed = PlFixedStateQuality::default().hash_one((k0, k1, k2, k3));

    // @scalar-opt
    Ok(c.as_materialized_series()
        .hash(PlSeedableRandomStateQuality::seed_from_u64(seed))
        .into_column())
}
