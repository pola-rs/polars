#[cfg(feature = "hash")]
use polars_core::export::ahash;
use polars_core::prelude::*;

use crate::series::ops::SeriesSealed;

pub trait SeriesMethods: SeriesSealed {
    /// Create a [`DataFrame`] with the unique `values` of this [`Series`] and a column `"counts"`
    /// with dtype [`IdxType`]
    fn value_counts(&self, multithreaded: bool, sorted: bool) -> PolarsResult<DataFrame> {
        let s = self.as_series();
        // we need to sort here as well in case of `maintain_order` because duplicates behavior is undefined
        let groups = s.group_tuples(multithreaded, sorted)?;
        let values = unsafe { s.agg_first(&groups) };
        let counts = groups.group_lengths("counts");
        let cols = vec![values.into_series(), counts.into_series()];
        let df = DataFrame::new_no_checks(cols);
        if sorted {
            df.sort(["counts"], true)
        } else {
            Ok(df)
        }
    }

    #[cfg(feature = "hash")]
    fn hash(&self, build_hasher: ahash::RandomState) -> UInt64Chunked {
        let s = self.as_series().to_physical_repr();
        match s.dtype() {
            DataType::List(_) => {
                let mut ca = s.list().unwrap().clone();
                crate::chunked_array::hash::hash(&mut ca, build_hasher)
            }
            _ => {
                let mut h = vec![];
                s.0.vec_hash(build_hasher, &mut h).unwrap();
                UInt64Chunked::from_vec(s.name(), h)
            }
        }
    }
}

impl SeriesMethods for Series {}
