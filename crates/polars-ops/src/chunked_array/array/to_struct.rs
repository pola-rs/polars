use polars_core::chunked_array::StructChunked;
use polars_core::chunked_array::builder::NewChunkedArray as _;
use polars_core::datatypes::{ArrayChunked, Int64Chunked};
use polars_core::runtime::RAYON;
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IndexedParallelIterator as _, IntoParallelIterator as _, ParallelIterator as _};

use crate::chunked_array::array::{ArrayNameSpace as _, AsArray};

pub trait ToStruct: AsArray {
    fn to_struct(&self, fields: &[PlSmallStr]) -> PolarsResult<StructChunked> {
        let ca = self.as_array();

        let fields = RAYON.install(|| {
            fields
                .into_par_iter()
                .enumerate()
                .map(|(i, name)| {
                    ca.array_get(
                        &Int64Chunked::from_slice(PlSmallStr::EMPTY, &[i as i64]),
                        true,
                    )
                    .map(|s| s.with_name(name.clone()))
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())
    }
}

impl ToStruct for ArrayChunked {}
