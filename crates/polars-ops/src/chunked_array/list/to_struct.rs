use polars_core::chunked_array::StructChunked;
use polars_core::datatypes::ListChunked;
use polars_core::runtime::RAYON;
use polars_core::series::Series;
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IndexedParallelIterator as _, IntoParallelIterator as _, ParallelIterator as _};

use crate::chunked_array::{AsList, ListNameSpaceImpl as _};

pub trait ToStruct: AsList {
    fn to_struct(&self, fields: &[PlSmallStr]) -> PolarsResult<StructChunked> {
        let ca = self.as_list();

        let fields: Vec<Series> = RAYON.install(|| {
            fields
                .into_par_iter()
                .enumerate()
                .map(|(i, name)| {
                    ca.lst_get(i as i64, true)
                        .map(|s| s.with_name(name.clone()))
                })
                .collect::<PolarsResult<_>>()
        })?;

        StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())
    }
}

impl ToStruct for ListChunked {}
