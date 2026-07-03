use polars_core::runtime::RAYON;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use rayon::prelude::*;

use super::*;

pub type ArrToStructNameGenerator = Arc<dyn Fn(usize) -> PolarsResult<PlSmallStr> + Send + Sync>;

pub fn arr_default_struct_name_gen(idx: usize) -> PlSmallStr {
    format_pl_smallstr!("field_{idx}")
}

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
