use polars_core::POOL;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use rayon::prelude::*;

use super::*;

pub type ArrToStructNameGenerator = Arc<dyn Fn(usize) -> PolarsResult<PlSmallStr> + Send + Sync>;

pub fn arr_default_struct_name_gen(idx: usize) -> PlSmallStr {
    format_pl_smallstr!("field_{idx}")
}

pub trait ToStruct: AsArray {
    fn to_struct(
        &self,
        name_generator: Option<ArrToStructNameGenerator>,
    ) -> PolarsResult<StructChunked> {
        let ca = self.as_array();
        let n_fields = ca.width();

        let name_generator = name_generator
            .as_deref()
            .unwrap_or(&|i| Ok(arr_default_struct_name_gen(i)));

        let fields = POOL.install(|| {
            (0..n_fields)
                .into_par_iter()
                .map(|i| {
                    ca.array_get(
                        &Int64Chunked::from_slice(PlSmallStr::EMPTY, &[i as i64]),
                        true,
                    )
                    .and_then(|mut s| {
                        s.rename(name_generator(i)?.clone());
                        Ok(s)
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())
    }
}

impl ToStruct for ArrayChunked {}
