use polars_core::export::rayon::prelude::*;
use polars_core::POOL;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::*;

pub type ArrToStructNameGenerator = Arc<dyn Fn(usize) -> PlSmallStr + Send + Sync>;

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
            .unwrap_or(&arr_default_struct_name_gen);

        polars_ensure!(n_fields != 0, ComputeError: "cannot create a struct with 0 fields");
        let fields = POOL.install(|| {
            (0..n_fields)
                .into_par_iter()
                .map(|i| {
                    ca.array_get(
                        &Int64Chunked::from_slice(PlSmallStr::const_default(), &[i as i64]),
                        true,
                    )
                    .map(|mut s| {
                        s.rename(name_generator(i).clone());
                        s
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        StructChunked::from_series(ca.name().clone(), &fields)
    }
}

impl ToStruct for ArrayChunked {}
