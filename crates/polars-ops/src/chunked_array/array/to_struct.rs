use polars_core::export::rayon::prelude::*;
use polars_core::POOL;
use polars_utils::format_smartstring;
use smartstring::alias::String as SmartString;

use super::*;

pub type ArrToStructNameGenerator = Arc<dyn Fn(usize) -> SmartString + Send + Sync>;

pub fn arr_default_struct_name_gen(idx: usize) -> SmartString {
    format_smartstring!("field_{idx}")
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
                    ca.array_get(&Int64Chunked::from_slice("", &[i as i64]), true)
                        .map(|mut s| {
                            s.rename(&name_generator(i));
                            s
                        })
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        StructChunked::from_series(ca.name(), &fields)
    }
}

impl ToStruct for ArrayChunked {}
