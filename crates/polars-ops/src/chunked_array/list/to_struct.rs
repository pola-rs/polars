use polars_core::export::rayon::prelude::*;
use polars_core::POOL;
use polars_utils::format_smartstring;
use smartstring::alias::String as SmartString;

use super::*;

#[derive(Copy, Clone, Debug)]
pub enum ListToStructWidthStrategy {
    FirstNonNull,
    MaxWidth,
}

fn det_n_fields(ca: &ListChunked, n_fields: ListToStructWidthStrategy) -> usize {
    match n_fields {
        ListToStructWidthStrategy::MaxWidth => {
            let mut max = 0;

            ca.downcast_iter().for_each(|arr| {
                let offsets = arr.offsets().as_slice();
                let mut last = offsets[0];
                for o in &offsets[1..] {
                    let len = (*o - last) as usize;
                    max = std::cmp::max(max, len);
                    last = *o;
                }
            });
            max
        },
        ListToStructWidthStrategy::FirstNonNull => {
            let mut len = 0;
            for arr in ca.downcast_iter() {
                let offsets = arr.offsets().as_slice();
                let mut last = offsets[0];
                for o in &offsets[1..] {
                    len = (*o - last) as usize;
                    if len > 0 {
                        break;
                    }
                    last = *o;
                }
                if len > 0 {
                    break;
                }
            }
            len
        },
    }
}

pub type NameGenerator = Arc<dyn Fn(usize) -> SmartString + Send + Sync>;

pub fn _default_struct_name_gen(idx: usize) -> SmartString {
    format_smartstring!("field_{idx}")
}

pub trait ToStruct: AsList {
    fn to_struct(
        &self,
        n_fields: ListToStructWidthStrategy,
        name_generator: Option<NameGenerator>,
    ) -> PolarsResult<StructChunked> {
        let ca = self.as_list();
        let n_fields = det_n_fields(ca, n_fields);

        let name_generator = name_generator
            .as_deref()
            .unwrap_or(&_default_struct_name_gen);

        polars_ensure!(n_fields != 0, ComputeError: "cannot create a struct with 0 fields");
        let fields = POOL.install(|| {
            (0..n_fields)
                .into_par_iter()
                .map(|i| {
                    ca.lst_get(i as i64, true).map(|mut s| {
                        s.rename(&name_generator(i));
                        s
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        StructChunked::from_series(ca.name(), &fields)
    }
}

impl ToStruct for ListChunked {}
