use super::*;
use polars_core::export::rayon::prelude::*;

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
        }
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
        }
    }
}

pub type NameGenerator = Arc<dyn Fn(usize) -> String + Send + Sync>;

pub trait ToStruct: AsList {
    fn to_struct(
        &self,
        n_fields: ListToStructWidthStrategy,
        name_generator: Option<NameGenerator>,
    ) -> Result<StructChunked> {
        let ca = self.as_list();
        let n_fields = det_n_fields(ca, n_fields);

        let default_name_gen = |idx| format!("field_{idx}");

        let name_generator = name_generator.as_deref().unwrap_or(&default_name_gen);

        if n_fields == 0 {
            Err(PolarsError::ComputeError(
                "cannot create a struct with 0 fields".into(),
            ))
        } else {
            let fields = (0..n_fields)
                .into_par_iter()
                .map(|i| {
                    ca.lst_get(i as i64).map(|mut s| {
                        s.rename(&name_generator(i));
                        s
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            StructChunked::new(ca.name(), &fields)
        }
    }
}

impl ToStruct for ListChunked {}
