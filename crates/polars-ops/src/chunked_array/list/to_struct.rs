use std::sync::Arc;

use polars_array::{PlArray, PlStructArray};
use polars_core::chunked_array::StructChunked;
use polars_core::datatypes::{DataType, Field, ListChunked};
use polars_core::runtime::RAYON;
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};

use crate::chunked_array::{AsList, ListNameSpaceImpl as _};

pub trait ToStruct: AsList {
    fn to_struct(&self, fields: &[PlSmallStr]) -> PolarsResult<StructChunked> {
        let ca = self.as_list();

        let field_arrays: Vec<Box<dyn PlArray>> = RAYON.install(|| {
            (0..fields.len())
                .into_par_iter()
                .map(|i| {
                    ca.lst_get(i as i64, true)
                        .map(|s| s.rechunk().chunks()[0].clone())
                })
                .collect::<PolarsResult<_>>()
        })?;

        let field_dtype = ca.inner_dtype();
        let outer_validity = ca.rechunk_validity();

        Ok(unsafe {
            StructChunked::new_with_dims(
                Arc::new(Field::new(
                    ca.name().clone(),
                    DataType::Struct(
                        fields
                            .iter()
                            .map(|name| Field::new(name.clone(), field_dtype.clone()))
                            .collect(),
                    ),
                )),
                // `rechunk_validity` hands back a flat mask, one bit per element, which is what
                // a struct array takes; its field names live in the `DataType` above.
                vec![Box::new(PlStructArray::new(
                    field_arrays,
                    ca.len(),
                    ca.rechunk_validity(),
                ))],
                ca.len(),
                outer_validity.map_or(0, |x| x.unset_bits()),
            )
        })
    }
}

impl ToStruct for ListChunked {}
