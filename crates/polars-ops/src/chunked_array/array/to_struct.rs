use std::sync::Arc;

use arrow::array::{Array, StructArray};
use arrow::datatypes::ArrowDataType;
use polars_core::chunked_array::StructChunked;
use polars_core::chunked_array::builder::NewChunkedArray as _;
use polars_core::datatypes::{ArrayChunked, CompatLevel, DataType, Field, Int64Chunked};
use polars_core::runtime::RAYON;
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};

use crate::chunked_array::array::{ArrayNameSpace as _, AsArray};

pub trait ToStruct: AsArray {
    fn to_struct(&self, fields: &[PlSmallStr]) -> PolarsResult<StructChunked> {
        let ca = self.as_array();

        let field_arrays: Vec<Box<dyn Array>> = RAYON.install(|| {
            (0..fields.len())
                .into_par_iter()
                .map(|i| {
                    ca.array_get(
                        &Int64Chunked::from_slice(PlSmallStr::EMPTY, &[i as i64]),
                        true,
                    )
                    .map(|s| s.rechunk().chunks()[0].clone())
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        let field_dtype = ca.inner_dtype();
        let field_phys_dtype = field_dtype.to_physical();
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
                vec![Box::new(StructArray::new(
                    ArrowDataType::Struct(
                        fields
                            .iter()
                            .map(|name| {
                                field_phys_dtype.to_arrow_field(name.clone(), CompatLevel::newest())
                            })
                            .collect(),
                    ),
                    ca.len(),
                    field_arrays,
                    ca.rechunk_validity(),
                ))],
                ca.len(),
                outer_validity.map_or(0, |x| x.unset_bits()),
            )
        })
    }
}

impl ToStruct for ArrayChunked {}
