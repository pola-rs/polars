use std::borrow::Cow;
#[cfg(feature = "dtype-array")]
use std::collections::VecDeque;

use arrow::array::*;
use arrow::legacy::kernels::list::array_to_unit_list;
use arrow::offset::Offsets;
use polars_error::{polars_bail, polars_ensure, PolarsResult};
#[cfg(feature = "dtype-array")]
use polars_utils::format_tuple;

use crate::chunked_array::builder::get_list_builder;
use crate::datatypes::{DataType, ListChunked};
use crate::prelude::{IntoSeries, Series, *};

fn reshape_fast_path(name: &str, s: &Series) -> Series {
    let mut ca = match s.dtype() {
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => {
            ListChunked::with_chunk(name, array_to_unit_list(s.array_ref(0).clone()))
        },
        _ => ListChunked::from_chunk_iter(
            name,
            s.chunks().iter().map(|arr| array_to_unit_list(arr.clone())),
        ),
    };
    ca.set_inner_dtype(s.dtype().clone());
    ca.set_fast_explode();
    ca.into_series()
}

impl Series {
    /// Recurse nested types until we are at the leaf array.
    pub fn get_leaf_array(&self) -> Series {
        let s = self;
        match s.dtype() {
            #[cfg(feature = "dtype-array")]
            DataType::Array(dtype, _) => {
                let ca = s.array().unwrap();
                let chunks = ca
                    .downcast_iter()
                    .map(|arr| arr.values().clone())
                    .collect::<Vec<_>>();
                // Safety: guarded by the type system
                unsafe { Series::from_chunks_and_dtype_unchecked(s.name(), chunks, dtype) }
                    .get_leaf_array()
            },
            DataType::List(dtype) => {
                let ca = s.list().unwrap();
                let chunks = ca
                    .downcast_iter()
                    .map(|arr| arr.values().clone())
                    .collect::<Vec<_>>();
                // Safety: guarded by the type system
                unsafe { Series::from_chunks_and_dtype_unchecked(s.name(), chunks, dtype) }
                    .get_leaf_array()
            },
            _ => s.clone(),
        }
    }

    /// Convert the values of this Series to a ListChunked with a length of 1,
    /// so a Series of `[1, 2, 3]` becomes `[[1, 2, 3]]`.
    pub fn implode(&self) -> PolarsResult<ListChunked> {
        let s = self;
        let s = s.rechunk();
        let values = s.array_ref(0);

        let offsets = vec![0i64, values.len() as i64];
        let inner_type = s.dtype();

        let data_type = ListArray::<i64>::default_datatype(values.data_type().clone());

        // SAFETY: offsets are correct.
        let arr = unsafe {
            ListArray::new(
                data_type,
                Offsets::new_unchecked(offsets).into(),
                values.clone(),
                None,
            )
        };

        let mut ca = ListChunked::with_chunk(s.name(), arr);
        unsafe { ca.to_logical(inner_type.clone()) };
        ca.set_fast_explode();
        Ok(ca)
    }

    #[cfg(feature = "dtype-array")]
    pub fn reshape_array(&self, dimensions: &[i64]) -> PolarsResult<Series> {
        let mut dims = dimensions.iter().copied().collect::<VecDeque<_>>();

        let leaf_array = self.get_leaf_array();
        let size = leaf_array.len() as i64;

        // Infer dimension
        if dims.contains(&-1) {
            let infer_dims = dims.iter().filter(|d| **d == -1).count();
            polars_ensure!(infer_dims == 1, InvalidOperation: "can only infer single dimension, found {}", infer_dims);

            let mut prod = 1;
            for &dim in &dims {
                if dim != -1 {
                    prod *= dim;
                }
            }
            polars_ensure!(size % prod == 0, InvalidOperation: "cannot reshape array of size {} into shape: {}", size, format_tuple!(dims));
            let inferred_value = size / prod;
            for dim in &mut dims {
                if *dim == -1 {
                    *dim = inferred_value;
                    break;
                }
            }
        }
        let leaf_array = leaf_array.rechunk();
        let mut prev_dtype = leaf_array.dtype().clone();
        let mut prev_array = leaf_array.chunks()[0].clone();

        // We pop the outer dimension as that is the height of the series.
        let _ = dims.pop_front();
        while let Some(dim) = dims.pop_back() {
            prev_dtype = DataType::Array(Box::new(prev_dtype), dim as usize);

            prev_array =
                FixedSizeListArray::new(prev_dtype.to_arrow(true), prev_array, None).boxed();
        }
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                leaf_array.name(),
                vec![prev_array],
                &prev_dtype,
            )
        })
    }

    pub fn reshape_list(&self, dimensions: &[i64]) -> PolarsResult<Series> {
        let s = self;

        if dimensions.is_empty() {
            polars_bail!(ComputeError: "reshape `dimensions` cannot be empty")
        }
        let s = if let DataType::List(_) = s.dtype() {
            Cow::Owned(s.explode()?)
        } else {
            Cow::Borrowed(s)
        };

        let s_ref = s.as_ref();

        let dimensions = dimensions.to_vec();

        match dimensions.len() {
            1 => {
                polars_ensure!(
                    dimensions[0] as usize == s_ref.len() || dimensions[0] == -1_i64,
                    ComputeError: "cannot reshape len {} into shape {:?}", s_ref.len(), dimensions,
                );
                Ok(s_ref.clone())
            },
            2 => {
                let mut rows = dimensions[0];
                let mut cols = dimensions[1];

                if s_ref.len() == 0_usize {
                    if (rows == -1 || rows == 0) && (cols == -1 || cols == 0) {
                        let s = reshape_fast_path(s.name(), s_ref);
                        return Ok(s);
                    } else {
                        polars_bail!(ComputeError: "cannot reshape len 0 into shape {:?}", dimensions,)
                    }
                }

                // Infer dimension.
                if rows == -1 && cols >= 1 {
                    rows = s_ref.len() as i64 / cols
                } else if cols == -1 && rows >= 1 {
                    cols = s_ref.len() as i64 / rows
                } else if rows == -1 && cols == -1 {
                    rows = s_ref.len() as i64;
                    cols = 1_i64;
                }

                // Fast path, we can create a unit list so we only allocate offsets.
                if rows as usize == s_ref.len() && cols == 1 {
                    let s = reshape_fast_path(s.name(), s_ref);
                    return Ok(s);
                }

                polars_ensure!(
                    (rows*cols) as usize == s_ref.len() && rows >= 1 && cols >= 1,
                    ComputeError: "cannot reshape len {} into shape {:?}", s_ref.len(), dimensions,
                );

                let mut builder =
                    get_list_builder(s_ref.dtype(), s_ref.len(), rows as usize, s.name())?;

                let mut offset = 0i64;
                for _ in 0..rows {
                    let row = s_ref.slice(offset, cols as usize);
                    builder.append_series(&row).unwrap();
                    offset += cols;
                }
                Ok(builder.finish().into_series())
            },
            _ => {
                polars_bail!(InvalidOperation: "more than two dimensions not supported in reshaping to List.\n\nConsider reshaping to Array type.");
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_to_list() -> PolarsResult<()> {
        let s = Series::new("a", &[1, 2, 3]);

        let mut builder = get_list_builder(s.dtype(), s.len(), 1, s.name())?;
        builder.append_series(&s).unwrap();
        let expected = builder.finish();

        let out = s.implode()?;
        assert!(expected.into_series().equals(&out.into_series()));

        Ok(())
    }

    #[test]
    fn test_reshape() -> PolarsResult<()> {
        let s = Series::new("a", &[1, 2, 3, 4]);

        for (dims, list_len) in [
            (&[-1, 1], 4),
            (&[4, 1], 4),
            (&[2, 2], 2),
            (&[-1, 2], 2),
            (&[2, -1], 2),
        ] {
            let out = s.reshape_list(dims)?;
            assert_eq!(out.len(), list_len);
            assert!(matches!(out.dtype(), DataType::List(_)));
            assert_eq!(out.explode()?.len(), 4);
        }

        Ok(())
    }
}
