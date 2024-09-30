use std::borrow::Cow;

use arrow::array::*;
use arrow::legacy::kernels::list::array_to_unit_list;
use arrow::offset::Offsets;
use polars_error::{polars_bail, polars_ensure, PolarsResult};
use polars_utils::format_tuple;

use crate::chunked_array::builder::get_list_builder;
use crate::datatypes::{DataType, ListChunked};
use crate::prelude::{IntoSeries, Series, *};

fn reshape_fast_path(name: PlSmallStr, s: &Series) -> Series {
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
                unsafe { Series::from_chunks_and_dtype_unchecked(s.name().clone(), chunks, dtype) }
                    .get_leaf_array()
            },
            DataType::List(dtype) => {
                let ca = s.list().unwrap();
                let chunks = ca
                    .downcast_iter()
                    .map(|arr| arr.values().clone())
                    .collect::<Vec<_>>();
                // Safety: guarded by the type system
                unsafe { Series::from_chunks_and_dtype_unchecked(s.name().clone(), chunks, dtype) }
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

        let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());

        // SAFETY: offsets are correct.
        let arr = unsafe {
            ListArray::new(
                dtype,
                Offsets::new_unchecked(offsets).into(),
                values.clone(),
                None,
            )
        };

        let mut ca = ListChunked::with_chunk(s.name().clone(), arr);
        unsafe { ca.to_logical(inner_type.clone()) };
        ca.set_fast_explode();
        Ok(ca)
    }

    #[cfg(feature = "dtype-array")]
    pub fn reshape_array(&self, dimensions: &[ReshapeDimension]) -> PolarsResult<Series> {
        polars_ensure!(
            !dimensions.is_empty(),
            InvalidOperation: "at least one dimension must be specified"
        );

        let leaf_array = self.get_leaf_array();
        let size = leaf_array.len();

        let mut total_dim_size = 1;
        let mut num_infers = 0;
        for &dim in dimensions {
            match dim {
                ReshapeDimension::Infer => num_infers += 1,
                ReshapeDimension::Specified(dim) => total_dim_size *= dim.get() as usize,
            }
        }

        polars_ensure!(num_infers <= 1, InvalidOperation: "can only specify one inferred dimension");

        if size == 0 {
            polars_ensure!(
                num_infers > 0 || total_dim_size == 0,
                InvalidOperation: "cannot reshape empty array into shape without zero dimension: {}",
                format_tuple!(dimensions),
            );

            let mut prev_arrow_dtype = leaf_array
                .dtype()
                .to_physical()
                .to_arrow(CompatLevel::newest());
            let mut prev_dtype = leaf_array.dtype().clone();
            let mut prev_array = leaf_array.chunks()[0].clone();

            // @NOTE: We need to collect the iterator here because it is lazily processed.
            let mut current_length = dimensions[0].get_or_infer(0);
            let len_iter = dimensions[1..]
                .iter()
                .map(|d| {
                    let length = current_length as usize;
                    current_length *= d.get_or_infer(0);
                    length
                })
                .collect::<Vec<_>>();

            // We pop the outer dimension as that is the height of the series.
            for (dim, length) in dimensions[1..].iter().zip(len_iter).rev() {
                // Infer dimension if needed
                let dim = dim.get_or_infer(0);
                prev_arrow_dtype = prev_arrow_dtype.to_fixed_size_list(dim as usize, true);
                prev_dtype = DataType::Array(Box::new(prev_dtype), dim as usize);

                prev_array =
                    FixedSizeListArray::new(prev_arrow_dtype.clone(), length, prev_array, None)
                        .boxed();
            }

            return Ok(unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    leaf_array.name().clone(),
                    vec![prev_array],
                    &prev_dtype,
                )
            });
        }

        polars_ensure!(
            total_dim_size > 0,
            InvalidOperation: "cannot reshape non-empty array into shape containing a zero dimension: {}",
            format_tuple!(dimensions)
        );

        polars_ensure!(
            size % total_dim_size == 0,
            InvalidOperation: "cannot reshape array of size {} into shape {}", size, format_tuple!(dimensions)
        );

        let leaf_array = leaf_array.rechunk();
        let mut prev_arrow_dtype = leaf_array
            .dtype()
            .to_physical()
            .to_arrow(CompatLevel::newest());
        let mut prev_dtype = leaf_array.dtype().clone();
        let mut prev_array = leaf_array.chunks()[0].clone();

        // We pop the outer dimension as that is the height of the series.
        for dim in dimensions[1..].iter().rev() {
            // Infer dimension if needed
            let dim = dim.get_or_infer((size / total_dim_size) as u64);
            prev_arrow_dtype = prev_arrow_dtype.to_fixed_size_list(dim as usize, true);
            prev_dtype = DataType::Array(Box::new(prev_dtype), dim as usize);

            prev_array = FixedSizeListArray::new(
                prev_arrow_dtype.clone(),
                prev_array.len() / dim as usize,
                prev_array,
                None,
            )
            .boxed();
        }
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                leaf_array.name().clone(),
                vec![prev_array],
                &prev_dtype,
            )
        })
    }

    pub fn reshape_list(&self, dimensions: &[ReshapeDimension]) -> PolarsResult<Series> {
        polars_ensure!(
            !dimensions.is_empty(),
            InvalidOperation: "at least one dimension must be specified"
        );

        let s = self;
        let s = if let DataType::List(_) = s.dtype() {
            Cow::Owned(s.explode()?)
        } else {
            Cow::Borrowed(s)
        };

        let s_ref = s.as_ref();

        // let dimensions = dimensions.to_vec();

        match dimensions.len() {
            1 => {
                polars_ensure!(
                    dimensions[0].get().map_or(true, |dim| dim as usize == s_ref.len()),
                    InvalidOperation: "cannot reshape len {} into shape {:?}", s_ref.len(), dimensions,
                );
                Ok(s_ref.clone())
            },
            2 => {
                let rows = dimensions[0];
                let cols = dimensions[1];

                if s_ref.len() == 0_usize {
                    if rows.get_or_infer(0) == 0 && cols.get_or_infer(0) <= 1 {
                        let s = reshape_fast_path(s.name().clone(), s_ref);
                        return Ok(s);
                    } else {
                        polars_bail!(InvalidOperation: "cannot reshape len 0 into shape {}", format_tuple!(dimensions))
                    }
                }

                use ReshapeDimension as RD;
                // Infer dimension.

                let (rows, cols) = match (rows, cols) {
                    (RD::Infer, RD::Specified(cols)) if cols.get() >= 1 => {
                        (s_ref.len() as u64 / cols.get(), cols.get())
                    },
                    (RD::Specified(rows), RD::Infer) if rows.get() >= 1 => {
                        (rows.get(), s_ref.len() as u64 / rows.get())
                    },
                    (RD::Infer, RD::Infer) => (s_ref.len() as u64, 1u64),
                    (RD::Specified(rows), RD::Specified(cols)) => (rows.get(), cols.get()),
                    _ => polars_bail!(InvalidOperation: "reshape of non-zero list into zero list"),
                };

                // Fast path, we can create a unit list so we only allocate offsets.
                if rows as usize == s_ref.len() && cols == 1 {
                    let s = reshape_fast_path(s.name().clone(), s_ref);
                    return Ok(s);
                }

                polars_ensure!(
                    (rows*cols) as usize == s_ref.len() && rows >= 1 && cols >= 1,
                    InvalidOperation: "cannot reshape len {} into shape {:?}", s_ref.len(), dimensions,
                );

                let mut builder =
                    get_list_builder(s_ref.dtype(), s_ref.len(), rows as usize, s.name().clone())?;

                let mut offset = 0u64;
                for _ in 0..rows {
                    let row = s_ref.slice(offset as i64, cols as usize);
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
        let s = Series::new("a".into(), &[1, 2, 3]);

        let mut builder = get_list_builder(s.dtype(), s.len(), 1, s.name().clone())?;
        builder.append_series(&s).unwrap();
        let expected = builder.finish();

        let out = s.implode()?;
        assert!(expected.into_series().equals(&out.into_series()));

        Ok(())
    }

    #[test]
    fn test_reshape() -> PolarsResult<()> {
        let s = Series::new("a".into(), &[1, 2, 3, 4]);

        for (dims, list_len) in [
            (&[-1, 1], 4),
            (&[4, 1], 4),
            (&[2, 2], 2),
            (&[-1, 2], 2),
            (&[2, -1], 2),
        ] {
            let dims = dims
                .iter()
                .map(|&v| ReshapeDimension::new(v))
                .collect::<Vec<_>>();
            let out = s.reshape_list(&dims)?;
            assert_eq!(out.len(), list_len);
            assert!(matches!(out.dtype(), DataType::List(_)));
            assert_eq!(out.explode()?.len(), 4);
        }

        Ok(())
    }
}
