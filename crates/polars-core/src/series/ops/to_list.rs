use std::borrow::Cow;

use arrow::legacy::kernels::list::array_to_unit_list;
use arrow::offset::Offsets;

use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;

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
    /// Convert the values of this Series to a ListChunked with a length of 1,
    /// so a Series of `[1, 2, 3]` becomes `[[1, 2, 3]]`.
    pub fn implode(&self) -> PolarsResult<ListChunked> {
        let s = self.rechunk();
        let values = s.array_ref(0);

        let offsets = vec![0i64, values.len() as i64];
        let inner_type = self.dtype();

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

        let mut ca = ListChunked::with_chunk(self.name(), arr);
        ca.to_logical(inner_type.clone());
        ca.set_fast_explode();
        Ok(ca)
    }

    pub fn reshape(&self, dims: &[i64]) -> PolarsResult<Series> {
        if dims.is_empty() {
            polars_bail!(ComputeError: "reshape `dimensions` cannot be empty")
        }
        let s = if let DataType::List(_) = self.dtype() {
            Cow::Owned(self.explode()?)
        } else {
            Cow::Borrowed(self)
        };

        // No rows.
        if dims[0] == 0 {
            let s = reshape_fast_path(self.name(), &s);
            return Ok(s);
        }

        let s_ref = s.as_ref();

        let mut dims = dims.to_vec();
        if let Some(idx) = dims.iter().position(|i| *i == -1) {
            let mut product = 1;

            for (cnt, dim) in dims.iter().enumerate() {
                if cnt != idx {
                    product *= *dim
                }
            }
            dims[idx] = s_ref.len() as i64 / product;
        }

        let prod = dims.iter().product::<i64>() as usize;
        polars_ensure!(
            prod == s_ref.len(),
            ComputeError: "cannot reshape len {} into shape {:?}", s_ref.len(), dims,
        );
        match dims.len() {
            1 => Ok(s_ref.slice(0, dims[0] as usize)),
            2 => {
                let mut rows = dims[0];
                let mut cols = dims[1];

                // Infer dimension.
                if rows == -1 {
                    rows = cols / s_ref.len() as i64
                }
                if cols == -1 {
                    cols = rows / s_ref.len() as i64
                }

                // Fast path, we can create a unit list so we only allocate offsets.
                if rows as usize == s_ref.len() && cols == 1 {
                    let s = reshape_fast_path(self.name(), s_ref);
                    return Ok(s);
                }

                let mut builder =
                    get_list_builder(s_ref.dtype(), s_ref.len(), rows as usize, self.name())?;

                let mut offset = 0i64;
                for _ in 0..rows {
                    let row = s_ref.slice(offset, cols as usize);
                    builder.append_series(&row).unwrap();
                    offset += cols;
                }
                Ok(builder.finish().into_series())
            },
            _ => {
                panic!("more than two dimensions not yet supported");
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_to_list() -> PolarsResult<()> {
        let s = Series::new("a", &[1, 2, 3]);

        let mut builder = get_list_builder(s.dtype(), s.len(), 1, s.name())?;
        builder.append_series(&s).unwrap();
        let expected = builder.finish();

        let out = s.implode()?;
        assert!(expected.into_series().series_equal(&out.into_series()));

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
            let out = s.reshape(dims)?;
            assert_eq!(out.len(), list_len);
            assert!(matches!(out.dtype(), DataType::List(_)));
            assert_eq!(out.explode()?.len(), 4);
        }

        Ok(())
    }
}
