use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use polars_arrow::kernels::list::array_to_unit_list;
use std::borrow::Cow;

impl Series {
    /// Convert the values of this Series to a ListChunked with a length of 1,
    /// So a Series of:
    /// `[1, 2, 3]` becomes `[[1, 2, 3]]`
    pub fn to_list(&self) -> Result<ListChunked> {
        let s = self.rechunk();
        let values = &s.chunks()[0];

        let mut offsets = AlignedVec::with_capacity(2);
        offsets.push(0i64);
        offsets.push(values.len() as i64);

        let data_type = ListArray::<i64>::default_datatype(self.dtype().to_arrow());

        let arr = ListArray::from_data(data_type, offsets.into(), values.clone(), None);

        Ok(ListChunked::new_from_chunks(
            self.name(),
            vec![Arc::new(arr)],
        ))
    }

    pub fn reshape(&self, dims: &[i64]) -> Result<Series> {
        let s = if let DataType::List(_) = self.dtype() {
            Cow::Owned(self.explode()?)
        } else {
            Cow::Borrowed(self)
        };
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
        if prod != s_ref.len() {
            return Err(PolarsError::ValueError(
                format!("cannot reshape len {} into shape {:?}", s_ref.len(), dims).into(),
            ));
        }

        match dims.len() {
            0 => {
                panic!("dimensions cannot be empty")
            }
            1 => Ok(s_ref.slice(0, dims[0] as usize)),
            2 => {
                let mut rows = dims[0];
                let mut cols = dims[1];

                // infer dimension
                if rows == -1 {
                    rows = cols / s_ref.len() as i64
                }
                if cols == -1 {
                    cols = rows / s_ref.len() as i64
                }

                // fast path, we can create a unit list so we only allocate offsets
                if rows as usize == s_ref.len() && cols == 1 {
                    let chunks = s_ref
                        .chunks()
                        .iter()
                        .map(|arr| Arc::new(array_to_unit_list(arr.clone())) as ArrayRef)
                        .collect::<Vec<_>>();

                    let mut ca = ListChunked::new_from_chunks(self.name(), chunks);
                    ca.set_fast_explode();
                    return Ok(ca.into_series());
                }

                let mut builder =
                    get_list_builder(s_ref.dtype(), s_ref.len(), rows as usize, self.name());

                let mut offset = 0i64;
                for _ in 0..rows {
                    let row = s_ref.slice(offset, cols as usize);
                    builder.append_series(&row);
                    offset += cols;
                }
                Ok(builder.finish().into_series())
            }
            _ => {
                panic!("more than two dimensions not yet supported");
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_to_list() -> Result<()> {
        let s = Series::new("a", &[1, 2, 3]);

        let mut builder = get_list_builder(s.dtype(), s.len(), 1, s.name());
        builder.append_series(&s);
        let expected = builder.finish();

        let out = s.to_list()?;
        assert!(expected.into_series().series_equal(&out.into_series()));

        Ok(())
    }

    #[test]
    fn test_reshape() -> Result<()> {
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
