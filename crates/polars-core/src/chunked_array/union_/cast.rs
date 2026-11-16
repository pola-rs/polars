//! Casts of unions: widening to another union, and formatting as `String`.
use polars_arrow::array::{ChildPositions, LargeUnionArray, new_empty_array};
use polars_buffer::Buffer;
use polars_compute::gather::gather_child;

use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

impl UnionChunked {
    /// Casts to `dtype`, which must be allowed by [`DataType::ensure_union_cast`].
    pub fn cast_with_options(
        &self,
        dtype: &DataType,
        options: CastOptions,
    ) -> PolarsResult<Series> {
        if dtype == self.dtype() {
            return Ok(self.clone().into_series());
        }
        self.dtype().ensure_union_cast(dtype)?;
        match dtype {
            DataType::Union(to_fields) => self.widen(dtype, to_fields),
            DataType::String => self.cast_to_string(options),
            _ => unreachable!("checked by `ensure_union_cast`"),
        }
    }

    /// Retags the slots for `to_fields` and retypes the nulls of variants whose dtype widens.
    fn widen(&self, dtype: &DataType, to_fields: &[Field]) -> PolarsResult<Series> {
        let from_fields = self.union_fields();
        // `ensure_union_cast` checked that every source variant is in the target.
        let target: Vec<usize> = from_fields
            .iter()
            .map(|f| to_fields.iter().position(|g| g.name() == f.name()).unwrap())
            .collect();
        let identity = target.iter().enumerate().all(|(s, &t)| s == t);
        let arrow_dtype = dtype.to_physical_arrow(CompatLevel::newest());

        let chunks = self
            .downcast_iter()
            .map(|arr| {
                let mut children: Vec<Option<ArrayRef>> = vec![None; to_fields.len()];
                for (s, &t) in target.iter().enumerate() {
                    let child = arr.fields()[s].clone();
                    let (from, to) = (from_fields[s].dtype(), to_fields[t].dtype());
                    children[t] = Some(if from == to {
                        child
                    } else {
                        // SAFETY: the child has the physical dtype of the variant.
                        let child = unsafe {
                            Series::from_chunks_and_dtype_unchecked(
                                PlSmallStr::EMPTY,
                                vec![child],
                                from,
                            )
                        };
                        // Widening only retypes nulls, so this cannot fail on the data.
                        let cast = child.cast_with_options(to, CastOptions::Strict)?.rechunk();
                        polars_ensure!(
                            cast.dtype() == to,
                            ComputeError: "widening `{from}` gave `{}` instead of `{to}`", cast.dtype()
                        );
                        cast.chunks()[0].clone()
                    });
                }
                let children = children
                    .into_iter()
                    .zip(to_fields)
                    .map(|(c, f)| {
                        c.unwrap_or_else(|| {
                            new_empty_array(f.dtype().to_physical_arrow(CompatLevel::newest()))
                        })
                    })
                    .collect();
                let types: Buffer<i8> = if identity {
                    arr.types().clone()
                } else {
                    arr.types().iter().map(|&t| target[t as usize] as i8).collect()
                };
                // SAFETY: the slots keep their offsets into children of unchanged lengths.
                Ok(unsafe {
                    LargeUnionArray::new_unchecked(
                        arrow_dtype.clone(),
                        types,
                        arr.offsets().clone(),
                        children,
                        arr.validity().cloned(),
                    )
                }
                .boxed())
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        // SAFETY: the chunks have the physical dtype of `dtype`.
        Ok(unsafe { Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, dtype) })
    }

    /// Formats every value as a string, without its variant.
    fn cast_to_string(&self, options: CastOptions) -> PolarsResult<Series> {
        let fields = self.union_fields();
        let mut out = StringChunked::full_null(self.name().clone(), 0);
        for arr in self.downcast_iter() {
            // Each variant's values in slot order, nulls for null slots, so a strict cast only
            // sees values the union holds.
            let mut positions: Vec<ChildPositions> = arr
                .fields()
                .iter()
                .map(|c| ChildPositions::new(c.len()))
                .collect();
            let mut counts = vec![0 as IdxSize; fields.len()];
            let mut ranks = Vec::with_capacity(arr.len());
            for (i, (&t, &o)) in arr.types().iter().zip(arr.offsets().iter()).enumerate() {
                let t = t as usize;
                positions[t].push(arr.is_valid(i).then_some(o as usize));
                ranks.push(counts[t]);
                counts[t] += 1;
            }

            let mut strings = StringChunked::full_null(PlSmallStr::EMPTY, 0);
            let mut bases = Vec::with_capacity(fields.len());
            for (s, positions) in positions.into_iter().enumerate() {
                bases.push(strings.len() as IdxSize);
                if counts[s] == 0 {
                    continue;
                }
                // SAFETY: the offsets of the union are in bounds.
                let values = unsafe { gather_child(arr.fields()[s].as_ref(), positions) };
                // SAFETY: the values have the physical dtype of the variant.
                let values = unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        PlSmallStr::EMPTY,
                        vec![values],
                        fields[s].dtype(),
                    )
                };
                strings.append_owned(
                    values
                        .cast_with_options(&DataType::String, options)?
                        .str()?
                        .clone(),
                )?;
            }
            let idx: IdxCa = arr
                .types()
                .iter()
                .zip(&ranks)
                .map(|(&t, &r)| bases[t as usize] + r)
                .collect_ca(PlSmallStr::EMPTY);
            // SAFETY: every index is below the number of slots, which is `strings.len()`.
            out.append_owned(unsafe { strings.take_unchecked(&idx) })?;
        }
        Ok(out.into_series())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn int_str() -> UnionChunked {
        let dtype = DataType::Union(
            UnionFields::try_new(vec![
                Field::new("a".into(), DataType::Int64),
                Field::new("b".into(), DataType::String),
            ])
            .unwrap(),
        );
        let arr = LargeUnionArray::new(
            dtype.to_physical_arrow(CompatLevel::newest()),
            vec![0i8, 1, 0].into(),
            vec![0i64, 0, 1].into(),
            vec![
                Int64Chunked::from_slice("".into(), &[1, 2]).chunks()[0].clone(),
                StringChunked::from_slice("".into(), &["x"]).chunks()[0].clone(),
            ],
            None,
        );
        unsafe {
            UnionChunked::from_chunks_and_dtype_unchecked("u".into(), vec![arr.boxed()], dtype)
        }
    }

    fn cast_offsets_shared(to: &DataType) -> bool {
        let ca = int_str();
        let out = ca.cast_with_options(to, CastOptions::Strict).unwrap();
        let out = out.union().unwrap().downcast_iter().next().unwrap();
        let input = ca.downcast_iter().next().unwrap();
        out.offsets().as_ptr() == input.offsets().as_ptr()
    }

    #[test]
    fn tag_only_cast_shares_offsets() {
        assert!(cast_offsets_shared(&DataType::Union(
            UnionFields::try_new(vec![
                Field::new("b".into(), DataType::String),
                Field::new("a".into(), DataType::Int64),
            ])
            .unwrap()
        )));
    }
}
