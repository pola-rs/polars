//! Casts between unions, matching variants by name.
use polars_arrow::array::{Array, ChildPositions, LargeUnionArray, new_empty_array};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_compute::gather::{gather_child, gather_union_slots};

use polars_buffer::Buffer;

use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

impl UnionChunked {
    /// Casts to `dtype`. Only casts to the same dtype or to another `Union` are supported.
    ///
    /// Variants are matched by name. Values of a variant missing from `dtype` become nulls,
    /// or raise if the cast is strict; variants only `dtype` has stay empty.
    pub fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        if dtype == self.dtype() {
            return Ok(self.clone().into_series());
        }
        let DataType::Union(to_fields) = dtype else {
            polars_bail!(
                InvalidOperation: "casting from {} to {} is not supported", self.dtype(), dtype
            );
        };
        dtype.ensure_valid_union_dtype()?;
        let from_fields = self.union_fields();
        let target: Vec<Option<usize>> = from_fields
            .iter()
            .map(|f| to_fields.iter().position(|g| g.name() == f.name()))
            .collect();
        let arrow_dtype = dtype.to_physical_arrow(CompatLevel::newest());

        let chunks = self
            .downcast_iter()
            .map(|arr| {
                cast_chunk(arr, from_fields, to_fields, &target, &arrow_dtype, options)
                    .map(|a| a.boxed())
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        // SAFETY: the chunks have the physical dtype of `dtype`.
        Ok(unsafe { Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, dtype) })
    }
}

fn cast_chunk(
    arr: &LargeUnionArray,
    from_fields: &[Field],
    to_fields: &[Field],
    target: &[Option<usize>],
    arrow_dtype: &ArrowDataType,
    options: CastOptions,
) -> PolarsResult<LargeUnionArray> {
    let types = arr.types();
    let offsets = arr.offsets();
    let is_valid = |i: usize| arr.validity().is_none_or(|v| v.get_bit(i));

    // Slots of variants missing from the target, valid or not, need a rebuild.
    let mut has_dropped_slots = false;
    for (i, &t) in types.iter().enumerate() {
        if target[t as usize].is_none() {
            has_dropped_slots = true;
            polars_ensure!(
                !options.is_strict() || !is_valid(i),
                InvalidOperation: "cannot cast {} to {}: variant `{}` is not in the target",
                DataType::Union(from_fields.to_vec()), DataType::Union(to_fields.to_vec()),
                from_fields[t as usize].name()
            );
        }
    }

    let mut children: Vec<Option<ArrayRef>> = vec![None; to_fields.len()];
    let mut new_offsets: Option<Vec<i64>> = None;
    let mut validity: Option<MutableBitmap> = None;
    for (s, t) in target.iter().enumerate() {
        let Some(t) = *t else { continue };
        let child = &arr.fields()[s];
        if from_fields[s].dtype() == to_fields[t].dtype() {
            children[t] = Some(child.clone());
            continue;
        }

        // Cast only the values the variant's slots reference, in slot order, with nulls for
        // null slots, so a strict cast ignores values the union does not hold.
        let mut positions = ChildPositions::new(child.len());
        for (i, (&ty, &o)) in types.iter().zip(offsets.iter()).enumerate() {
            if ty as usize == s {
                positions.push(is_valid(i).then_some(o as usize));
            }
        }
        // SAFETY: the offsets of the union are in bounds.
        let values = unsafe { gather_child(child.as_ref(), positions) };
        let values = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![values],
                from_fields[s].dtype(),
            )
        };
        let cast = values
            .cast_with_options(to_fields[t].dtype(), options)?
            .rechunk();
        let cast = cast.chunks()[0].clone();

        // A value a non-strict cast turned into a null nulls its slot too.
        let out_offsets = new_offsets.get_or_insert_with(|| offsets.as_slice().to_vec());
        let mut k = 0;
        for (i, &ty) in types.iter().enumerate() {
            if ty as usize == s {
                out_offsets[i] = k as i64;
                if !cast.is_valid(k) && is_valid(i) {
                    validity
                        .get_or_insert_with(|| match arr.validity() {
                            Some(v) => v.clone().make_mut(),
                            None => MutableBitmap::from_len_set(arr.len()),
                        })
                        .set(i, false);
                }
                k += 1;
            }
        }
        children[t] = Some(cast);
    }

    let children: Vec<ArrayRef> = children
        .into_iter()
        .zip(to_fields)
        .map(|(c, f)| {
            c.unwrap_or_else(|| new_empty_array(f.dtype().to_physical_arrow(CompatLevel::newest())))
        })
        .collect();
    let identity = target.iter().enumerate().all(|(s, t)| *t == Some(s));
    let new_types: Buffer<i8> = if identity {
        types.clone()
    } else {
        types
            .iter()
            .map(|&t| target[t as usize].unwrap_or(0) as i8)
            .collect()
    };
    let validity: Option<Bitmap> = match validity {
        Some(v) => Some(v.freeze()),
        None => arr.validity().cloned(),
    };

    if has_dropped_slots {
        let mut v = validity.map_or_else(|| MutableBitmap::from_len_set(arr.len()), |v| v.make_mut());
        for (i, &t) in types.iter().enumerate() {
            if target[t as usize].is_none() {
                v.set(i, false);
            }
        }
        let offsets = new_offsets.as_deref().unwrap_or(offsets.as_slice());
        // SAFETY: every valid slot has a variant of the target and an in-bounds offset.
        return Ok(unsafe {
            gather_union_slots(
                arrow_dtype,
                &new_types,
                offsets,
                &children,
                Some(v.freeze()),
                0..arr.len(),
            )
        });
    }

    let offsets = match new_offsets {
        Some(o) => o.into(),
        None => offsets.clone(),
    };
    // SAFETY: re-cast variants have offsets `0..k` over a child of length `k`, the others keep
    // theirs, and only slots whose value became null were made null.
    Ok(unsafe {
        LargeUnionArray::new_unchecked(arrow_dtype.clone(), new_types, offsets, children, validity)
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tag_only_cast_shares_offsets() {
        let dtype = DataType::Union(vec![
            Field::new("a".into(), DataType::Int64),
            Field::new("b".into(), DataType::String),
        ]);
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
        let ca = unsafe {
            UnionChunked::from_chunks_and_dtype_unchecked("u".into(), vec![arr.boxed()], dtype)
        };
        let to = DataType::Union(vec![
            Field::new("b".into(), DataType::String),
            Field::new("a".into(), DataType::Int64),
        ]);
        let out = ca.cast_with_options(&to, CastOptions::Strict).unwrap();
        let out = out.union().unwrap().downcast_iter().next().unwrap();
        let input = ca.downcast_iter().next().unwrap();
        assert_eq!(out.types().as_slice(), &[1, 0, 1]);
        assert_eq!(out.offsets().as_ptr(), input.offsets().as_ptr());
    }
}
