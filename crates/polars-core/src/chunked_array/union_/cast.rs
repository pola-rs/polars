//! Casts between unions, matching variants by name.
use polars_arrow::array::{ChildPositions, LargeUnionArray, new_empty_array};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_buffer::Buffer;
use polars_compute::gather::{gather_child, gather_union_slots};

use crate::chunked_array::align_inner_chunks;
use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

impl UnionChunked {
    /// Casts to `dtype`. Only casts to the same dtype or to another `Union` are supported.
    ///
    /// Variants are matched by name. Values of a variant missing from `dtype` become nulls,
    /// or raise if the cast is strict; variants only `dtype` has stay empty.
    pub fn cast_with_options(
        &self,
        dtype: &DataType,
        options: CastOptions,
    ) -> PolarsResult<Series> {
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
        let arrs: Vec<&LargeUnionArray> = self.downcast_iter().collect();
        for arr in &arrs {
            check_dropped_variants(arr, from_fields, to_fields, &target, options)?;
        }

        // The dtype a cast produces can depend on the data (e.g. the inner dtype of a bare
        // `List` is only resolved for values that are not all null), so each variant is cast
        // once over all chunks to get a single dtype.
        let mut out_fields = to_fields.clone();
        let mut cast_children: Vec<Option<Vec<CastChild>>> = vec![None; from_fields.len()];
        for (s, t) in target.iter().enumerate() {
            let Some(t) = *t else { continue };
            if from_fields[s].dtype() == to_fields[t].dtype() {
                continue;
            }
            let (whole, values): (Vec<bool>, Vec<ArrayRef>) =
                arrs.iter().map(|arr| variant_values(arr, s)).unzip();
            let lens: Vec<usize> = values.iter().map(|v| v.len()).collect();
            // SAFETY: the values have the physical dtype of the variant.
            let values = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    PlSmallStr::EMPTY,
                    values,
                    from_fields[s].dtype(),
                )
            };
            let cast = values.cast_with_options(to_fields[t].dtype(), options)?;
            out_fields[t].dtype = cast.dtype().clone();
            let cast = align_inner_chunks(lens.into_iter(), &cast);
            cast_children[s] = Some(
                whole
                    .into_iter()
                    .zip(cast.chunks().iter().cloned())
                    .map(|(whole_child, values)| CastChild {
                        values,
                        whole_child,
                    })
                    .collect(),
            );
        }

        let dtype = DataType::Union(out_fields.clone());
        let arrow_dtype = dtype.to_physical_arrow(CompatLevel::newest());
        let chunks = arrs
            .iter()
            .enumerate()
            .map(|(c, arr)| {
                let cast: Vec<Option<&CastChild>> = cast_children
                    .iter()
                    .map(|v| v.as_ref().map(|v| &v[c]))
                    .collect();
                assemble_chunk(arr, &out_fields, &target, &cast, &arrow_dtype).boxed()
            })
            .collect();
        // SAFETY: the chunks have the physical dtype of `dtype`.
        Ok(unsafe { Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, &dtype) })
    }
}

/// The cast values of a variant in one chunk.
#[derive(Clone)]
struct CastChild {
    values: ArrayRef,
    /// Whether `values` is the whole child cast as is, rather than its referenced values in slot
    /// order with nulls for null slots.
    whole_child: bool,
}

fn check_dropped_variants(
    arr: &LargeUnionArray,
    from_fields: &[Field],
    to_fields: &[Field],
    target: &[Option<usize>],
    options: CastOptions,
) -> PolarsResult<()> {
    if !options.is_strict() {
        return Ok(());
    }
    for (i, &t) in arr.types().iter().enumerate() {
        polars_ensure!(
            target[t as usize].is_some() || !arr.is_valid(i),
            InvalidOperation: "cannot cast {} to {}: variant `{}` is not in the target",
            DataType::Union(from_fields.to_vec()), DataType::Union(to_fields.to_vec()),
            from_fields[t as usize].name()
        );
    }
    Ok(())
}

/// The values of variant `s` to cast, and whether they are the whole child.
fn variant_values(arr: &LargeUnionArray, s: usize) -> (bool, ArrayRef) {
    let child = &arr.fields()[s];
    let types = arr.types();
    let valid_count = (0..arr.len())
        .filter(|&i| types[i] as usize == s && arr.is_valid(i))
        .count();
    // Offsets increase strictly, so if valid slots reference as many values as the child holds,
    // they reference all of it, in order, and the child can be cast as is.
    if valid_count == child.len() {
        return (true, child.clone());
    }
    // Otherwise cast only the values the variant's slots reference, so a strict cast ignores
    // values the union does not hold.
    let mut positions = ChildPositions::new(child.len());
    for (i, (&ty, &o)) in types.iter().zip(arr.offsets().iter()).enumerate() {
        if ty as usize == s {
            positions.push(arr.is_valid(i).then_some(o as usize));
        }
    }
    // SAFETY: the offsets of the union are in bounds.
    (false, unsafe { gather_child(child.as_ref(), positions) })
}

fn assemble_chunk(
    arr: &LargeUnionArray,
    to_fields: &[Field],
    target: &[Option<usize>],
    cast: &[Option<&CastChild>],
    arrow_dtype: &ArrowDataType,
) -> LargeUnionArray {
    let types = arr.types();
    let offsets = arr.offsets();
    let is_valid = |i: usize| arr.is_valid(i);

    let mut children: Vec<Option<ArrayRef>> = vec![None; to_fields.len()];
    let mut new_offsets: Option<Vec<i64>> = None;
    let mut validity: Option<MutableBitmap> = None;
    let mut null_slot = |i: usize| {
        validity
            .get_or_insert_with(|| match arr.validity() {
                Some(v) => v.clone().make_mut(),
                None => MutableBitmap::from_len_set(arr.len()),
            })
            .set(i, false)
    };
    for (s, t) in target.iter().enumerate() {
        let Some(t) = *t else { continue };
        let Some(cast) = cast[s] else {
            children[t] = Some(arr.fields()[s].clone());
            continue;
        };
        let values = &cast.values;

        // A value a non-strict cast turned into a null nulls its slot too.
        if cast.whole_child {
            if values.has_nulls() {
                for (i, (&ty, &o)) in types.iter().zip(offsets.iter()).enumerate() {
                    if ty as usize == s && !values.is_valid(o as usize) {
                        null_slot(i);
                    }
                }
            }
        } else {
            let out_offsets = new_offsets.get_or_insert_with(|| offsets.as_slice().to_vec());
            let mut k = 0;
            for (i, &ty) in types.iter().enumerate() {
                if ty as usize == s {
                    out_offsets[i] = k as i64;
                    if !values.is_valid(k) && is_valid(i) {
                        null_slot(i);
                    }
                    k += 1;
                }
            }
        }
        children[t] = Some(values.clone());
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

    if types.iter().any(|&t| target[t as usize].is_none()) {
        let mut v =
            validity.map_or_else(|| MutableBitmap::from_len_set(arr.len()), |v| v.make_mut());
        for (i, &t) in types.iter().enumerate() {
            if target[t as usize].is_none() {
                v.set(i, false);
            }
        }
        let offsets = new_offsets.as_deref().unwrap_or(offsets.as_slice());
        // SAFETY: every valid slot has a variant of the target and an in-bounds offset.
        return unsafe {
            gather_union_slots(
                arrow_dtype,
                &new_types,
                offsets,
                &children,
                Some(v.freeze()),
                0..arr.len(),
            )
        };
    }

    let offsets = match new_offsets {
        Some(o) => o.into(),
        None => offsets.clone(),
    };
    // SAFETY: re-cast variants have offsets `0..k` over a child of length `k` or keep theirs over
    // a cast child of the same length, and only slots whose value became null were made null.
    unsafe {
        LargeUnionArray::new_unchecked(arrow_dtype.clone(), new_types, offsets, children, validity)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn int_str() -> UnionChunked {
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
        assert!(cast_offsets_shared(&DataType::Union(vec![
            Field::new("b".into(), DataType::String),
            Field::new("a".into(), DataType::Int64),
        ])));
    }

    #[test]
    fn whole_child_cast_shares_offsets() {
        assert!(cast_offsets_shared(&DataType::Union(vec![
            Field::new("a".into(), DataType::Float64),
            Field::new("b".into(), DataType::String),
        ])));
    }
}
