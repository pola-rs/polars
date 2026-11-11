//! Conversion between Arrow's dense `UnionArray` and Polars' `LargeUnionArray`.
use polars_arrow::array::{Array, LargeUnionArray, NullArray, UnionArray};
use polars_arrow::bitmap::{BitmapBuilder, MutableBitmap};
use polars_compute::gather::gather_union_slots;

use crate::prelude::*;

/// Imports dense Arrow union chunks. Variants whose converted dtype is `Null` are dropped and
/// their slots become nulls; with no variant left the result is a `Null` column.
///
/// `convert_children` converts the chunks of one child to physical arrays and their dtype.
pub(crate) fn union_from_arrow(
    chunks: Vec<ArrayRef>,
    mut convert_children: impl FnMut(
        Vec<ArrayRef>,
        &ArrowField,
    ) -> PolarsResult<(Vec<ArrayRef>, DataType)>,
) -> PolarsResult<(Vec<ArrayRef>, DataType)> {
    let ArrowDataType::Union(u) = chunks[0].dtype().to_storage() else {
        unreachable!()
    };
    polars_ensure!(
        u.mode.is_dense(),
        ComputeError: "cannot create series from a sparse Union, only dense unions are supported"
    );
    let arrays: Vec<&UnionArray> = chunks
        .iter()
        .map(|c| c.as_any().downcast_ref::<UnionArray>().unwrap())
        .collect();

    let mut kept_fields = Vec::new();
    let mut kept_children: Vec<Vec<ArrayRef>> = Vec::new();
    let mut new_index = Vec::with_capacity(u.fields.len());
    for (j, field) in u.fields.iter().enumerate() {
        let children = arrays.iter().map(|a| a.fields()[j].clone()).collect();
        let (children, dtype) = convert_children(children, field)?;
        if dtype.is_null() {
            new_index.push(None);
        } else {
            new_index.push(Some(kept_fields.len()));
            kept_fields.push(Field::new(field.name.clone(), dtype));
            kept_children.push(children);
        }
    }

    if kept_fields.is_empty() {
        let chunks = arrays
            .iter()
            .map(|a| NullArray::new(ArrowDataType::Null, a.len()).boxed())
            .collect();
        return Ok((chunks, DataType::Null));
    }

    // Arrow type id -> `Some(new index)`, or `None` for a dropped variant.
    let mut id_map: [Option<Option<usize>>; 128] = [None; 128];
    for (pos, new) in new_index.iter().enumerate() {
        let id = match &u.ids {
            Some(ids) => ids[pos],
            None => pos as i32,
        };
        polars_ensure!((0..128).contains(&id), ComputeError: "union type id {id} is out of range");
        id_map[id as usize] = Some(*new);
    }
    let any_dropped = new_index.iter().any(Option::is_none);
    let types_unchanged = !any_dropped
        && id_map
            .iter()
            .enumerate()
            .all(|(i, m)| m.is_none_or(|m| m == Some(i)));

    let dtype = DataType::Union(UnionFields::try_new(kept_fields)?);
    let arrow_dtype = dtype.to_physical_arrow(CompatLevel::newest());
    let n = kept_children.len();

    let out = arrays
        .iter()
        .enumerate()
        .map(|(c, arr)| {
            let children: Vec<ArrayRef> = kept_children.iter().map(|ch| ch[c].clone()).collect();
            import_chunk(
                arr,
                children,
                &id_map,
                types_unchanged,
                any_dropped,
                n,
                &arrow_dtype,
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok((out, dtype))
}

fn import_chunk(
    arr: &UnionArray,
    children: Vec<ArrayRef>,
    id_map: &[Option<Option<usize>>; 128],
    types_unchanged: bool,
    any_dropped: bool,
    n: usize,
    arrow_dtype: &ArrowDataType,
) -> PolarsResult<ArrayRef> {
    let len = arr.len();
    let types = arr.types();
    let offsets = arr.offsets().unwrap();

    let lift = any_dropped || children.iter().any(|c| c.has_nulls());
    let mut validity = lift.then(|| BitmapBuilder::with_capacity(len));
    let mut new_types = (!types_unchanged).then(|| Vec::with_capacity(len));
    let mut wide = Vec::with_capacity(len);
    let mut next = vec![0i64; n];
    let mut rebuild = any_dropped;

    for (&id, &o) in types.iter().zip(offsets.iter()) {
        let mapped = usize::try_from(id)
            .ok()
            .and_then(|id| id_map.get(id).copied().flatten());
        let Some(mapped) = mapped else {
            polars_bail!(ComputeError: "union type id {id} is out of range");
        };
        let (t, o, valid) = match mapped {
            Some(t) => {
                let o = o as i64;
                polars_ensure!(
                    o >= 0 && (o as usize) < children[t].len(),
                    ComputeError: "union offset {o} is out of bounds for variant {t}"
                );
                if o < next[t] {
                    rebuild = true;
                } else {
                    next[t] = o + 1;
                }
                (t, o, lift && children[t].is_valid(o as usize))
            },
            None => (0, 0, false),
        };
        if let Some(nt) = &mut new_types {
            nt.push(t as i8);
        }
        wide.push(o);
        if let Some(v) = &mut validity {
            v.push(valid);
        }
    }
    let validity = validity.and_then(|v| v.into_opt_validity());

    let out = if rebuild {
        let types: &[i8] = new_types.as_deref().unwrap_or(types);
        // SAFETY: valid slots were bounds-checked above.
        unsafe { gather_union_slots(arrow_dtype, types, &wide, &children, validity, 0..len) }
    } else {
        let types = match new_types {
            Some(t) => t.into(),
            None => types.clone(),
        };
        // SAFETY: bounds and strict monotonicity were checked above.
        unsafe {
            LargeUnionArray::new_unchecked(
                arrow_dtype.clone(),
                types,
                wide.into(),
                children,
                validity,
            )
        }
    };
    Ok(out.boxed())
}

/// Exports a chunk to a dense Arrow union of `arrow_dtype`. Null slots are written into the
/// validity of the child value they point at, and each child is cut to the range it uses.
///
/// `convert_child(t, child)` converts the physical child of variant `t` to its Arrow form.
pub(crate) fn union_array_to_arrow(
    arr: &LargeUnionArray,
    arrow_dtype: ArrowDataType,
    mut convert_child: impl FnMut(usize, &dyn Array) -> PolarsResult<ArrayRef>,
) -> PolarsResult<UnionArray> {
    let mut children: Vec<ArrayRef> = arr.fields().to_vec();

    if let Some(validity) = arr.validity() {
        let mut masks: Vec<Option<MutableBitmap>> = vec![None; children.len()];
        for i in (!validity).true_idx_iter() {
            let (t, o) = arr.index(i);
            match &mut masks[t] {
                Some(mask) => mask.set(o, false),
                None if children[t].is_valid(o) => {
                    let mut mask = match children[t].validity() {
                        Some(v) => v.clone().make_mut(),
                        None => MutableBitmap::from_len_set(children[t].len()),
                    };
                    mask.set(o, false);
                    masks[t] = Some(mask);
                },
                None => {},
            }
        }
        for (child, mask) in children.iter_mut().zip(masks) {
            if let Some(mask) = mask {
                *child = child.with_validity(Some(mask.freeze()));
            }
        }
    }

    let n = children.len();
    let mut first = vec![-1i64; n];
    let mut last = vec![-1i64; n];
    for (&t, &o) in arr.types().iter().zip(arr.offsets().iter()) {
        let t = t as usize;
        if first[t] < 0 {
            first[t] = o;
        }
        last[t] = o;
    }
    for t in 0..n {
        polars_ensure!(
            last[t] - first[t] <= i32::MAX as i64,
            InvalidOperation:
            "to_arrow() conversion failed: variant {t} of the union spans {} values, which \
            overflows the i32 offsets of the arrow Union type",
            last[t] - first[t] + 1,
        );
    }
    let offsets: Vec<i32> = arr
        .types()
        .iter()
        .zip(arr.offsets().iter())
        .map(|(&t, &o)| (o - first[t as usize]) as i32)
        .collect();

    let children = children
        .into_iter()
        .enumerate()
        .map(|(t, child)| {
            let child = if first[t] < 0 {
                child.sliced(0, 0)
            } else {
                child.sliced(first[t] as usize, (last[t] - first[t] + 1) as usize)
            };
            convert_child(t, child.as_ref())
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    UnionArray::try_new(
        arrow_dtype,
        arr.types().clone(),
        children,
        Some(offsets.into()),
    )
}
