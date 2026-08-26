#[cfg(feature = "dtype-map")]
use crate::chunked_array::logical::canonicalize_map_storage;
use crate::prelude::*;

impl Series {
    /// Canonicalize all nested `Map`s bottom-up using first-position/last-value
    /// semantics. Returns `None` if unchanged.
    pub fn canonicalize_maps(&self) -> PolarsResult<Option<Series>> {
        #[cfg(feature = "dtype-map")]
        {
            canonicalize_maps_rec(self)
        }
        #[cfg(not(feature = "dtype-map"))]
        {
            Ok(None)
        }
    }
}

#[cfg(feature = "dtype-map")]
fn canonicalize_maps_rec(series: &Series) -> PolarsResult<Option<Series>> {
    if !series.dtype().contains_map() {
        return Ok(None);
    }

    match series.dtype() {
        DataType::Map(_, _) => {
            let map = series.map().unwrap();

            // Parent keys are row-encoded, so canonicalize nested maps first.
            let nested = canonicalize_maps_rec(map.storage())?;
            let storage = nested.as_ref().unwrap_or(map.storage());
            let deduped = canonicalize_map_storage(storage)?;

            match deduped.or(nested) {
                None => Ok(None),
                Some(storage) => Ok(Some(
                    unsafe { MapChunked::from_storage_unchecked(map.dtype().clone(), storage) }
                        .into_series(),
                )),
            }
        },
        DataType::List(inner_dtype) => {
            let ca = series.list().unwrap();
            let mut new_chunks: Option<Vec<ArrayRef>> = None;

            // Canonicalize each values array separately to preserve chunk offsets.
            for (i, chunk) in ca.downcast_iter().enumerate() {
                match canonicalize_maps_rec(&chunk_values(chunk.values().clone(), inner_dtype))? {
                    Some(values) => new_chunks
                        .get_or_insert_with(|| ca.chunks()[..i].to_vec())
                        .push(
                            LargeListArray::new(
                                chunk.dtype().clone(),
                                chunk.offsets().clone(),
                                single_chunk(values),
                                chunk.validity().cloned(),
                            )
                            .boxed(),
                        ),
                    None => {
                        if let Some(new_chunks) = new_chunks.as_mut() {
                            new_chunks.push(chunk.clone().boxed());
                        }
                    },
                }
            }

            Ok(new_chunks.map(|chunks| unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    series.name().clone(),
                    chunks,
                    series.dtype(),
                )
            }))
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(inner_dtype, _) => {
            let ca = series.array().unwrap();
            let mut new_chunks: Option<Vec<ArrayRef>> = None;

            for (i, chunk) in ca.downcast_iter().enumerate() {
                match canonicalize_maps_rec(&chunk_values(chunk.values().clone(), inner_dtype))? {
                    Some(values) => new_chunks
                        .get_or_insert_with(|| ca.chunks()[..i].to_vec())
                        .push(
                            FixedSizeListArray::new(
                                chunk.dtype().clone(),
                                chunk.len(),
                                single_chunk(values),
                                chunk.validity().cloned(),
                            )
                            .boxed(),
                        ),
                    None => {
                        if let Some(new_chunks) = new_chunks.as_mut() {
                            new_chunks.push(chunk.clone().boxed());
                        }
                    },
                }
            }

            Ok(new_chunks.map(|chunks| unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    series.name().clone(),
                    chunks,
                    series.dtype(),
                )
            }))
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => {
            let ca = series.struct_().unwrap();

            // Avoid rebuilding the struct when no field changes.
            let fields = ca.fields_as_series();
            let mut new_fields = Vec::with_capacity(fields.len());
            let mut changed = false;
            for field in &fields {
                let new_field = canonicalize_maps_rec(field)?;
                changed |= new_field.is_some();
                new_fields.push(new_field);
            }

            if !changed {
                return Ok(None);
            }

            // `try_apply_fields` preserves the outer validity.
            let mut new_fields = new_fields.into_iter();
            let out = ca.try_apply_fields(|field| {
                Ok(new_fields.next().unwrap().unwrap_or_else(|| field.clone()))
            })?;

            Ok(Some(out.into_series()))
        },
        #[cfg(feature = "dtype-extension")]
        DataType::Extension(typ, _) => {
            let ext = series.ext().unwrap();
            Ok(canonicalize_maps_rec(ext.storage())?.map(|s| s.into_extension(typ.clone())))
        },
        _ => Ok(None),
    }
}

#[cfg(feature = "dtype-map")]
fn chunk_values(values: ArrayRef, dtype: &DataType) -> Series {
    unsafe { Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![values], dtype) }
}

#[cfg(feature = "dtype-map")]
fn single_chunk(series: Series) -> ArrayRef {
    series.rechunk().chunks()[0].clone()
}
