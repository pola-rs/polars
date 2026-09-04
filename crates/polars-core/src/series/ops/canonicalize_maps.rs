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
        DataType::List(_) => {
            let ca = series.list().unwrap();
            Ok(canonicalize_maps_rec(&ca.get_inner())?
                .map(|values| ca.with_inner_values(&values).into_series()))
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => {
            let ca = series.array().unwrap();
            Ok(canonicalize_maps_rec(&ca.get_inner())?
                .map(|values| ca.with_inner_values(&values).into_series()))
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
