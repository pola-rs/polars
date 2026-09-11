#[cfg(feature = "dtype-map")]
use crate::chunked_array::logical::{
    CanonicalizeMode, canonicalize_map_storage, compact_null_map_rows,
};
use crate::prelude::*;

/// Operation applied to each nested Map.
#[cfg(feature = "dtype-map")]
#[derive(Clone, Copy)]
enum MapPass {
    /// Deduplicate keys using first-position/last-value semantics.
    Canonicalize,
    /// Drop entries under null rows.
    CompactNullRows,
}

impl Series {
    /// Canonicalize all nested `Map`s bottom-up using first-position/last-value
    /// semantics. Returns `None` if unchanged.
    pub fn canonicalize_maps(&self) -> PolarsResult<Option<Series>> {
        #[cfg(feature = "dtype-map")]
        {
            map_pass(self, MapPass::Canonicalize)
        }
        #[cfg(not(feature = "dtype-map"))]
        {
            Ok(None)
        }
    }

    /// Drop entries under null Map rows at every depth; return `None` if unchanged.
    ///
    /// Aligns physical child layouts for strict-cast validity comparisons.
    pub fn compact_map_null_rows(&self) -> PolarsResult<Option<Series>> {
        #[cfg(feature = "dtype-map")]
        {
            map_pass(self, MapPass::CompactNullRows)
        }
        #[cfg(not(feature = "dtype-map"))]
        {
            Ok(None)
        }
    }
}

#[cfg(feature = "dtype-map")]
fn map_pass(series: &Series, pass: MapPass) -> PolarsResult<Option<Series>> {
    if !series.dtype().contains_map() {
        return Ok(None);
    }

    match series.dtype() {
        DataType::Map(_, _) => {
            let map = series.map().unwrap();

            // Visit children first so canonicalization row-encodes normalized keys.
            let nested = map_pass(map.storage(), pass)?;
            let storage = nested.as_ref().unwrap_or(map.storage());
            let changed = match pass {
                MapPass::Canonicalize => canonicalize_map_storage(storage, CanonicalizeMode::Full)?,
                MapPass::CompactNullRows => {
                    compact_null_map_rows(storage.list().unwrap()).map(IntoSeries::into_series)
                },
            };

            match changed.or(nested) {
                None => Ok(None),
                Some(storage) => Ok(Some(
                    unsafe { MapChunked::from_storage_unchecked(map.dtype().clone(), storage) }
                        .into_series(),
                )),
            }
        },
        DataType::List(_) => {
            let ca = series.list().unwrap();
            Ok(map_pass(&ca.get_inner(), pass)?
                .map(|values| ca.with_inner_values(&values).into_series()))
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => {
            let ca = series.array().unwrap();
            Ok(map_pass(&ca.get_inner(), pass)?
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
                let new_field = map_pass(field, pass)?;
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
            Ok(map_pass(series.ext().unwrap().storage(), pass)?
                .map(|s| s.into_extension(typ.clone())))
        },
        _ => Ok(None),
    }
}
