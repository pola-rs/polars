//! JSON support for `Map`. A Map is a JSON object, so its keys must be strings. Readers decode
//! a Map as `List(Struct {key: String, value})` in source order and rebuild it with
//! [`Series::from_json_decoded`].
#[cfg(feature = "dtype-map")]
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};

use crate::chunked_array::cast::CastOptions;
#[cfg(feature = "dtype-map")]
use crate::chunked_array::logical::try_apply_map_entries;
use crate::prelude::*;

impl DataType {
    /// Reject nested `Map`s whose keys are not strings, which JSON object keys must be.
    pub fn ensure_json_map_keys(&self) -> PolarsResult<()> {
        use DataType as D;
        match self {
            #[cfg(feature = "dtype-map")]
            D::Map(key, value) => {
                polars_ensure!(
                    is_json_map_key(key),
                    ComputeError:
                    "JSON only supports Map keys of type String, Categorical or Enum, got `{key}`\n\nConsider casting the keys to String, or `Expr.map.entries` to use the entries as a list of structs instead."
                );
                value.ensure_json_map_keys()
            },
            D::List(inner) => inner.ensure_json_map_keys(),
            #[cfg(feature = "dtype-array")]
            D::Array(inner, _) => inner.ensure_json_map_keys(),
            #[cfg(feature = "dtype-struct")]
            D::Struct(fields) => fields
                .iter()
                .try_for_each(|field| field.dtype.ensure_json_map_keys()),
            #[cfg(feature = "dtype-extension")]
            D::Extension(_, storage) => storage.ensure_json_map_keys(),
            _ => Ok(()),
        }
    }

    /// The dtype a JSON reader decodes into before [`Series::from_json_decoded`].
    ///
    /// Maps become `List(Struct {key: String, value})`, Arrays containing a Map become Lists
    /// and Enum/Categorical leaves become String.
    pub fn json_map_decode_dtype(&self) -> DataType {
        use DataType as D;
        match self {
            #[cfg(feature = "dtype-map")]
            D::Map(_, value) => {
                D::Map(Box::new(D::String), Box::new(value.json_map_decode_dtype()))
                    .map_storage_dtype()
                    .unwrap()
            },
            D::List(inner) => D::List(Box::new(inner.json_map_decode_dtype())),
            #[cfg(feature = "dtype-array")]
            D::Array(inner, _) if inner.contains_map() => {
                D::List(Box::new(inner.json_map_decode_dtype()))
            },
            #[cfg(feature = "dtype-array")]
            D::Array(inner, width) => D::Array(Box::new(inner.json_map_decode_dtype()), *width),
            #[cfg(feature = "dtype-struct")]
            D::Struct(fields) => D::Struct(
                fields
                    .iter()
                    .map(|f| Field::new(f.name.clone(), f.dtype.json_map_decode_dtype()))
                    .collect(),
            ),
            #[cfg(feature = "dtype-categorical")]
            D::Enum(..) | D::Categorical(..) => D::String,
            dt => dt.clone(),
        }
    }
}

#[cfg(feature = "dtype-map")]
fn is_json_map_key(dtype: &DataType) -> bool {
    match dtype {
        DataType::String => true,
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(..) | DataType::Enum(..) => true,
        _ => false,
    }
}

impl Series {
    /// Build `target` from a series decoded as [`DataType::json_map_decode_dtype`].
    ///
    /// Map keys are cast to the key dtype and deduplicated with first-position/last-value
    /// semantics. With `ignore_errors`, rows with an unknown Enum key or a wrong Array width
    /// become null.
    pub fn from_json_decoded(self, target: &DataType, ignore_errors: bool) -> PolarsResult<Series> {
        #[cfg(feature = "dtype-map")]
        if target.contains_map() {
            target.ensure_json_map_keys()?;
            let decoded = cast_leaf(self, &target.json_map_decode_dtype(), ignore_errors)?;
            return from_json_decoded_rec(&decoded, target, ignore_errors);
        }
        cast_leaf(self, target, ignore_errors)
    }
}

fn cast_leaf(s: Series, target: &DataType, ignore_errors: bool) -> PolarsResult<Series> {
    if s.dtype() == target {
        return Ok(s);
    }
    let options = if ignore_errors {
        CastOptions::NonStrict
    } else {
        CastOptions::Strict
    };
    s.cast_with_options(target, options)
}

#[cfg(feature = "dtype-map")]
fn from_json_decoded_rec(
    series: &Series,
    target: &DataType,
    ignore_errors: bool,
) -> PolarsResult<Series> {
    if !target.contains_map() {
        return cast_leaf(series.clone(), target, ignore_errors);
    }

    match target {
        DataType::Map(key_dtype, value_dtype) => {
            let mut ok_keys: Option<Bitmap> = None;
            let storage = try_apply_map_entries(series.list()?, |key, value| {
                let decoded = decode_keys(key, key_dtype)?;
                if decoded.null_count() > key.null_count() {
                    let ok = (decoded.is_not_null() | key.is_null()).rechunk().into_owned();
                    if !ignore_errors {
                        let idx = ok.iter().position(|ok| ok == Some(false)).unwrap();
                        let key = key.str()?.get(idx).unwrap();
                        polars_bail!(
                            ComputeError:
                            "cannot decode JSON object key \"{key}\" as Map key of type `{key_dtype}`"
                        );
                    }
                    ok_keys = Some(ok.downcast_as_array().values().clone());
                }
                let value = from_json_decoded_rec(value, value_dtype, ignore_errors)?;
                Ok((decoded, value))
            })?
            .into_series();

            let storage = match ok_keys {
                Some(ok) => {
                    null_rows_where(&storage, |start, len| ok.null_count_range(start, len) > 0)
                },
                None => storage,
            };
            Ok(MapChunked::try_from_storage(target.clone(), storage)?.into_series())
        },
        DataType::List(inner) => {
            let ca = series.list()?;
            let values = from_json_decoded_rec(&ca.get_inner(), inner, ignore_errors)?;
            Ok(ca.with_inner_values(&values).into_series())
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(inner, width) => {
            let ca = series.list()?;
            let values = from_json_decoded_rec(&ca.get_inner(), inner, ignore_errors)?;
            let mut list = ca.with_inner_values(&values).into_series();
            if ignore_errors {
                list = null_rows_where(&list, |_, len| len != *width);
            }
            list.cast_with_options(target, CastOptions::Strict)
        },
        DataType::Struct(fields) => {
            let ca = series.struct_()?;
            let out = ca.try_apply_fields(|field| {
                match fields.iter().find(|f| f.name() == field.name()) {
                    Some(f) => from_json_decoded_rec(field, f.dtype(), ignore_errors),
                    None => Ok(field.clone()),
                }
            })?;
            Ok(out.into_series())
        },
        _ => cast_leaf(series.clone(), target, ignore_errors),
    }
}

/// Null the valid rows of a list series for which `bad(start, len)` holds. `start` indexes
/// the concatenated offset windows of all chunks.
#[cfg(feature = "dtype-map")]
fn null_rows_where(list: &Series, mut bad: impl FnMut(usize, usize) -> bool) -> Series {
    let ca = list.list().unwrap();
    let mut validity = BitmapBuilder::with_capacity(ca.len());
    let mut changed = false;
    let mut base = 0;
    for arr in ca.downcast_iter() {
        let offsets = arr.offsets();
        let first = *offsets.first() as usize;
        for row in 0..arr.len() {
            let (start, end) = offsets.start_end(row);
            let valid = arr.is_valid(row);
            let keep = valid && !bad(base + start - first, end - start);
            changed |= valid != keep;
            validity.push(keep);
        }
        base += offsets.range() as usize;
    }

    if changed {
        list.with_validity(Some(validity.freeze()))
    } else {
        list.clone()
    }
}

/// Cast JSON object keys to `dtype`; keys that are not valid categories become null.
#[cfg(feature = "dtype-map")]
fn decode_keys(keys: &Series, dtype: &DataType) -> PolarsResult<Series> {
    if keys.dtype() == dtype {
        return Ok(keys.clone());
    }
    keys.cast_with_options(dtype, CastOptions::NonStrict)
}
