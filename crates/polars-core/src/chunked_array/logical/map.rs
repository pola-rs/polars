use arrow::bitmap::Bitmap;
use arrow::offset::OffsetsBuffer;

use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::ops::row_encode::encode_rows_unordered;
use crate::prelude::*;

/// A `Map` backed by a `List(Struct {key, value})` [`Series`].
///
/// Keys are unique and non-null within each row. Equality is entry-order-sensitive.
#[derive(Clone)]
pub struct MapChunked {
    dtype: DataType,
    storage: Series,
}

impl MapChunked {
    /// # Safety
    /// `dtype` must be a [`DataType::Map`] matching `storage`, with unique,
    /// non-null keys in every row.
    pub unsafe fn from_storage_unchecked(dtype: DataType, storage: Series) -> Self {
        debug_assert_eq!(
            dtype.map_entries_list_dtype().as_ref(),
            Some(storage.dtype())
        );
        Self { dtype, storage }
    }

    /// Validate and canonicalize a map's storage.
    pub fn try_from_storage(dtype: DataType, storage: Series) -> PolarsResult<Self> {
        let DataType::Map(key, _) = &dtype else {
            polars_bail!(InvalidOperation: "`{dtype}` is not a Map dtype");
        };
        key.ensure_valid_map_key()?;

        let storage_dtype = dtype.map_entries_list_dtype().unwrap();
        polars_ensure!(
            storage.dtype() == &storage_dtype,
            InvalidOperation: "expected `{storage_dtype}` storage for `{dtype}`, got `{}`",
            storage.dtype()
        );

        let storage = canonicalize_map_storage(&storage)?.unwrap_or(storage);
        Ok(Self { dtype, storage })
    }

    pub fn name(&self) -> &PlSmallStr {
        self.storage.name()
    }

    pub fn rename(&mut self, name: PlSmallStr) {
        self.storage.rename(name);
    }

    pub fn field(&self) -> Field {
        Field::new(self.storage.name().clone(), self.dtype.clone())
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn key_dtype(&self) -> &DataType {
        match &self.dtype {
            DataType::Map(key, _) => key,
            _ => unreachable!("MapChunked must have DataType::Map"),
        }
    }

    pub fn value_dtype(&self) -> &DataType {
        match &self.dtype {
            DataType::Map(_, value) => value,
            _ => unreachable!("MapChunked must have DataType::Map"),
        }
    }

    pub fn entries_dtype(&self) -> DataType {
        DataType::map_entries(self.key_dtype().clone(), self.value_dtype().clone())
    }

    pub fn storage(&self) -> &Series {
        &self.storage
    }

    /// Mutable access is crate-private to protect the key invariants.
    pub(crate) fn storage_mut(&mut self) -> &mut Series {
        &mut self.storage
    }

    pub fn into_storage(self) -> Series {
        self.storage
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn get_any_value(&self, _i: usize) -> PolarsResult<AnyValue<'_>> {
        todo!("AnyValue::Map")
    }

    pub fn cast_with_options(
        &self,
        _dtype: &DataType,
        _options: CastOptions,
    ) -> PolarsResult<Series> {
        todo!("MapChunked::cast_with_options")
    }
}

/// Canonicalize this map's entries using first-position/last-value semantics.
/// Returns `None` if unchanged; use [`canonicalize_maps`] to recurse into nested maps.
pub(crate) fn canonicalize_map_storage(storage: &Series) -> PolarsResult<Option<Series>> {
    let DataType::List(entries_dtype) = storage.dtype() else {
        unreachable!("map storage must be List(Struct {{key, value}})")
    };
    let DataType::Struct(entry_fields) = entries_dtype.as_ref() else {
        unreachable!("map storage must be List(Struct {{key, value}})")
    };
    let list_ca = storage.list().unwrap();
    let mut new_chunks = Vec::with_capacity(list_ca.chunks().len());
    let mut any_duplicates = false;

    for chunk in list_ca.downcast_iter() {
        match canonicalize_list_chunk(chunk, entry_fields)? {
            Some(canonicalized) => {
                any_duplicates = true;
                new_chunks.push(canonicalized);
            },
            None => new_chunks.push(chunk.clone().boxed()),
        }
    }

    if !any_duplicates {
        return Ok(None);
    }

    Ok(Some(unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), new_chunks, storage.dtype())
    }))
}

/// Canonicalize every nested `Map` bottom-up, returning `None` if unchanged.
pub fn canonicalize_maps(series: &Series) -> PolarsResult<Option<Series>> {
    if !series.dtype().contains_map() {
        return Ok(None);
    }

    match series.dtype() {
        DataType::Map(_, _) => {
            let map = series.map().unwrap();

            // Parent keys are row-encoded, so canonicalize nested maps first.
            let nested = canonicalize_maps(map.storage())?;
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
            let mut new_chunks = Vec::with_capacity(ca.chunks().len());
            let mut changed = false;

            // Canonicalize each values array separately to preserve chunk offsets.
            for chunk in ca.downcast_iter() {
                match canonicalize_maps(&chunk_values(chunk.values().clone(), inner_dtype))? {
                    Some(values) => {
                        changed = true;
                        new_chunks.push(
                            LargeListArray::new(
                                chunk.dtype().clone(),
                                chunk.offsets().clone(),
                                single_chunk(values),
                                chunk.validity().cloned(),
                            )
                            .boxed(),
                        );
                    },
                    None => new_chunks.push(chunk.clone().boxed()),
                }
            }

            rebuild(series, new_chunks, changed)
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(inner_dtype, _) => {
            let ca = series.array().unwrap();
            let mut new_chunks = Vec::with_capacity(ca.chunks().len());
            let mut changed = false;

            for chunk in ca.downcast_iter() {
                match canonicalize_maps(&chunk_values(chunk.values().clone(), inner_dtype))? {
                    Some(values) => {
                        changed = true;
                        new_chunks.push(
                            FixedSizeListArray::new(
                                chunk.dtype().clone(),
                                chunk.len(),
                                single_chunk(values),
                                chunk.validity().cloned(),
                            )
                            .boxed(),
                        );
                    },
                    None => new_chunks.push(chunk.clone().boxed()),
                }
            }

            rebuild(series, new_chunks, changed)
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => {
            let ca = series.struct_().unwrap();
            let mut changed = false;
            // `try_apply_fields` carries the outer validity across for us.
            let out = ca.try_apply_fields(|field| match canonicalize_maps(field)? {
                Some(new_field) => {
                    changed = true;
                    Ok(new_field)
                },
                None => Ok(field.clone()),
            })?;

            Ok(changed.then(|| out.into_series()))
        },
        #[cfg(feature = "dtype-extension")]
        DataType::Extension(typ, _) => {
            let ext = series.ext().unwrap();
            Ok(canonicalize_maps(ext.storage())?.map(|s| s.into_extension(typ.clone())))
        },
        _ => Ok(None),
    }
}

fn chunk_values(values: ArrayRef, dtype: &DataType) -> Series {
    unsafe { Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![values], dtype) }
}

fn single_chunk(series: Series) -> ArrayRef {
    series.rechunk().chunks()[0].clone()
}

fn rebuild(
    series: &Series,
    new_chunks: Vec<ArrayRef>,
    changed: bool,
) -> PolarsResult<Option<Series>> {
    if !changed {
        return Ok(None);
    }

    Ok(Some(unsafe {
        Series::from_chunks_and_dtype_unchecked(series.name().clone(), new_chunks, series.dtype())
    }))
}

/// Returns `None` when `arr` has no duplicate keys in any row.
fn canonicalize_list_chunk(
    arr: &LargeListArray,
    entry_fields: &[Field],
) -> PolarsResult<Option<ArrayRef>> {
    let entries = arr.values();
    let entries = entries.as_any().downcast_ref::<StructArray>().unwrap();
    let key_arr = entries.values()[0].clone();
    let value_arr = entries.values()[1].clone();

    // Preserve the canonical field names when rebuilding the entries struct.
    let keys = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            entry_fields[0].name.clone(),
            vec![key_arr],
            entry_fields[0].dtype(),
        )
    };

    // Match Polars grouping/join equality without relying on hashes alone.
    let encoded = encode_rows_unordered(&[keys.clone().into_column()])?;
    let encoded = encoded.rechunk();
    let encoded = encoded.downcast_iter().next().unwrap();

    let offsets = arr.offsets().as_slice();
    // Keep each key's first position but take its last value.
    let mut key_idx: Vec<IdxSize> = Vec::new();
    let mut value_idx: Vec<IdxSize> = Vec::new();
    let mut new_offsets: Vec<i64> = Vec::with_capacity(offsets.len());
    new_offsets.push(0);

    // Reuse the table across rows to bound state by the widest row.
    let mut seen: PlHashMap<&[u8], usize> = PlHashMap::new();
    let mut any_duplicates = false;

    for row in 0..arr.len() {
        seen.clear();
        for i in offsets[row] as usize..offsets[row + 1] as usize {
            let key = unsafe { encoded.value_unchecked(i) };
            if let Some(&slot) = seen.get(key) {
                any_duplicates = true;
                value_idx[slot] = i as IdxSize;
            } else {
                seen.insert(key, key_idx.len());
                key_idx.push(i as IdxSize);
                value_idx.push(i as IdxSize);
            }
        }
        new_offsets.push(key_idx.len() as i64);
    }

    if !any_duplicates {
        return Ok(None);
    }

    let values = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            entry_fields[1].name.clone(),
            vec![value_arr],
            entry_fields[1].dtype(),
        )
    };
    let key_out = unsafe { keys.take_slice_unchecked(&key_idx) };
    let value_out = unsafe { values.take_slice_unchecked(&value_idx) };

    // Preserve entry validity from each key's first occurrence.
    let validity = entries.validity().map(|validity| {
        key_idx
            .iter()
            .map(|&i| validity.get_bit(i as usize))
            .collect::<Bitmap>()
    });

    let out_len = key_idx.len();
    let new_entries =
        StructChunked::from_series(PlSmallStr::EMPTY, out_len, [key_out, value_out].iter())?
            .with_outer_validity(validity);

    let new_entries = new_entries.into_series().rechunk().chunks()[0].clone();
    let new_offsets = unsafe { OffsetsBuffer::new_unchecked(new_offsets.into()) };

    Ok(Some(
        LargeListArray::new(
            arr.dtype().clone(),
            new_offsets,
            new_entries,
            arr.validity().cloned(),
        )
        .boxed(),
    ))
}
