use arrow::offset::OffsetsBuffer;
use polars_compute::gather::bitmap::take_bitmap_unchecked;
use polars_compute::gather::take_unchecked;

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
/// Returns `None` if unchanged; use [`Series::canonicalize_maps`] to also reach
/// maps nested inside the keys or values.
pub(crate) fn canonicalize_map_storage(storage: &Series) -> PolarsResult<Option<Series>> {
    let DataType::List(entries_dtype) = storage.dtype() else {
        unreachable!("map storage must be List(Struct {{key, value}})")
    };
    let DataType::Struct(entry_fields) = entries_dtype.as_ref() else {
        unreachable!("map storage must be List(Struct {{key, value}})")
    };
    let [key_field, _] = entry_fields.as_slice() else {
        unreachable!("map entries must have two fields")
    };
    let list_ca = storage.list().unwrap();
    // Only allocated once a chunk actually needs repair.
    let mut new_chunks: Option<Vec<ArrayRef>> = None;

    for (i, chunk) in list_ca.downcast_iter().enumerate() {
        match canonicalize_list_chunk(chunk, key_field.dtype())? {
            Some(canonicalized) => new_chunks
                .get_or_insert_with(|| list_ca.chunks()[..i].to_vec())
                .push(canonicalized),
            None => {
                if let Some(new_chunks) = new_chunks.as_mut() {
                    new_chunks.push(chunk.clone().boxed());
                }
            },
        }
    }

    Ok(new_chunks.map(|chunks| unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    }))
}

struct CanonicalMapIndices {
    first_keys: IdxArr,
    last_values: IdxArr,
    offsets: OffsetsBuffer<i64>,
}

/// Build take indices only when a row contains duplicate keys.
fn canonical_map_indices(keys: &BinaryArray<i64>, offsets: &[i64]) -> Option<CanonicalMapIndices> {
    let mut seen = PlHashSet::new();
    let has_duplicates = offsets.windows(2).any(|range| {
        seen.clear();
        (range[0] as usize..range[1] as usize)
            .any(|i| !seen.insert(unsafe { keys.value_unchecked(i) }))
    });
    if !has_duplicates {
        return None;
    }

    let n_entries = (offsets[offsets.len() - 1] - offsets[0]) as usize;
    let mut key_idx = Vec::with_capacity(n_entries);
    let mut value_idx = Vec::with_capacity(n_entries);
    let mut new_offsets = Vec::with_capacity(offsets.len());
    new_offsets.push(0);

    let mut slots = PlHashMap::new();
    for range in offsets.windows(2) {
        slots.clear();
        for i in range[0] as usize..range[1] as usize {
            let key = unsafe { keys.value_unchecked(i) };
            if let Some(&slot) = slots.get(key) {
                value_idx[slot] = i as IdxSize;
            } else {
                slots.insert(key, key_idx.len());
                key_idx.push(i as IdxSize);
                value_idx.push(i as IdxSize);
            }
        }
        new_offsets.push(key_idx.len() as i64);
    }

    Some(CanonicalMapIndices {
        first_keys: IdxArr::from_vec(key_idx),
        last_values: IdxArr::from_vec(value_idx),
        offsets: unsafe { OffsetsBuffer::new_unchecked(new_offsets.into()) },
    })
}

/// Gather the deduplicated entries and rebuild the list chunk around them.
fn gather_entries(
    arr: &LargeListArray,
    entries: &StructArray,
    indices: CanonicalMapIndices,
) -> ArrayRef {
    let [key_arr, value_arr] = entries.values() else {
        unreachable!("map entries must have two arrays")
    };
    let CanonicalMapIndices {
        first_keys,
        last_values,
        offsets,
    } = indices;

    // Entry validity follows the keys, i.e. each key's first occurrence.
    let validity = entries
        .validity()
        .map(|validity| unsafe { take_bitmap_unchecked(validity, first_keys.values()) });

    let new_entries = StructArray::new(
        entries.dtype().clone(),
        first_keys.len(),
        vec![
            unsafe { take_unchecked(key_arr.as_ref(), &first_keys) },
            unsafe { take_unchecked(value_arr.as_ref(), &last_values) },
        ],
        validity,
    );

    LargeListArray::new(
        arr.dtype().clone(),
        offsets,
        new_entries.boxed(),
        arr.validity().cloned(),
    )
    .boxed()
}

/// Returns `None` when `arr` has no duplicate keys in any row.
fn canonicalize_list_chunk(
    arr: &LargeListArray,
    key_dtype: &DataType,
) -> PolarsResult<Option<ArrayRef>> {
    let entries = arr.values();
    let entries = entries.as_any().downcast_ref::<StructArray>().unwrap();
    let [key_arr, _] = entries.values() else {
        unreachable!("map entries must have two arrays")
    };

    // A Series only to row-encode the keys: that needs the *logical* dtype, so a
    // categorical key carries its mapping. The name never reaches the output.
    let keys = unsafe {
        Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![key_arr.clone()], key_dtype)
    };

    // Match Polars grouping/join equality without relying on hashes alone.
    let encoded = encode_rows_unordered(&[keys.into_column()])?;
    let encoded = encoded.downcast_iter().next().unwrap();
    let Some(indices) = canonical_map_indices(encoded, arr.offsets().as_slice()) else {
        return Ok(None);
    };

    Ok(Some(gather_entries(arr, entries, indices)))
}
