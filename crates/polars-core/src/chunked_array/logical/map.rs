use arrow::offset::OffsetsBuffer;
use polars_compute::gather::take_unchecked;

use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::iterator::PolarsIterator;
use crate::chunked_array::ops::row_encode::encode_rows_unordered;
use crate::prelude::*;

/// A `Map` backed by a `List(Struct {key, value})` [`Series`].
///
/// Map entries and keys are non-null, and keys are unique within each row. Equality
/// is entry-order-sensitive.
#[derive(Clone)]
pub struct MapChunked {
    dtype: DataType,
    storage: Series,
}

impl MapChunked {
    /// # Safety
    /// `dtype` must be a [`DataType::Map`] matching `storage`, with non-null entries
    /// and unique, non-null keys in every row.
    pub unsafe fn from_storage_unchecked(dtype: DataType, storage: Series) -> Self {
        debug_assert_eq!(
            dtype.map_entries_list_dtype().as_ref(),
            Some(storage.dtype())
        );
        Self { dtype, storage }
    }

    /// Validate map storage and canonicalize duplicate keys.
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

    /// Mutable access is crate-private to protect the map invariants.
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

    /// One freshly-allocated `AnyValue::Map` per row.
    pub fn any_value_iter(&self) -> impl PolarsIterator<Item = AnyValue<'_>> {
        self.storage
            .list()
            .unwrap()
            .series_iter()
            .map(|entries| match entries {
                Some(entries) => AnyValue::Map(entries),
                None => AnyValue::Null,
            })
    }

    pub fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        Ok(map_av(self.storage.get(i)?))
    }

    /// # Safety
    /// `i` must be in bounds.
    pub unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        map_av(unsafe { self.storage.get_unchecked(i) })
    }

    pub fn cast_with_options(
        &self,
        dtype: &DataType,
        options: CastOptions,
    ) -> PolarsResult<Series> {
        if dtype == &self.dtype {
            return Ok(self.clone().into_series());
        }

        match dtype {
            DataType::Map(key, value) => self.cast_entries(key, value, options),
            DataType::List(_) => self.storage.cast_with_options(dtype, options),
            _ => polars_bail!(InvalidOperation: "cannot cast `{}` to `{dtype}`", self.dtype),
        }
    }

    /// Cast the entry children, leaving the offsets and outer validity untouched.
    fn cast_entries(
        &self,
        to_key: &DataType,
        to_value: &DataType,
        options: CastOptions,
    ) -> PolarsResult<Series> {
        let cast_key = self.key_dtype().matches_schema_type(to_key).map_err(
            |_| polars_err!(InvalidOperation: "cannot cast Map key `{}` to `{to_key}`", self.key_dtype()),
        )?;

        let storage = self.storage.rechunk();
        let arr = storage.list().unwrap().downcast_as_array();
        let entries = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                MAP_ENTRIES_NAME.clone(),
                vec![arr.values().clone()],
                &self.entries_dtype(),
            )
        };
        let entries = entries.struct_().unwrap().fields_as_series();
        let [key, value] = entries.as_slice() else {
            unreachable!("map entries must have two fields")
        };

        let key = if cast_key {
            key.cast_with_options(to_key, options)?
        } else {
            key.clone()
        };
        let value = value.cast_with_options(to_value, options)?;
        let dtype = DataType::Map(
            Box::new(key.dtype().clone()),
            Box::new(value.dtype().clone()),
        );
        let storage = map_storage_from_entries(self.name().clone(), arr, key, value)?;

        if cast_key {
            // Even with matching key schemas, we are allowed to rescale Decimals, which might collapse distinct keys into duplicates.
            Ok(Self::try_from_storage(dtype, storage)?.into_series())
        } else {
            Ok(unsafe { Self::from_storage_unchecked(dtype, storage) }.into_series())
        }
    }
}

/// Rebuild map storage around new entry children, reusing `arr`'s offsets and validity.
pub(crate) fn map_storage_from_entries(
    name: PlSmallStr,
    arr: &LargeListArray,
    key: Series,
    value: Series,
) -> PolarsResult<Series> {
    let key = key.with_name(MAP_KEY_NAME.clone());
    let value = value.with_name(MAP_VALUE_NAME.clone());
    let dtype = DataType::map_entries(key.dtype().clone(), value.dtype().clone());

    let entries = StructChunked::from_series(
        MAP_ENTRIES_NAME.clone(),
        arr.values().len(),
        [&key, &value].into_iter(),
    )?
    .into_series()
    .rechunk();
    let entries = entries.array_ref(0).clone();

    let new_arr = LargeListArray::new(
        LargeListArray::default_datatype(entries.dtype().clone()),
        arr.offsets().clone(),
        entries,
        arr.validity().cloned(),
    );
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(
            name,
            vec![new_arr.boxed()],
            &DataType::List(Box::new(dtype)),
        )
    })
}

fn map_av(av: AnyValue<'_>) -> AnyValue<'_> {
    match av {
        AnyValue::List(entries) => AnyValue::Map(entries),
        AnyValue::Null => AnyValue::Null,
        av => unreachable!("map storage must yield a list, got {av:?}"),
    }
}

/// Reject null entries and keys, then canonicalize duplicates by keeping each key's
/// first position and last value.
///
/// Returns `None` if unchanged; use [`Series::canonicalize_maps`] to also reach maps
/// nested inside the keys or values.
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
    // Allocate only after the first changed chunk.
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

    let new_entries = StructArray::new(
        entries.dtype().clone(),
        first_keys.len(),
        vec![
            unsafe { take_unchecked(key_arr.as_ref(), &first_keys) },
            unsafe { take_unchecked(value_arr.as_ref(), &last_values) },
        ],
        // Entry validity is all true.
        None,
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

    // Entry and key validity are independent; null values are allowed.
    polars_ensure!(
        entries.null_count() == 0,
        InvalidOperation: "Map entries cannot be null"
    );
    polars_ensure!(
        key_arr.null_count() == 0,
        InvalidOperation: "Map keys cannot be null"
    );

    // Row encoding uses the logical dtype and matches Polars key equality.
    let keys = unsafe {
        Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![key_arr.clone()], key_dtype)
    };
    let encoded = encode_rows_unordered(&[keys.into_column()])?;
    let encoded = encoded.downcast_iter().next().unwrap();

    let Some(indices) = canonical_map_indices(encoded, arr.offsets().as_slice()) else {
        return Ok(None);
    };

    Ok(Some(gather_entries(arr, entries, indices)))
}
