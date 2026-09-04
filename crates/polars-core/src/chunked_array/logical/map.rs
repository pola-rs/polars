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
        debug_assert_eq!(dtype.map_storage_dtype().as_ref(), Some(storage.dtype()));
        debug_assert!(
            dtype.ensure_valid_map_dtype().is_ok(),
            "invalid Map dtype: {dtype}"
        );
        Self { dtype, storage }
    }

    /// Validate map storage and canonicalize duplicate keys.
    pub fn try_from_storage(dtype: DataType, storage: Series) -> PolarsResult<Self> {
        dtype.ensure_valid_map_dtype()?;

        let storage_dtype = dtype.map_storage_dtype().unwrap();
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
        self.dtype.as_map().unwrap().0
    }

    pub fn value_dtype(&self) -> &DataType {
        self.dtype.as_map().unwrap().1
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

    /// The key child, one element per entry across all rows.
    pub fn keys(&self) -> Series {
        unpack_map_entries(&self.entries()).0
    }

    /// The value child, one element per entry across all rows.
    pub fn values(&self) -> Series {
        unpack_map_entries(&self.entries()).1
    }

    /// Replace the value child, keeping the keys and the entry layout.
    pub fn with_values(&self, values: &Series) -> Self {
        let storage = self.storage.list().unwrap();
        let (keys, _) = unpack_map_entries(&storage.get_inner());
        let storage = repack_map_storage(storage, &keys, values).into_series();

        let dtype = DataType::Map(
            Box::new(self.key_dtype().clone()),
            Box::new(values.dtype().clone()),
        );

        unsafe { Self::from_storage_unchecked(dtype, storage) }
    }

    /// The entries of every row, flattened.
    fn entries(&self) -> Series {
        self.storage.list().unwrap().get_inner()
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
        let cast_key = self
            .key_dtype()
            .matches_schema_type(to_key)
            .map_err(|_| {
                polars_err!(InvalidOperation: "cannot cast Map key `{}` to `{to_key}`", self.key_dtype())
            })?;

        let dtype = DataType::Map(Box::new(to_key.clone()), Box::new(to_value.clone()));
        dtype.ensure_valid_map_dtype()?;
        let storage = try_apply_map_entries(self.storage.list().unwrap(), |key, value| {
            // `Series::cast_with_options` only short-circuits an identity cast for
            // primitives, so a nested key or value would be rebuilt for nothing.
            let key = if cast_key {
                key.cast_with_options(to_key, options)?
            } else {
                key.clone()
            };
            let value = if value.dtype() == to_value {
                value.clone()
            } else {
                value.cast_with_options(to_value, options)?
            };
            Ok((key, value))
        })?
        .into_series();

        let storage_dtype = dtype.map_storage_dtype().unwrap();
        polars_ensure!(
            storage.dtype() == &storage_dtype,
            ComputeError: "Map entry transform produced `{}` storage instead of `{storage_dtype}`",
            storage.dtype(),
        );

        if cast_key {
            // Even with matching key schemas, we are allowed to rescale Decimals, which might collapse distinct keys into duplicates.
            Ok(Self::try_from_storage(dtype, storage)?.into_series())
        } else {
            Ok(unsafe { Self::from_storage_unchecked(dtype, storage) }.into_series())
        }
    }
}

/// Require named entry fields outside Arrow and Parquet, whose specifications define
/// entries positionally.
pub(crate) fn ensure_map_entries_dtype(dtype: &DataType) -> PolarsResult<()> {
    let DataType::Struct(fields) = dtype else {
        polars_bail!(InvalidOperation: "Map entries must be `Struct {{key, value}}`, got `{dtype}`")
    };
    // Spell the names out because `Struct` display abbreviates them to `struct[n]`.
    let mut names: Vec<&PlSmallStr> = fields.iter().map(|f| f.name()).collect();
    names.sort();
    polars_ensure!(
        names == [&MAP_KEY_NAME, &MAP_VALUE_NAME],
        InvalidOperation:
        "Map entries must be exactly two fields named `{}` and `{}`, got [{}]",
        MAP_KEY_NAME, MAP_VALUE_NAME,
        fields.iter().map(|f| format!("`{}`", f.name())).collect::<Vec<_>>().join(", "),
    );
    Ok(())
}

fn unpack_map_entries(entries: &Series) -> (Series, Series) {
    let fields = entries.struct_().unwrap().fields_as_series();
    let Ok([first, second]) = <[Series; 2]>::try_from(fields) else {
        unreachable!("map entries have two fields")
    };

    // Reversed fields are legal input to the `List(Struct) -> Map` cast.
    let (keys, values) = if first.name() == &MAP_KEY_NAME {
        (first, second)
    } else {
        (second, first)
    };
    debug_assert_eq!(keys.name(), &MAP_KEY_NAME);
    debug_assert_eq!(values.name(), &MAP_VALUE_NAME);

    (keys, values)
}

/// Return the key and value fields of map entries, matched by name.
///
/// Arrow and Parquet match fields positionally, so their importers must not use this helper.
#[doc(hidden)]
pub fn try_unpack_map_entries(entries: &Series) -> PolarsResult<(Series, Series)> {
    ensure_map_entries_dtype(entries.dtype())?;
    Ok(unpack_map_entries(entries))
}

/// Pack equally sized flat key and value fields into map entries.
///
/// The result does not retain Map row boundaries. Rebuilding a Map column must reattach it to
/// its original [`ListChunked`] storage.
pub(crate) fn pack_map_entries(keys: &Series, values: &Series) -> Series {
    // `StructChunked::from_series` broadcasts unit-length fields.
    assert_eq!(
        keys.len(),
        values.len(),
        "map keys and values must have equal lengths"
    );

    StructChunked::from_series(
        MAP_ENTRIES_NAME.clone(),
        keys.len(),
        [
            &keys.clone().with_name(MAP_KEY_NAME.clone()),
            &values.clone().with_name(MAP_VALUE_NAME.clone()),
        ]
        .into_iter(),
    )
    .expect("map entry children are equal-length and distinctly named")
    .into_series()
}

/// Rebuild Map storage from flat fields, preserving all nested layout.
///
/// This keeps entry validity, list offsets, and list validity.
fn repack_map_storage(storage: &ListChunked, keys: &Series, values: &Series) -> ListChunked {
    let entries = storage.get_inner();
    assert_eq!(
        keys.len(),
        entries.len(),
        "map keys must have one element per entry"
    );
    assert_eq!(
        values.len(),
        entries.len(),
        "map values must have one element per entry"
    );

    let packed = pack_map_entries(keys, values);
    let mut packed = packed.struct_().unwrap().clone();
    packed.zip_outer_validity(entries.struct_().expect("map entries are a struct"));

    storage.with_inner_values(&packed.into_series())
}

/// Transform the flat entry fields and rebuild the original Map storage.
///
/// Preserves entry validity, list offsets, and list validity. The transform must preserve the
/// total number of entries; the output dtype is derived from its returned fields.
pub(crate) fn try_apply_map_entries(
    storage: &ListChunked,
    f: impl FnOnce(&Series, &Series) -> PolarsResult<(Series, Series)>,
) -> PolarsResult<ListChunked> {
    let entries = storage.get_inner();
    let (key, value) = try_unpack_map_entries(&entries)?;

    let entries_len = entries.len();
    let (key, value) = f(&key, &value)?;
    polars_ensure!(
        key.len() == entries_len && value.len() == entries_len,
        ShapeMismatch: "Map entry transform changed the entry count from {entries_len} to ({}, {})",
        key.len(), value.len(),
    );

    Ok(repack_map_storage(storage, &key, &value))
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
