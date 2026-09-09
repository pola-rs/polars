use arrow::offset::OffsetsBuffer;
use polars_compute::gather::take_unchecked;

use crate::chunked_array::align_inner_chunks;
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

    /// Raw storage; the entries child may extend beyond the list offsets after slicing.
    pub fn storage(&self) -> &Series {
        &self.storage
    }

    /// Rebuild this Map around storage known to preserve its entries.
    ///
    /// # Safety
    /// `storage` must uphold the Map invariants: non-null entries, non-null keys, and
    /// keys unique within each row. Null rows are exempt, as their entries are
    /// unreachable.
    ///
    /// Adding, removing, or reordering whole rows satisfies this requirement. Operations
    /// that modify entries must also preserve key uniqueness.
    ///
    /// # Panics
    /// If `storage` does not have the dtype of [`Self::storage`].
    pub(crate) unsafe fn with_storage_unchecked(&self, storage: Series) -> Self {
        assert_eq!(
            storage.dtype(),
            self.storage.dtype(),
            "operation on Map storage changed its dtype",
        );
        Self {
            dtype: self.dtype.clone(),
            storage,
        }
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

    /// Non-null keys within each chunk's list offsets, including entries retained by null rows.
    pub fn keys(&self) -> Series {
        unpack_map_entries(&self.entries()).0
    }

    /// Values within each chunk's list offsets, including entries retained by null rows.
    pub fn values(&self) -> Series {
        unpack_map_entries(&self.entries()).1
    }

    /// Replace the value child, keeping the keys and the entry layout.
    ///
    /// Requires the same length as [`Self::values`] and a valid Map value dtype.
    pub fn with_values(&self, values: &Series) -> PolarsResult<Self> {
        let dtype = DataType::Map(
            Box::new(self.key_dtype().clone()),
            Box::new(values.dtype().clone()),
        );
        dtype.ensure_valid_map_dtype()?;

        let keys = self.keys();
        polars_ensure!(
            values.len() == keys.len(),
            ShapeMismatch:
            "Map values must have one element per entry: expected {}, got {}",
            keys.len(),
            values.len(),
        );
        let storage = repack_map_storage(self.storage.list().unwrap(), &keys, values).into_series();

        // SAFETY: the keys and the entry layout are untouched, and the dtype is checked.
        Ok(unsafe { Self::from_storage_unchecked(dtype, storage) })
    }

    /// Propagate nulls within entry children without changing row or entry validity.
    pub(crate) fn propagate_nulls(&self) -> Option<Self> {
        let (keys, values) = unpack_map_entries(&self.entries());

        let new_keys = keys.propagate_nulls();
        let new_values = values.propagate_nulls();
        if new_keys.is_none() && new_values.is_none() {
            return None;
        }
        let keys = new_keys.unwrap_or(keys);
        let values = new_values.unwrap_or(values);
        let storage =
            repack_map_storage(self.storage.list().unwrap(), &keys, &values).into_series();

        // SAFETY: only values below nested nulls change; layout and key semantics remain.
        Some(unsafe { self.with_storage_unchecked(storage) })
    }

    /// Flatten entries within each chunk's list offsets, excluding sliced-away rows.
    fn entries(&self) -> Series {
        windowed_entries(self.storage.list().unwrap())
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
            // Decimal rescaling can merge distinct keys even when schemas match.
            Ok(Self::try_from_storage(dtype, storage)?.into_series())
        } else {
            // SAFETY: keys and layout are unchanged; cast values remain valid throughout.
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

/// Slice the entries child to its list offsets in O(1), without recursion.
fn windowed_entries_array(arr: &LargeListArray) -> ArrayRef {
    let offsets = arr.offsets();
    let first = *offsets.first() as usize;
    let len = offsets.range() as usize;
    let values = arr.values();
    if first == 0 && len == values.len() {
        values.clone()
    } else {
        values.sliced(first, len)
    }
}

/// Flatten entries within each chunk's list offsets. Unlike [`ListChunked::get_inner`],
/// this excludes sliced-away entries.
fn windowed_entries(storage: &ListChunked) -> Series {
    let chunks = storage
        .downcast_iter()
        .map(windowed_entries_array)
        .collect();
    // SAFETY: chunks are slices of the storage's entry children.
    unsafe {
        Series::from_chunks_and_dtype_unchecked(
            storage.name().clone(),
            chunks,
            storage.inner_dtype(),
        )
    }
}

/// Rebase offsets to zero for a sliced child.
fn rebased_offsets(offsets: &OffsetsBuffer<i64>) -> OffsetsBuffer<i64> {
    let first = *offsets.first();
    if first == 0 {
        return offsets.clone();
    }
    let rebased: Vec<i64> = offsets.as_slice().iter().map(|o| o - first).collect();
    // SAFETY: rebasing preserves monotonicity and makes the first offset zero.
    unsafe { OffsetsBuffer::new_unchecked(rebased.into()) }
}

/// Rebuild Map storage from flat fields with one element per windowed entry.
///
/// Preserves entry and list validity, rebasing offsets onto the new child.
/// Rebasing allocates only for sliced inputs.
fn repack_map_storage(storage: &ListChunked, keys: &Series, values: &Series) -> ListChunked {
    let entries = windowed_entries(storage);
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
    let packed = packed.into_series();

    // Align chunks by offset-window length, not full child length.
    let window_lens = storage
        .downcast_iter()
        .map(|arr| arr.offsets().range() as usize);
    let packed = align_inner_chunks(window_lens, &packed);
    let entries_dtype = packed.dtype().clone();

    let chunks = storage
        .downcast_iter()
        .zip(packed.into_chunks())
        .map(|(arr, values)| {
            debug_assert_eq!(arr.offsets().range() as usize, values.len());
            LargeListArray::new(
                LargeListArray::default_datatype(values.dtype().clone()),
                rebased_offsets(arr.offsets()),
                values,
                arr.validity().cloned(),
            )
            .boxed()
        })
        .collect();

    // SAFETY: the list dtype is derived from the packed entries.
    unsafe {
        ListChunked::from_chunks_and_dtype_unchecked(
            storage.name().clone(),
            chunks,
            DataType::List(Box::new(entries_dtype)),
        )
    }
}

/// Transform the flat entry fields and rebuild the original Map storage.
///
/// Preserves validity and row lengths. The transform must preserve the windowed entry count;
/// its returned fields determine the output dtype.
pub(crate) fn try_apply_map_entries(
    storage: &ListChunked,
    f: impl FnOnce(&Series, &Series) -> PolarsResult<(Series, Series)>,
) -> PolarsResult<ListChunked> {
    let entries = windowed_entries(storage);
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

/// Check storage invariants directly, before higher-level operations can repair them.
#[cfg(test)]
mod test {
    use arrow::bitmap::Bitmap;
    use arrow::offset::OffsetsBuffer;

    use super::*;

    fn map_dtype(key: DataType, value: DataType) -> DataType {
        DataType::Map(Box::new(key), Box::new(value))
    }

    fn str_keys(keys: &[Option<&str>]) -> Series {
        Series::new(MAP_KEY_NAME.clone(), keys)
    }

    fn i64_values(values: &[Option<i64>]) -> Series {
        Series::new(MAP_VALUE_NAME.clone(), values)
    }

    /// Build storage with explicit offsets and validity to test hidden entries.
    fn storage(entries: &Series, offsets: &[i64], row_validity: Option<&[bool]>) -> Series {
        let entries = entries.rechunk();
        let values = entries.chunks()[0].clone();
        let arr = LargeListArray::new(
            LargeListArray::default_datatype(values.dtype().clone()),
            unsafe { OffsetsBuffer::new_unchecked(offsets.to_vec().into()) },
            values,
            row_validity.map(Bitmap::from),
        );
        unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::from_static("m"),
                vec![arr.boxed()],
                &DataType::List(Box::new(entries.dtype().clone())),
            )
        }
    }

    fn list_offsets(storage: &Series) -> Vec<i64> {
        let ca = storage.list().unwrap();
        assert_eq!(ca.chunks().len(), 1);
        ca.downcast_iter()
            .next()
            .unwrap()
            .offsets()
            .as_slice()
            .to_vec()
    }

    /// Whole-child length, including entries outside the offsets.
    fn child_len(storage: &Series) -> usize {
        storage.list().unwrap().get_inner().len()
    }

    /// Rows `{a: 1, b: 2}`, `{c: 3}`, `{d: 4, e: 5}`.
    fn three_row_map() -> MapChunked {
        let keys = str_keys(&[Some("a"), Some("b"), Some("c"), Some("d"), Some("e")]);
        let values = i64_values(&[Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let storage = storage(&pack_map_entries(&keys, &values), &[0, 2, 3, 5], None);
        MapChunked::try_from_storage(map_dtype(DataType::String, DataType::Int64), storage).unwrap()
    }

    fn str_values(s: &Series) -> Vec<Option<String>> {
        s.str()
            .unwrap()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    }

    #[test]
    fn sliced_map_windows_flat_access() {
        let sliced = three_row_map().into_series().slice(1, 2);
        let sliced = sliced.map().unwrap();

        // Storage retains all entries; flat access respects the slice.
        assert_eq!(child_len(sliced.storage()), 5);
        assert_eq!(
            str_values(&sliced.keys()),
            [
                Some("c".to_owned()),
                Some("d".to_owned()),
                Some("e".to_owned())
            ]
        );
        assert_eq!(sliced.values().len(), 3);
        // Valid entries outside the offsets need no repair.
        assert!(
            canonicalize_map_storage(sliced.storage())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn with_values_round_trip_on_a_sliced_map() {
        let sliced = three_row_map().into_series().slice(1, 2);
        let sliced = sliced.map().unwrap();
        let values = sliced.values().cast(&DataType::Float64).unwrap();
        let out = sliced.with_values(&values).unwrap();

        let dtype = map_dtype(DataType::String, DataType::Float64);
        assert_eq!(out.dtype(), &dtype);
        assert_eq!(out.len(), 2);
        assert_eq!(list_offsets(out.storage()), [0, 1, 3]);
        assert_eq!(child_len(out.storage()), 3);

        // Casting must use the same offset window.
        let expected = three_row_map()
            .into_series()
            .slice(1, 2)
            .cast(&dtype)
            .unwrap();
        assert!(
            out.storage()
                .equals_missing(expected.map().unwrap().storage())
        );
    }
}
