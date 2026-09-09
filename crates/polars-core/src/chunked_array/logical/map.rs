use arrow::bitmap::Bitmap;
use arrow::offset::OffsetsBuffer;
use polars_compute::gather::take_unchecked;

use crate::chunked_array::align_inner_chunks;
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::iterator::PolarsIterator;
use crate::chunked_array::ops::row_encode::encode_rows_unordered;
use crate::prelude::*;

/// A `Map` backed by a `List(Struct {key, value})` [`Series`].
///
/// # Storage safety contract
///
/// Every `MapChunked` must satisfy:
///
/// 1. The storage dtype is the [`DataType::map_storage_dtype`] of a Map dtype that passes
///    [`DataType::ensure_valid_map_dtype`].
/// 2. Child arrays are valid over their entire extent, including outside the list offsets:
///    categorical codes in range, Decimals within precision, no `Object`.
/// 3. Entries and keys have no nulls anywhere in the child arrays.
///
/// Map values are nullable. Null rows may retain valid entries. Repairs drop null entries
/// or keys; clearing their validity would expose arbitrary payloads.
///
/// # Canonical semantics
///
/// Validated ingestion ([`Self::try_from_storage`], `List(Struct) -> Map`, `from_any_values`,
/// Parquet) deduplicates keys within each row. Whole-row operations and value-only replacements
/// preserve uniqueness; key-changing operations re-establish it. Arrow, IPC and FFI imports
/// trust the producer and may contain duplicates. Uniqueness is never a safety requirement;
/// callers that need it must canonicalize first.
///
/// Equality is entry-order-sensitive on the storage.
#[derive(Clone)]
pub struct MapChunked {
    dtype: DataType,
    storage: Series,
}

impl MapChunked {
    /// # Safety
    /// `storage` must satisfy the [`MapChunked`] storage safety contract for `dtype`.
    pub unsafe fn from_storage_unchecked(dtype: DataType, storage: Series) -> Self {
        debug_assert_eq!(dtype.map_storage_dtype().as_ref(), Some(storage.dtype()));
        debug_assert!(
            dtype.ensure_valid_map_dtype().is_ok(),
            "invalid Map dtype: {dtype}"
        );
        Self { dtype, storage }
    }

    /// Validate map storage and canonicalize duplicate keys.
    ///
    /// Rejects null entries or keys in live rows; drops them elsewhere. Duplicate keys keep
    /// their first position and last value.
    pub fn try_from_storage(dtype: DataType, storage: Series) -> PolarsResult<Self> {
        dtype.ensure_valid_map_dtype()?;

        let storage_dtype = dtype.map_storage_dtype().unwrap();
        polars_ensure!(
            storage.dtype() == &storage_dtype,
            InvalidOperation: "expected `{storage_dtype}` storage for `{dtype}`, got `{}`",
            storage.dtype()
        );

        let storage =
            canonicalize_map_storage(&storage, CanonicalizeMode::Full)?.unwrap_or(storage);
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
    /// `storage` must satisfy the [`MapChunked`] storage safety contract for this Map's dtype.
    ///
    /// Whole-row operations and value-only replacements preserve safety and key uniqueness.
    /// Key changes must restore uniqueness or use [`Self::try_from_storage`].
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

/// Both modes reject live null entries/keys and compact hidden ones. Callers must validate
/// the dtype and child payloads separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalizeMode {
    /// Also deduplicate keys within each row.
    Full,
    /// Skip row encoding and deduplication; preserve any existing uniqueness.
    NullsOnly,
}

/// Remove hidden null entries/keys and reject live ones. [`CanonicalizeMode::Full`] also
/// deduplicates keys, keeping each key's first position and last value.
///
/// Hidden means under a null row or outside the list offsets. Valid entries under null rows
/// are retained. Null payloads are never exposed; dtype and payload validity are not checked.
///
/// Returns `None` if unchanged; use [`Series::canonicalize_maps`] to also reach maps
/// nested inside the keys or values.
pub(crate) fn canonicalize_map_storage(
    storage: &Series,
    mode: CanonicalizeMode,
) -> PolarsResult<Option<Series>> {
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
        match canonicalize_list_chunk(chunk, key_field.dtype(), mode)? {
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

/// Entry and key validity for a chunk containing nulls.
#[derive(Clone, Copy)]
struct EntryNulls<'a> {
    entries: Option<&'a Bitmap>,
    keys: Option<&'a Bitmap>,
}

impl EntryNulls<'_> {
    fn entry_nulls_in(&self, start: usize, len: usize) -> usize {
        self.entries.map_or(0, |v| v.null_count_range(start, len))
    }

    fn key_nulls_in(&self, start: usize, len: usize) -> usize {
        self.keys.map_or(0, |v| v.null_count_range(start, len))
    }

    fn is_null(&self, i: usize) -> bool {
        self.entries.is_some_and(|v| !v.get_bit(i)) || self.keys.is_some_and(|v| !v.get_bit(i))
    }
}

fn has_unset_bits(validity: Option<&Bitmap>) -> bool {
    validity.is_some_and(|v| v.unset_bits() > 0)
}

/// Build take indices for deduplication or null removal; return `None` if unchanged.
///
/// `keys` contains row encodings in [`CanonicalizeMode::Full`]; `nulls` tracks child nulls.
/// Gathering visits only row windows, dropping entries outside them.
fn canonical_map_indices(
    arr: &LargeListArray,
    keys: Option<&BinaryArray<i64>>,
    nulls: Option<EntryNulls<'_>>,
) -> PolarsResult<Option<CanonicalMapIndices>> {
    let offsets = arr.offsets();

    if nulls.is_none() {
        let Some(keys) = keys else {
            return Ok(None);
        };
        let mut seen = PlHashSet::new();
        let has_duplicates = offsets.as_slice().windows(2).any(|range| {
            seen.clear();
            (range[0] as usize..range[1] as usize)
                .any(|i| !seen.insert(unsafe { keys.value_unchecked(i) }))
        });
        if !has_duplicates {
            return Ok(None);
        }
    }

    let row_validity = arr.validity();
    let n_entries = offsets.range() as usize;
    let mut key_idx = Vec::with_capacity(n_entries);
    let mut value_idx = Vec::with_capacity(n_entries);
    let mut new_offsets = Vec::with_capacity(offsets.len());
    new_offsets.push(0i64);

    let mut slots = PlHashMap::new();
    for row in 0..arr.len() {
        let (start, end) = offsets.start_end(row);

        // Reject null entries/keys in live rows; skip them in null rows.
        let entry_nulls = nulls.map_or(0, |n| n.entry_nulls_in(start, end - start));
        let key_nulls = nulls.map_or(0, |n| n.key_nulls_in(start, end - start));
        let dirty_row = nulls.filter(|_| entry_nulls > 0 || key_nulls > 0);
        if dirty_row.is_some() && row_validity.is_none_or(|v| v.get_bit(row)) {
            polars_ensure!(
                entry_nulls == 0,
                InvalidOperation: "Map entries cannot be null"
            );
            polars_bail!(InvalidOperation: "Map keys cannot be null");
        }

        slots.clear();
        for i in start..end {
            if dirty_row.is_some_and(|nulls| nulls.is_null(i)) {
                continue;
            }
            match keys {
                Some(keys) => {
                    let key = unsafe { keys.value_unchecked(i) };
                    if let Some(&slot) = slots.get(key) {
                        value_idx[slot] = i as IdxSize;
                    } else {
                        slots.insert(key, key_idx.len());
                        key_idx.push(i as IdxSize);
                        value_idx.push(i as IdxSize);
                    }
                },
                None => {
                    key_idx.push(i as IdxSize);
                    value_idx.push(i as IdxSize);
                },
            }
        }
        new_offsets.push(key_idx.len() as i64);
    }

    Ok(Some(CanonicalMapIndices {
        first_keys: IdxArr::from_vec(key_idx),
        last_values: IdxArr::from_vec(value_idx),
        offsets: unsafe { OffsetsBuffer::new_unchecked(new_offsets.into()) },
    }))
}

/// Gather entries and rebuild the list chunk.
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
        // Only valid entries are gathered.
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

/// Returns `None` if neither null removal nor the requested deduplication changes `arr`.
fn canonicalize_list_chunk(
    arr: &LargeListArray,
    key_dtype: &DataType,
    mode: CanonicalizeMode,
) -> PolarsResult<Option<ArrayRef>> {
    let entries = arr.values();
    let entries = entries.as_any().downcast_ref::<StructArray>().unwrap();
    let [key_arr, _] = entries.values() else {
        unreachable!("map entries must have two arrays")
    };

    // Check entries and keys over the whole child, which flat storage access exposes.
    // Values may be null.
    let nulls = EntryNulls {
        entries: entries.validity(),
        keys: key_arr.validity(),
    };
    let dirty = has_unset_bits(nulls.entries) || has_unset_bits(nulls.keys);
    if !dirty && mode == CanonicalizeMode::NullsOnly {
        return Ok(None);
    }

    let encoded = match mode {
        CanonicalizeMode::Full => {
            // Row encoding matches logical key equality without reading null payloads.
            let keys = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    PlSmallStr::EMPTY,
                    vec![key_arr.clone()],
                    key_dtype,
                )
            };
            Some(encode_rows_unordered(&[keys.into_column()])?)
        },
        CanonicalizeMode::NullsOnly => None,
    };
    let encoded = encoded
        .as_ref()
        .map(|ca| ca.downcast_iter().next().unwrap());

    let Some(indices) = canonical_map_indices(arr, encoded, dirty.then_some(nulls))? else {
        return Ok(None);
    };
    Ok(Some(gather_entries(arr, entries, indices)))
}

/// Check storage invariants directly, before higher-level operations can repair them.
#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;
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

    /// Check nulls over the whole child, including outside the offsets.
    fn assert_no_null_entries_or_keys(storage: &Series) {
        let entries = storage.list().unwrap().get_inner();
        assert_eq!(entries.null_count(), 0, "null entries");
        let (keys, _) = unpack_map_entries(&entries);
        assert_eq!(keys.null_count(), 0, "null keys");
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

    #[cfg(feature = "dtype-categorical")]
    #[test]
    fn hidden_null_enum_key_is_compacted_not_fabricated() {
        use polars_dtype::categorical::FrozenCategories;

        // With no categories, exposing a null payload would make formatting access
        // an out-of-range code through `cat_to_str_unchecked`.
        let enum_dtype =
            DataType::from_frozen_categories(FrozenCategories::new(std::iter::empty()).unwrap());
        let keys = Series::full_null(MAP_KEY_NAME.clone(), 1, &enum_dtype);
        let values = i64_values(&[Some(1)]);
        let storage = storage(&pack_map_entries(&keys, &values), &[0, 1], Some(&[false]));

        let dtype = map_dtype(enum_dtype, DataType::Int64);
        let map = MapChunked::try_from_storage(dtype, storage).unwrap();
        assert_eq!(list_offsets(map.storage()), [0, 0]);
        assert_eq!(child_len(map.storage()), 0);
        assert_eq!(map.keys().len(), 0);
        assert_no_null_entries_or_keys(map.storage());
        // Formatting resolves all remaining codes.
        let _ = format!("{}", map.keys());
        let _ = format!("{}", map.into_series());
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
            canonicalize_map_storage(sliced.storage(), CanonicalizeMode::Full)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn sliced_storage_with_null_entries_outside_the_window_is_compacted() {
        // Slicing away row 0 must allow its null key to be dropped.
        let keys = str_keys(&[None, Some("b"), Some("c")]);
        let values = i64_values(&[Some(1), Some(2), Some(3)]);
        let storage = storage(&pack_map_entries(&keys, &values), &[0, 1, 2, 3], None);
        let dtype = map_dtype(DataType::String, DataType::Int64);
        assert!(MapChunked::try_from_storage(dtype.clone(), storage.clone()).is_err());

        let map = MapChunked::try_from_storage(dtype, storage.slice(1, 2)).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(list_offsets(map.storage()), [0, 1, 2]);
        assert_eq!(child_len(map.storage()), 2);
        assert_no_null_entries_or_keys(map.storage());
        assert_eq!(
            str_values(&map.keys()),
            [Some("b".to_owned()), Some("c".to_owned())]
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

    #[test]
    fn valid_entries_under_a_null_row_are_retained() {
        let keys = str_keys(&[Some("a"), Some("b")]);
        let values = i64_values(&[Some(1), Some(2)]);
        let storage = storage(
            &pack_map_entries(&keys, &values),
            &[0, 1, 2],
            Some(&[false, true]),
        );
        let dtype = map_dtype(DataType::String, DataType::Int64);
        let map = MapChunked::try_from_storage(dtype, storage).unwrap();

        assert_eq!(list_offsets(map.storage()), [0, 1, 2]);
        assert_eq!(map.storage().null_count(), 1);
        assert!(matches!(map.get_any_value(0).unwrap(), AnyValue::Null));
        // Retained entries are reachable through flat access.
        assert_eq!(
            str_values(&map.keys()),
            [Some("a".to_owned()), Some("b".to_owned())]
        );
    }

    #[test]
    fn with_validity_keeps_retained_entries_valid() {
        let map = three_row_map().into_series();
        let nulled = map.with_validity(Some(Bitmap::from([false, true, false])));
        let nulled = nulled.map().unwrap();

        assert_eq!(nulled.storage().null_count(), 2);
        assert_eq!(list_offsets(nulled.storage()), [0, 2, 3, 5]);
        assert_no_null_entries_or_keys(nulled.storage());
        assert_eq!(nulled.keys().len(), 5);

        // A physical round trip must preserve storage validity.
        let physical = nulled.clone().into_series().to_physical_repr().into_owned();
        let back = unsafe { physical.from_physical_unchecked(nulled.dtype()) }.unwrap();
        let back = back.map().unwrap();
        assert_no_null_entries_or_keys(back.storage());
        assert_eq!(back.storage().null_count(), 2);
    }

    #[test]
    fn nested_map_with_duplicate_inner_keys_is_canonicalized() {
        // Inner rows `{x: 1, x: 2}` and `{y: 3}`, under outer keys `a` and `b`.
        let inner_keys = str_keys(&[Some("x"), Some("x"), Some("y")]);
        let inner_values = i64_values(&[Some(1), Some(2), Some(3)]);
        let inner_storage = storage(
            &pack_map_entries(&inner_keys, &inner_values),
            &[0, 2, 3],
            None,
        );
        let outer_keys = str_keys(&[Some("a"), Some("b")]);
        let outer_storage = storage(
            &pack_map_entries(&outer_keys, &inner_storage),
            &[0, 2],
            None,
        );

        let inner_dtype = map_dtype(DataType::String, DataType::Int64);
        let dtype = map_dtype(DataType::String, inner_dtype.clone());
        assert_eq!(outer_storage.dtype(), &dtype.to_physical());

        let map = outer_storage.try_from_physical(&dtype).unwrap();
        let map = map.map().unwrap();
        assert_eq!(map.dtype(), &dtype);
        let inner = map.values();
        let inner = inner.map().unwrap();
        assert_eq!(inner.dtype(), &inner_dtype);
        assert_eq!(list_offsets(inner.storage()), [0, 1, 2]);
        assert_eq!(
            inner.values().i64().unwrap().iter().collect::<Vec<_>>(),
            [Some(2), Some(3)]
        );
    }

    #[test]
    fn from_physical_unchecked_compacts_nulled_hidden_entries() {
        // Simulate dtype-blind propagation nulling a hidden entry and its key.
        let keys = str_keys(&[None, Some("b")]);
        let values = i64_values(&[None, Some(2)]);
        let entries =
            pack_map_entries(&keys, &values).with_validity(Some(Bitmap::from([false, true])));
        let dtype = map_dtype(DataType::String, DataType::Int64);

        let nulled = storage(&entries, &[0, 1, 2], Some(&[false, true]));
        let map = unsafe { nulled.from_physical_unchecked(&dtype) }.unwrap();
        let map = map.map().unwrap();
        assert_eq!(list_offsets(map.storage()), [0, 0, 1]);
        assert_no_null_entries_or_keys(map.storage());
        assert_eq!(str_values(&map.keys()), [Some("b".to_owned())]);

        // The same null in a live row must be rejected.
        let corrupt = storage(&entries, &[0, 1, 2], None);
        assert!(unsafe { corrupt.from_physical_unchecked(&dtype) }.is_err());
    }

    /// Simulate a live Arrow row with a null entry/key; PyArrow aborts on this input.
    fn malformed_arrow_map(null_key: bool) -> ArrayRef {
        use arrow::array::{MapArray, StructArray, Utf8ViewArray};

        let fields = vec![
            ArrowField::new(PlSmallStr::from_static("k"), ArrowDataType::Utf8View, false),
            ArrowField::new(PlSmallStr::from_static("v"), ArrowDataType::Int64, true),
        ];
        let entries_dtype = ArrowDataType::Struct(fields);
        let keys = Utf8ViewArray::from_slice([Some("a")]);
        let keys = if null_key {
            keys.with_validity(Some(Bitmap::from([false])))
        } else {
            keys
        };
        let entries = StructArray::new(
            entries_dtype.clone(),
            1,
            vec![
                keys.boxed(),
                PrimitiveArray::<i64>::from_vec(vec![1]).boxed(),
            ],
            (!null_key).then(|| Bitmap::from([false])),
        );
        let map_dtype = ArrowDataType::Map(
            Box::new(ArrowField::new(
                PlSmallStr::from_static("entries"),
                entries_dtype,
                false,
            )),
            false,
        );
        MapArray::new(
            map_dtype,
            vec![0i32, 1].try_into().unwrap(),
            entries.boxed(),
            None,
        )
        .boxed()
    }

    #[test]
    fn arrow_import_rejects_live_row_nulls_at_every_depth() {
        use arrow::array::{ListArray, MapArray, StructArray, Utf8ViewArray};

        let nest_in_list = |arr: ArrayRef| -> ArrayRef {
            ListArray::<i64>::new(
                ListArray::<i64>::default_datatype(arr.dtype().clone()),
                vec![0i64, 1].try_into().unwrap(),
                arr,
                None,
            )
            .boxed()
        };
        let nest_in_struct = |arr: ArrayRef| -> ArrayRef {
            let fields = vec![ArrowField::new(
                PlSmallStr::from_static("f"),
                arr.dtype().clone(),
                true,
            )];
            StructArray::new(ArrowDataType::Struct(fields), 1, vec![arr], None).boxed()
        };
        let nest_in_map = |arr: ArrayRef| -> ArrayRef {
            let fields = vec![
                ArrowField::new(PlSmallStr::from_static("k"), ArrowDataType::Utf8View, false),
                ArrowField::new(PlSmallStr::from_static("v"), arr.dtype().clone(), true),
            ];
            let entries_dtype = ArrowDataType::Struct(fields);
            let entries = StructArray::new(
                entries_dtype.clone(),
                1,
                vec![Utf8ViewArray::from_slice([Some("outer")]).boxed(), arr],
                None,
            );
            let map_dtype = ArrowDataType::Map(
                Box::new(ArrowField::new(
                    PlSmallStr::from_static("entries"),
                    entries_dtype,
                    false,
                )),
                false,
            );
            MapArray::new(
                map_dtype,
                vec![0i32, 1].try_into().unwrap(),
                entries.boxed(),
                None,
            )
            .boxed()
        };
        let nests: [(&str, &dyn Fn(ArrayRef) -> ArrayRef); 4] = [
            ("top-level", &|arr| arr),
            ("list", &nest_in_list),
            ("struct", &nest_in_struct),
            ("map", &nest_in_map),
        ];

        for (null_key, message) in [
            (false, "Map entries cannot be null"),
            (true, "Map keys cannot be null"),
        ] {
            for (depth, nest) in &nests {
                let arr = nest(malformed_arrow_map(null_key));
                // Nested cases must not panic in `to_physical_and_dtype`.
                let err = Series::try_from((PlSmallStr::from_static("m"), arr))
                    .err()
                    .unwrap();
                assert!(
                    err.to_string().contains(message),
                    "{depth}: expected `{message}`, got `{err}`"
                );
            }
        }
    }

    #[test]
    fn unsafe_set_inner_dtype_relabels_valid_storage() {
        let map = three_row_map();
        let container = Series::new(PlSmallStr::from_static("c"), &[map.storage().clone()]);
        let mut container = container.list().unwrap().clone();
        // SAFETY: the storage comes from a valid Map.
        unsafe { container.set_inner_dtype(map.dtype().clone()) };

        let inner = container.get_inner();
        let inner = inner.map().unwrap();
        assert_eq!(inner.keys().null_count(), 0);
        assert_eq!(inner.keys().len(), 5);

        // Relabelling also accepts an already-logical inner dtype.
        let mut list = container.clone();
        // SAFETY: the storage is unchanged and valid for the same dtype.
        unsafe { list.set_inner_dtype(map.dtype().clone()) };
        assert_eq!(list.inner_dtype(), map.dtype());
        #[cfg(feature = "dtype-array")]
        {
            let mut array = list
                .cast(&DataType::Array(Box::new(map.dtype().clone()), map.len()))
                .unwrap()
                .array()
                .unwrap()
                .clone();
            // SAFETY: as above.
            unsafe { array.set_inner_dtype(map.dtype().clone()) };
            assert_eq!(array.inner_dtype(), map.dtype());
        }
    }
}
