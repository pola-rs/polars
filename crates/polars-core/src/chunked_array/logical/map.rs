use std::borrow::Cow;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_compute::filter::filter_with_bitmap;
use polars_compute::gather::take_unchecked;
use polars_compute::rebuild_list::rebuild_list_shallow;

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
///    categorical codes in range for every non-null slot, no `Object`.
/// 3. Entries and keys are non-null within the offset window of every non-null row of every
///    chunk.
///
/// Map values are nullable. Everything a live row does not own -- entries under null rows or
/// outside the offsets -- is unconstrained, as for any [`ListChunked`]. Flat accessors mask
/// those entries and Arrow export drops them; repairs never clear entry or key validity,
/// which would expose arbitrary payloads.
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
    /// Rejects null entries or keys that a live row owns. Duplicate keys keep their first
    /// position and last value.
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

    /// The raw `List(Struct {key, value})` storage, including entries that no live row owns.
    ///
    /// Entries and keys are non-null within the offset windows of live rows. Null rows and
    /// slicing can hide arbitrary entries, as they can for any [`ListChunked`];
    /// [`Self::live_storage`] empties the null rows.
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

    /// Keys of all entries in live rows, flattened in row order.
    pub fn keys(&self) -> Series {
        self.entry_field(&MAP_KEY_NAME)
    }

    /// Values of all entries in live rows, flattened in row order.
    pub fn values(&self) -> Series {
        self.entry_field(&MAP_VALUE_NAME)
    }

    /// Flatten one entry field over live rows without filtering the other field.
    fn entry_field(&self, name: &PlSmallStr) -> Series {
        let storage = self.storage.list().unwrap();
        let DataType::Struct(fields) = storage.inner_dtype() else {
            unreachable!("map entries are a struct")
        };
        // Reversed fields are legal input to the `List(Struct) -> Map` cast.
        let i = fields
            .iter()
            .position(|field| field.name() == name)
            .expect("map entries have canonical key and value fields");

        let chunks = storage
            .downcast_iter()
            .map(|arr| {
                let entries = windowed_entries_array(arr);
                let entries = entries
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .expect("map entries are a struct");
                let field = entries.values()[i].clone();
                match live_entry_mask(arr) {
                    Some(mask) => filter_with_bitmap(field.as_ref(), &mask),
                    None => field,
                }
            })
            .collect();

        // SAFETY: the chunks are one entry field, filtered to the entries of live rows.
        unsafe { Series::from_chunks_and_dtype_unchecked(name.clone(), chunks, fields[i].dtype()) }
    }

    /// Replace live entry values.
    ///
    /// Requires one value per live entry, in [`Self::values`] order, and a valid Map value
    /// dtype. This also applies to list-valued entries.
    ///
    /// Preserves row count, validity, live keys and entry order. Drops entries that no live
    /// row owns and may rebase offsets.
    pub fn with_values(&self, values: &Series) -> PolarsResult<Self> {
        let dtype = DataType::Map(
            Box::new(self.key_dtype().clone()),
            Box::new(values.dtype().clone()),
        );
        dtype.ensure_valid_map_dtype()?;

        let storage = try_apply_map_entries(&self.live_storage(), |keys, _| {
            polars_ensure!(
                values.len() == keys.len(),
                ShapeMismatch:
                "Map values must have one element per entry: expected {}, got {}",
                keys.len(),
                values.len(),
            );
            Ok((keys.clone(), values.clone()))
        })?
        .into_series();

        // SAFETY: live keys and row assignments are preserved; replacements have a valid
        // Map value dtype. Only entries no live row owns are dropped.
        Ok(unsafe { Self::from_storage_unchecked(dtype, storage) })
    }

    /// Propagate nulls through the storage, exactly as for a [`ListChunked`].
    ///
    /// Entries under null rows may end up null, which the contract allows.
    pub(crate) fn propagate_nulls(&self) -> Option<Self> {
        let storage = self.storage.propagate_nulls()?;

        // SAFETY: only child validity changes; rows, live entries and keys remain.
        Some(unsafe { self.with_storage_unchecked(storage) })
    }

    /// All entries in live rows, flattened in row order.
    ///
    /// Excludes entries retained by null rows or sliced away.
    pub fn entries(&self) -> Series {
        windowed_entries(&self.live_storage())
    }

    /// Storage with empty windows for null rows.
    ///
    /// Preserves row count, validity and live entries. Borrows if no compaction is needed.
    /// Mutating the returned [`Cow`] affects only its owned copy.
    pub fn live_storage(&self) -> Cow<'_, ListChunked> {
        let storage = self.storage.list().unwrap();
        match compact_null_map_rows(storage) {
            Some(compacted) => Cow::Owned(compacted),
            None => Cow::Borrowed(storage),
        }
    }

    /// Set row validity, emptying the rows that turn from null to valid.
    ///
    /// A null row's entries are unconstrained, so revealing them as they stand could expose a
    /// null entry or key. Rows that stay null keep their entries where they were.
    pub(crate) fn with_row_validity(&self, validity: Option<Bitmap>) -> Self {
        let revives_rows = match self.storage.rechunk_validity() {
            None => false,
            Some(old) => match &validity {
                None => old.unset_bits() > 0,
                Some(new) => arrow::bitmap::and_not(new, &old).set_bits() > 0,
            },
        };
        let storage = if revives_rows {
            Cow::Owned(self.live_storage().into_owned().into_series())
        } else {
            Cow::Borrowed(&self.storage)
        };
        // SAFETY: only row validity changes, and no revived row owns an entry.
        unsafe { self.with_storage_unchecked(storage.with_validity(validity)) }
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
            DataType::List(_) => self.live_storage().cast_with_options(dtype, options),
            _ => polars_bail!(InvalidOperation: "cannot cast `{}` to `{dtype}`", self.dtype),
        }
    }

    /// Cast the live entry children, preserving row count and outer validity.
    ///
    /// Drops entries that no live row owns and may rebase offsets. Key casts may merge
    /// duplicate keys.
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
        let storage = try_apply_map_entries(&self.live_storage(), |key, value| {
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
            // SAFETY: keys and row assignments are preserved; cast values remain valid.
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

/// Mask the windowed entries by row validity; `None` if no null row spans an entry.
///
/// Scans validity runs once, allocating only once a null row is found to span entries.
fn live_entry_mask(arr: &LargeListArray) -> Option<Bitmap> {
    let validity = arr.validity().filter(|v| v.unset_bits() > 0)?;
    let offsets = arr.offsets();
    let first = *offsets.first() as usize;
    let n_entries = offsets.range() as usize;

    let mut mask: Option<BitmapBuilder> = None;
    let mut runs = validity.iter();
    let mut row = 0;
    while runs.num_remaining() > 0 {
        row += runs.take_leading_ones();
        let end = row + runs.take_leading_zeros();
        let (start, stop) = (offsets[row] as usize, offsets[end] as usize);
        if stop > start {
            // Fill the live run before this null-row window in bulk.
            let mask = mask.get_or_insert_with(|| BitmapBuilder::with_capacity(n_entries));
            mask.extend_constant(start - first - mask.len(), true);
            mask.extend_constant(stop - start, false);
        }
        row = end;
    }

    let mut mask = mask?;
    mask.extend_constant(n_entries - mask.len(), true);
    Some(mask.freeze())
}

/// Filter out entries under null rows and rebuild offsets; `None` if unchanged.
pub(crate) fn compact_null_rows_chunk(arr: &LargeListArray) -> Option<LargeListArray> {
    let mask = live_entry_mask(arr)?;
    let entries = windowed_entries_array(arr);
    let entries = filter_with_bitmap(entries.as_ref(), &mask);

    let validity = arr.validity().expect("a masked chunk has null rows");
    let live_lengths = validity
        .iter()
        .zip(arr.offsets().lengths())
        .map(|(valid, len)| if valid { len } else { 0 });
    let offsets = Offsets::try_from_lengths(live_lengths)
        .expect("live lengths sum to at most the entry count");

    Some(LargeListArray::new(
        arr.dtype().clone(),
        offsets.into(),
        entries,
        Some(validity.clone()),
    ))
}

/// Drop entries under null rows; `None` if unchanged.
///
/// Rebuilt chunks have trimmed children and zero-based offsets.
pub(crate) fn compact_null_map_rows(storage: &ListChunked) -> Option<ListChunked> {
    if storage.null_count() == 0 {
        return None;
    }

    // Allocate only after the first changed chunk.
    let mut new_chunks: Option<Vec<ArrayRef>> = None;

    for (i, chunk) in storage.downcast_iter().enumerate() {
        match compact_null_rows_chunk(chunk) {
            Some(dropped) => new_chunks
                .get_or_insert_with(|| storage.chunks()[..i].to_vec())
                .push(dropped.boxed()),
            None => {
                if let Some(new_chunks) = new_chunks.as_mut() {
                    new_chunks.push(chunk.clone().boxed());
                }
            },
        }
    }

    // SAFETY: only entries under null rows are removed; the dtype is unchanged.
    new_chunks.map(|chunks| unsafe {
        ListChunked::from_chunks_and_dtype_unchecked(
            storage.name().clone(),
            chunks,
            storage.dtype().clone(),
        )
    })
}

/// Flatten entries within each chunk's offsets.
/// Unlike [`ListChunked::get_inner`], excludes sliced-away entries.
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

/// Transform the flat entry fields and rebuild the original Map storage.
///
/// Also accepts `List(Struct {key, value})` that is not Map storage yet, whose entries and
/// keys may be null. Preserves entry and list validity and row lengths. Only the outer rows
/// are rebuilt: entries nested inside the new fields keep their own offsets. The transform
/// must preserve the windowed entry count; its returned fields determine the output dtype.
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

    let mut packed = pack_map_entries(&key, &value).struct_().unwrap().clone();
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
            let dtype = LargeListArray::default_datatype(values.dtype().clone());
            rebuild_list_shallow(arr, dtype, values).boxed()
        })
        .collect();

    // SAFETY: the list dtype is derived from the packed entries.
    Ok(unsafe {
        ListChunked::from_chunks_and_dtype_unchecked(
            storage.name().clone(),
            chunks,
            DataType::List(Box::new(entries_dtype)),
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

/// Reject null entries or keys under live rows.
///
/// Entries that no live row owns are unconstrained, so only the offset window of each run
/// of valid rows is checked. Callers must validate the dtype and child payloads separately.
pub(crate) fn ensure_live_entries_non_null(storage: &ListChunked) -> PolarsResult<()> {
    for arr in storage.downcast_iter() {
        let entries = arr.values();
        let entries = entries
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("map entries are a struct");
        let [key_arr, _] = entries.values() else {
            unreachable!("map entries must have two arrays")
        };

        // Nothing can be null anywhere, so the windows need not be walked at all.
        let entry_nulls = entries.validity().filter(|v| v.unset_bits() > 0);
        let key_nulls = key_arr.validity().filter(|v| v.unset_bits() > 0);
        if entry_nulls.is_none() && key_nulls.is_none() {
            continue;
        }

        let offsets = arr.offsets();
        let Some(validity) = arr.validity().filter(|v| v.unset_bits() > 0) else {
            ensure_window_non_null(
                entry_nulls,
                key_nulls,
                *offsets.first() as usize,
                offsets.range() as usize,
            )?;
            continue;
        };

        let mut runs = validity.iter();
        let mut row = 0;
        while runs.num_remaining() > 0 {
            let end = row + runs.take_leading_ones();
            let start = offsets[row] as usize;
            ensure_window_non_null(entry_nulls, key_nulls, start, offsets[end] as usize - start)?;
            row = end + runs.take_leading_zeros();
        }
    }

    Ok(())
}

fn ensure_window_non_null(
    entries: Option<&Bitmap>,
    keys: Option<&Bitmap>,
    start: usize,
    len: usize,
) -> PolarsResult<()> {
    polars_ensure!(
        entries.is_none_or(|v| v.null_count_range(start, len) == 0),
        InvalidOperation: "Map entries cannot be null"
    );
    polars_ensure!(
        keys.is_none_or(|v| v.null_count_range(start, len) == 0),
        InvalidOperation: "Map keys cannot be null"
    );
    Ok(())
}

/// Deduplicate keys within each row, keeping each key's first position and last value.
///
/// Rejects null entries or keys under live rows; entries that no live row owns are left
/// alone, except that the deduplicating gather drops them when it runs. Callers must
/// validate the dtype and child payloads.
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
    ensure_live_entries_non_null(list_ca)?;

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

/// Build take indices that deduplicate keys within each row; `None` if unchanged.
///
/// `keys` holds the row encodings of the whole entry child. Only live rows are scanned for
/// duplicates: two hidden keys can encode identically without anyone being able to see it.
/// A gather visits the windows of live rows only, so it also drops every hidden entry.
fn canonical_map_indices(
    arr: &LargeListArray,
    keys: &BinaryArray<i64>,
) -> Option<CanonicalMapIndices> {
    let offsets = arr.offsets();
    let row_validity = arr.validity().filter(|v| v.unset_bits() > 0);

    let mut seen = PlHashSet::new();
    let has_duplicates = (0..arr.len()).any(|row| {
        if row_validity.is_some_and(|v| !v.get_bit(row)) {
            return false;
        }
        let (start, end) = offsets.start_end(row);
        seen.clear();
        (start..end).any(|i| !seen.insert(unsafe { keys.value_unchecked(i) }))
    });
    if !has_duplicates {
        return None;
    }

    let capacity = offsets.range() as usize;
    let mut key_idx = Vec::with_capacity(capacity);
    let mut value_idx = Vec::with_capacity(capacity);
    let mut new_offsets = Vec::with_capacity(offsets.len());
    new_offsets.push(0i64);

    let mut slots = PlHashMap::new();
    for row in 0..arr.len() {
        // A null row owns nothing an observer can reach, so its whole window is dropped.
        if row_validity.is_some_and(|v| !v.get_bit(row)) {
            new_offsets.push(key_idx.len() as i64);
            continue;
        }
        let (start, end) = offsets.start_end(row);

        slots.clear();
        for i in start..end {
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

/// Returns `None` if no live row has duplicate keys.
fn canonicalize_list_chunk(
    arr: &LargeListArray,
    key_dtype: &DataType,
) -> PolarsResult<Option<ArrayRef>> {
    let entries = arr.values();
    let entries = entries.as_any().downcast_ref::<StructArray>().unwrap();
    let [key_arr, _] = entries.values() else {
        unreachable!("map entries must have two arrays")
    };

    // Row encoding matches logical key equality without reading null payloads.
    let keys = unsafe {
        Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![key_arr.clone()], key_dtype)
    };
    let encoded = encode_rows_unordered(&[keys.into_column()])?;
    let encoded = encoded.downcast_iter().next().unwrap();

    let Some(indices) = canonical_map_indices(arr, encoded) else {
        return Ok(None);
    };
    Ok(Some(gather_entries(arr, entries, indices)))
}

/// Check storage invariants directly, before higher-level operations can mask them.
#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;
    use arrow::bitmap::Bitmap;
    use arrow::offset::OffsetsBuffer;

    use super::*;
    use crate::frame::column::Column;
    use crate::scalar::Scalar;

    fn map_dtype(key: DataType, value: DataType) -> DataType {
        DataType::Map(Box::new(key), Box::new(value))
    }

    fn str_keys(keys: &[Option<&str>]) -> Series {
        Series::new(MAP_KEY_NAME.clone(), keys)
    }

    fn i64_values(values: &[Option<i64>]) -> Series {
        Series::new(MAP_VALUE_NAME.clone(), values)
    }

    /// Build raw storage with explicit offsets and validity, bypassing every repair.
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

    /// Whole-child length, including any entries left outside the offsets by slicing.
    fn child_len(storage: &Series) -> usize {
        storage.list().unwrap().get_inner().len()
    }

    /// Check nulls over the entries that live rows own.
    fn assert_no_live_null_entries_or_keys(map: &MapChunked) {
        assert_eq!(map.entries().null_count(), 0, "null entries");
        assert_eq!(map.keys().null_count(), 0, "null keys");
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
    fn hidden_null_enum_key_is_masked_not_fabricated() {
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
        assert_eq!(map.keys().len(), 0);
        assert_no_live_null_entries_or_keys(&map);
        // Formatting resolves all reachable codes.
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
            canonicalize_map_storage(sliced.storage())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn sliced_storage_with_null_entries_outside_the_window_is_accepted() {
        // Slicing away row 0 must put its null key beyond the reach of validation.
        let keys = str_keys(&[None, Some("b"), Some("c")]);
        let values = i64_values(&[Some(1), Some(2), Some(3)]);
        let storage = storage(&pack_map_entries(&keys, &values), &[0, 1, 2, 3], None);
        let dtype = map_dtype(DataType::String, DataType::Int64);
        assert!(MapChunked::try_from_storage(dtype.clone(), storage.clone()).is_err());

        let map = MapChunked::try_from_storage(dtype, storage.slice(1, 2)).unwrap();
        assert_eq!(map.len(), 2);
        assert_no_live_null_entries_or_keys(&map);
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
    fn valid_entries_under_a_null_row_are_kept_but_hidden() {
        let keys = str_keys(&[Some("a"), Some("b")]);
        let values = i64_values(&[Some(1), Some(2)]);
        let storage = storage(
            &pack_map_entries(&keys, &values),
            &[0, 1, 2],
            Some(&[false, true]),
        );
        let dtype = map_dtype(DataType::String, DataType::Int64);
        let map = MapChunked::try_from_storage(dtype, storage).unwrap();

        // The storage keeps `a` where it was; only the accessors hide it.
        assert_eq!(list_offsets(map.storage()), [0, 1, 2]);
        assert_eq!(map.storage().null_count(), 1);
        assert!(matches!(map.get_any_value(0).unwrap(), AnyValue::Null));
        assert_eq!(map.entries().len(), 1);
        assert_eq!(str_values(&map.keys()), [Some("b".to_owned())]);
        let live = map.live_storage().into_owned().into_series();
        assert_eq!(list_offsets(&live), [0, 0, 1]);
    }

    #[test]
    fn with_validity_hides_the_entries_of_nulled_rows() {
        let map = three_row_map().into_series();
        let nulled = map.with_validity(Some(Bitmap::from([false, true, false])));
        let nulled = nulled.map().unwrap();

        // Nulling rows in place leaves the offsets alone, as it does for a list.
        assert_eq!(nulled.storage().null_count(), 2);
        assert_eq!(list_offsets(nulled.storage()), [0, 2, 3, 5]);
        assert_no_live_null_entries_or_keys(nulled);
        assert_eq!(str_values(&nulled.keys()), [Some("c".to_owned())]);

        // A physical round trip must preserve storage validity.
        let physical = nulled.clone().into_series().to_physical_repr().into_owned();
        let back = unsafe { physical.from_physical_unchecked(nulled.dtype()) }.unwrap();
        let back = back.map().unwrap();
        assert_no_live_null_entries_or_keys(back);
        assert_eq!(back.storage().null_count(), 2);
        assert_eq!(str_values(&back.keys()), [Some("c".to_owned())]);
    }

    #[test]
    fn with_values_counts_only_live_entries() {
        let map = three_row_map().into_series();
        let nulled = map.with_validity(Some(Bitmap::from([false, true, false])));
        let nulled = nulled.map().unwrap();

        // Nulling rows 0 and 2 hid their entries; only row 1's `c` is reachable.
        assert_eq!(child_len(nulled.storage()), 5);
        assert_eq!(str_values(&nulled.keys()), [Some("c".to_owned())]);

        let out = nulled.with_values(&i64_values(&[Some(30)])).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.storage().null_count(), 2);
        assert_eq!(list_offsets(out.storage()), [0, 0, 1, 1]);
        assert_eq!(child_len(out.storage()), 1);
        assert_eq!(str_values(&out.keys()), [Some("c".to_owned())]);
        assert_eq!(
            out.values().i64().unwrap().iter().collect::<Vec<_>>(),
            [Some(30)]
        );

        // Reject the physical entry count.
        let err = nulled
            .with_values(&i64_values(&[Some(1), Some(2), Some(3), Some(4), Some(5)]))
            .err()
            .unwrap();
        assert!(err.to_string().contains("expected 1, got 5"), "{err}");
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
    fn from_physical_unchecked_accepts_entries_nulled_under_a_null_row() {
        // Null propagation nulls the entry and key of a null row; both stay hidden.
        let keys = str_keys(&[None, Some("b")]);
        let values = i64_values(&[None, Some(2)]);
        let entries =
            pack_map_entries(&keys, &values).with_validity(Some(Bitmap::from([false, true])));
        let dtype = map_dtype(DataType::String, DataType::Int64);

        let nulled = storage(&entries, &[0, 1, 2], Some(&[false, true]));
        let map = unsafe { nulled.from_physical_unchecked(&dtype) }.unwrap();
        let map = map.map().unwrap();
        assert_no_live_null_entries_or_keys(map);
        assert_eq!(str_values(&map.keys()), [Some("b".to_owned())]);

        // The same null in a live row must be rejected.
        let corrupt = storage(&entries, &[0, 1, 2], None);
        assert!(unsafe { corrupt.from_physical_unchecked(&dtype) }.is_err());
    }

    /// Row 0 null over a nulled entry and key, row 1 `{b: 2}`.
    fn map_with_nulled_entries_under_a_null_row() -> MapChunked {
        let keys = str_keys(&[None, Some("b")]);
        let values = i64_values(&[None, Some(2)]);
        let entries =
            pack_map_entries(&keys, &values).with_validity(Some(Bitmap::from([false, true])));
        let storage = storage(&entries, &[0, 1, 2], Some(&[false, true]));
        let dtype = map_dtype(DataType::String, DataType::Int64);
        unsafe { storage.from_physical_unchecked(&dtype) }
            .unwrap()
            .map()
            .unwrap()
            .clone()
    }

    #[test]
    fn with_validity_empties_a_revived_row() {
        let map = map_with_nulled_entries_under_a_null_row();
        let revived = map.into_series().with_validity(None);
        let revived = revived.map().unwrap();

        assert_eq!(revived.storage().null_count(), 0);
        assert_no_live_null_entries_or_keys(revived);
        assert_eq!(list_offsets(revived.storage()), [0, 0, 1]);
        assert_eq!(str_values(&revived.keys()), [Some("b".to_owned())]);
        // The revived row reads as an empty map, not as a null.
        let AnyValue::Map(row) = revived.get_any_value(0).unwrap() else {
            panic!("revived row is not a map");
        };
        assert_eq!(row.len(), 0);

        // Arrow export checks the non-nullable MAP key field.
        let exported = revived
            .clone()
            .into_series()
            .rechunk()
            .to_arrow(0, CompatLevel::newest());
        assert_eq!(exported.len(), 2);
    }

    #[test]
    fn with_validity_that_only_adds_nulls_keeps_the_entries() {
        let map = map_with_nulled_entries_under_a_null_row();
        let offsets = list_offsets(map.storage());
        let nulled = map
            .into_series()
            .with_validity(Some(Bitmap::from([false, false])));
        let nulled = nulled.map().unwrap();

        assert_eq!(nulled.storage().null_count(), 2);
        assert_eq!(list_offsets(nulled.storage()), offsets);
        assert_eq!(child_len(nulled.storage()), 2);
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
    fn compact_null_map_rows_is_noop_on_canonical_storage() {
        let map = three_row_map();
        assert!(compact_null_map_rows(map.storage().list().unwrap()).is_none());

        let dtype = map_dtype(DataType::String, DataType::Int64);
        let nulls = Series::full_null(PlSmallStr::from_static("m"), 3, &dtype);
        assert!(compact_null_map_rows(nulls.map().unwrap().storage().list().unwrap()).is_none());

        let sliced = map.into_series().slice(1, 2);
        assert!(
            compact_null_map_rows(sliced.map().unwrap().storage().list().unwrap()).is_none(),
            "entries outside the offsets are not hidden by null rows"
        );
    }

    #[test]
    fn deposit_pads_with_null_rows() {
        let map = three_row_map().into_series();
        let deposited = map.deposit(&Bitmap::from([true, false, true, false, true]));
        let deposited = deposited.map().unwrap();

        assert_eq!(deposited.len(), 5);
        assert_eq!(deposited.storage().null_count(), 2);
        for row in [1, 3] {
            assert!(matches!(
                deposited.get_any_value(row).unwrap(),
                AnyValue::Null
            ));
        }
        assert_eq!(
            str_values(&deposited.keys()),
            ["a", "b", "c", "d", "e"].map(|k| Some(k.to_owned()))
        );
    }

    #[cfg(feature = "algorithm_group_by")]
    #[test]
    fn agg_first_on_a_scalar_nulls_empty_group_rows() {
        let scalar = Column::new_scalar(
            PlSmallStr::from_static("m"),
            Scalar::new(
                three_row_map().dtype().clone(),
                three_row_map()
                    .into_series()
                    .slice(0, 1)
                    .get(0)
                    .unwrap()
                    .into_static(),
            ),
            2,
        );
        let groups = GroupsType::new_slice(vec![[0, 1], [1, 0], [1, 1]], false, false);

        let agg = unsafe { scalar.agg_first(&groups) };
        let agg = agg.as_materialized_series().map().unwrap();
        assert_eq!(agg.storage().null_count(), 1);
        assert!(matches!(agg.get_any_value(1).unwrap(), AnyValue::Null));
        assert_eq!(agg.keys().len(), 4);
    }

    #[cfg(all(feature = "algorithm_group_by", feature = "dtype-struct"))]
    #[test]
    fn agg_first_on_a_struct_scalar_propagates_the_empty_group_nulls() {
        let inner = Series::new(PlSmallStr::from_static("f"), [1i64]);
        let value =
            StructChunked::from_series(PlSmallStr::from_static("s"), 1, [&inner].into_iter())
                .unwrap()
                .into_series();
        let scalar = Column::new_scalar(
            PlSmallStr::from_static("s"),
            Scalar::new(value.dtype().clone(), value.get(0).unwrap().into_static()),
            2,
        );
        let groups = GroupsType::new_slice(vec![[0, 1], [1, 0]], false, false);

        let agg = unsafe { scalar.agg_first(&groups) };
        let agg = agg.as_materialized_series();
        assert_eq!(agg.null_count(), 1);
        // The outer null must reach the field, as it would for a non-scalar column.
        assert_eq!(agg.struct_().unwrap().fields_as_series()[0].null_count(), 1);
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
