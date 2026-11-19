use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use polars_arrow::array::{Array, BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::aliases::{InitHashMaps, PlHashMap, PlRandomState};
use polars_utils::pl_str::PlSmallStr;
use rayon::prelude::*;

use super::{Column, DataType, IntoColumn, Series};
use crate::prelude::*;
use crate::runtime::RAYON;
use crate::series::IsSorted;

static NEXT_DICTIONARY_ID: AtomicU64 = AtomicU64::new(0);

/// How many hashers' hashes a dictionary keeps.
const MAX_CACHED_HASHERS: usize = 4;

/// The values of a dictionary that [`DictionaryColumn`]s index into, such as the dictionary page
/// of a parquet column chunk. They can be of any dtype; String and Binary values (the only ones
/// that are produced so far) also have byte fast paths: the codes, the hashes and the lookup table
/// work on their bytes, see [`Self::bytes`].
///
/// The values need not be unique: equal codes mean equal values, but different codes may also.
pub struct SharedDictionary {
    /// Process-unique id; caches in operators are keyed by it.
    id: u64,
    /// The dictionary values, in one chunk and without nulls; materialized columns share its
    /// buffers.
    values: Series,
    /// The dtype of `values`, kept here so that reading a row does not ask `values` for it.
    dtype: DataType,
    /// Whether the values are String, Binary or neither (then `bytes` is `None`).
    kind: ByteKind,
    /// The values of a String (valid UTF-8) or Binary dictionary as bytes, sharing `values`'
    /// buffers.
    bytes: Option<BinaryViewArray>,
    /// The hash of each value under recently used hashers, keyed by [`hasher_fingerprint`].
    hashes: Mutex<Vec<CachedHashes>>,
    /// The first code of each value, built on first use (see [`Self::code_of`]).
    lookup: OnceLock<CodeLookup>,
}

/// What a dictionary's values are as bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ByteKind {
    String,
    Binary,
    Other,
}

/// The hashes of a dictionary's values under one hasher.
struct CachedHashes {
    fingerprint: u128,
    random_state: PlRandomState,
    hashes: Buffer<u64>,
}

/// A table from each distinct value of a dictionary to its first code.
#[derive(Clone)]
struct CodeLookup {
    random_state: PlRandomState,
    codes: hashbrown::HashTable<u32>,
}

impl std::fmt::Debug for SharedDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedDictionary")
            .field("id", &self.id)
            .field("dtype", self.values.dtype())
            .field("len", &self.values.len())
            .finish()
    }
}

/// Identifies a hasher by the hashes it gives two values.
fn hasher_fingerprint(random_state: &PlRandomState) -> u128 {
    let lo = random_state.hash_one(0x243F_6A88_85A3_08D3_u64);
    let hi = random_state.hash_one(0x1319_8A2E_0370_7344_u64);
    ((hi as u128) << 64) | lo as u128
}

impl SharedDictionary {
    /// A String dictionary. The values must not contain nulls.
    pub fn new(values: Utf8ViewArray) -> Arc<Self> {
        Self::with_values(values.to_binview(), true)
    }

    /// A Binary dictionary. The values must not contain nulls.
    pub fn new_binary(values: BinaryViewArray) -> Arc<Self> {
        Self::with_values(values, false)
    }

    /// A dictionary of `values` of any dtype but Null and Object, which must not be null (the
    /// values inside nested ones may be). A String or Binary series gets the byte fast paths.
    pub fn try_new(values: Series) -> PolarsResult<Arc<Self>> {
        polars_ensure!(
            can_be_dictionary_values(values.dtype()),
            ComputeError: "a dictionary cannot hold values of dtype {}", values.dtype()
        );
        polars_ensure!(
            values.null_count() == 0,
            ComputeError: "dictionary values must not be null"
        );
        let values = values.rechunk().with_name(PlSmallStr::EMPTY);
        let bytes = match values.dtype() {
            DataType::String => Some(
                values
                    .str()
                    .unwrap()
                    .as_binary()
                    .downcast_as_array()
                    .clone(),
            ),
            DataType::Binary => Some(values.binary().unwrap().downcast_as_array().clone()),
            _ => None,
        };
        Ok(Self::make(values, bytes, Vec::new(), OnceLock::new()))
    }

    fn with_values(values: BinaryViewArray, is_string: bool) -> Arc<Self> {
        assert!(
            values.validity().is_none_or(|v| v.unset_bits() == 0),
            "dictionary values must not be null"
        );
        let bytes = values.with_validity(None);
        let series = if is_string {
            // SAFETY: a String dictionary holds valid UTF-8.
            let values = unsafe { bytes.to_utf8view_unchecked() };
            StringChunked::with_chunk(PlSmallStr::EMPTY, values).into_series()
        } else {
            BinaryChunked::with_chunk(PlSmallStr::EMPTY, bytes.clone()).into_series()
        };
        Self::make(series, Some(bytes), Vec::new(), OnceLock::new())
    }

    fn make(
        values: Series,
        bytes: Option<BinaryViewArray>,
        hashes: Vec<CachedHashes>,
        lookup: OnceLock<CodeLookup>,
    ) -> Arc<Self> {
        let dtype = values.dtype().clone();
        let kind = match (&bytes, &dtype) {
            (None, _) => ByteKind::Other,
            (Some(_), DataType::String) => ByteKind::String,
            (Some(_), _) => ByteKind::Binary,
        };
        Arc::new(Self {
            id: NEXT_DICTIONARY_ID.fetch_add(1, Ordering::Relaxed),
            values,
            dtype,
            kind,
            bytes,
            hashes: Mutex::new(hashes),
            lookup,
        })
    }

    /// `None` unless the dictionary holds bytes.
    fn lookup(&self) -> Option<&CodeLookup> {
        let bytes = self.bytes.as_ref()?;
        Some(self.lookup.get_or_init(|| {
            let random_state = PlRandomState::default();
            let mut codes = hashbrown::HashTable::with_capacity(self.len());
            for (c, v) in bytes.values_iter().enumerate() {
                let h = random_state.hash_one(v);
                // SAFETY: the codes in the table are in bounds.
                let value = |k: &u32| unsafe { bytes.value_unchecked(*k as usize) };
                if codes.find(h, |k| value(k) == v).is_none() {
                    codes.insert_unique(h, c as u32, |k| random_state.hash_one(value(k)));
                }
            }
            CodeLookup {
                random_state,
                codes,
            }
        }))
    }

    /// The first code of `value`, if this is a String or Binary dictionary that holds it. The
    /// lookup table is built on the first call and kept with the dictionary.
    pub fn code_of(&self, value: &[u8]) -> Option<u32> {
        let bytes = self.bytes.as_ref()?;
        let lookup = self.lookup()?;
        lookup
            .codes
            .find(lookup.random_state.hash_one(value), |k| {
                // SAFETY: the codes in the table are in bounds.
                unsafe { bytes.value_unchecked(*k as usize) == value }
            })
            .copied()
    }

    /// A new dictionary of these values followed by `added`, none of which this one holds. It
    /// shares these values' buffers, and extends the lookup table and the cached hashes instead of
    /// computing them again. Only for a String or Binary dictionary.
    fn extended(&self, added: &[&[u8]]) -> Arc<Self> {
        let old_bytes = self.bytes.as_ref().expect("a String or Binary dictionary");
        let is_string = self.kind == ByteKind::String;
        debug_assert!(
            !is_string || added.iter().all(|v| std::str::from_utf8(v).is_ok()),
            "added values of a String dictionary must be valid UTF-8"
        );
        let old_len = self.len();
        let mut values = MutableBinaryViewArray::<[u8]>::with_capacity(old_len + added.len());
        for buffer in old_bytes.data_buffers().iter() {
            values.push_buffer(buffer.clone());
        }
        // SAFETY: the views index the buffers just pushed, in the same order.
        unsafe {
            values.views_mut().extend(old_bytes.views().iter());
            values.set_total_bytes_len(old_bytes.total_bytes_len());
        }
        values.extend_values(added.iter().copied());
        let values: BinaryViewArray = values.freeze();

        // SAFETY: the codes in the tables are in bounds.
        let value = |k: &u32| unsafe { values.value_unchecked(*k as usize) };
        let lookup = OnceLock::new();
        if let Some(old) = self.lookup.get() {
            let mut new = old.clone();
            for (i, v) in added.iter().enumerate() {
                let h = new.random_state.hash_one(*v);
                let random_state = &new.random_state;
                new.codes
                    .insert_unique(h, (old_len + i) as u32, |k| random_state.hash_one(value(k)));
            }
            _ = lookup.set(new);
        }

        let views = values.views();
        let buffers = values.data_buffers();
        let hashes = self
            .hashes
            .lock()
            .unwrap()
            .iter()
            .map(|c| {
                let tail = views[old_len..]
                    .iter()
                    // SAFETY: the views are valid for the buffers.
                    .map(|v| unsafe { v.hash_with_buffers_unchecked(buffers, &c.random_state) });
                CachedHashes {
                    fingerprint: c.fingerprint,
                    random_state: c.random_state.clone(),
                    hashes: c
                        .hashes
                        .iter()
                        .copied()
                        .chain(tail)
                        .collect::<Vec<_>>()
                        .into(),
                }
            })
            .collect();

        let series = if is_string {
            // SAFETY: a String dictionary holds valid UTF-8.
            let values = unsafe { values.to_utf8view_unchecked() };
            StringChunked::with_chunk(PlSmallStr::EMPTY, values).into_series()
        } else {
            BinaryChunked::with_chunk(PlSmallStr::EMPTY, values.clone()).into_series()
        };
        Self::make(series, Some(values), hashes, lookup)
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// The values as bytes, for a String or Binary dictionary; `None` for any other dtype.
    pub fn bytes(&self) -> Option<&BinaryViewArray> {
        self.bytes.as_ref()
    }

    /// Whether the values are String rather than Binary.
    pub fn is_string(&self) -> bool {
        self.kind == ByteKind::String
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    /// The values as a [`Series`] of `name`, sharing the buffers.
    pub fn values_series(&self, name: PlSmallStr) -> Series {
        let mut values = self.values.clone();
        values.rename(name);
        values
    }

    /// The value of `code`.
    ///
    /// # Safety
    /// `code` must be in bounds.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, code: u32) -> AnyValue<'_> {
        match (self.kind, &self.bytes) {
            (ByteKind::String, Some(bytes)) => {
                let value = unsafe { bytes.value_unchecked(code as usize) };
                // SAFETY: a String dictionary holds valid UTF-8.
                AnyValue::String(unsafe { std::str::from_utf8_unchecked(value) })
            },
            (ByteKind::Binary, Some(bytes)) => {
                AnyValue::Binary(unsafe { bytes.value_unchecked(code as usize) })
            },
            _ => unsafe { self.values.get_unchecked(code as usize) },
        }
    }

    /// Estimated size of the values, see [`Series::estimated_size`].
    fn estimated_size(&self) -> usize {
        match &self.bytes {
            Some(bytes) => polars_arrow::compute::aggregate::estimated_bytes_size(bytes),
            None => self.values.estimated_size(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The hashes [`Self::hashes`] gives, if they were already computed (never for a dictionary of
    /// another dtype than String or Binary).
    pub fn cached_hashes(&self, random_state: &PlRandomState) -> Option<Buffer<u64>> {
        let fingerprint = hasher_fingerprint(random_state);
        self.hashes
            .lock()
            .unwrap()
            .iter()
            .find(|c| c.fingerprint == fingerprint)
            .map(|c| c.hashes.clone())
    }

    /// The hash of each value as `View::hash_with_buffers_unchecked` computes it with
    /// `random_state`, computed once per hasher. `None` for a dictionary of another dtype than
    /// String or Binary, whose keys hash as their materialized values.
    pub fn hashes(&self, random_state: &PlRandomState) -> Option<Buffer<u64>> {
        let bytes = self.bytes.as_ref()?;
        if let Some(hashes) = self.cached_hashes(random_state) {
            return Some(hashes);
        }
        let fingerprint = hasher_fingerprint(random_state);

        let buffers = bytes.data_buffers();
        let hashes: Buffer<u64> = bytes
            .views()
            .iter()
            .map(|v| unsafe { v.hash_with_buffers_unchecked(buffers, random_state) })
            .collect::<Vec<_>>()
            .into();

        let mut cache = self.hashes.lock().unwrap();
        if !cache.iter().any(|c| c.fingerprint == fingerprint) {
            if cache.len() == MAX_CACHED_HASHERS {
                cache.remove(0);
            }
            cache.push(CachedHashes {
                fingerprint,
                random_state: random_state.clone(),
                hashes: hashes.clone(),
            });
        }
        Some(hashes)
    }
}

/// A [`Column`] of any dtype held as codes into a [`SharedDictionary`]: row `i`, when valid, is
/// the dictionary's value at `codes[i]`.
///
/// This is lazily materialized into a [`Series`] that gathers the values.
#[derive(Debug, Clone)]
pub struct DictionaryColumn {
    name: PlSmallStr,
    dict: Arc<SharedDictionary>,
    // invariants:
    // codes[i] < dict.len() for every valid row i
    codes: UInt32Chunked,
    /// Whether the values are sorted, a hint as on a [`Series`].
    sorted: IsSorted,

    /// A lazily materialized [`Series`] variant of this [`DictionaryColumn`]. Clones share it, so
    /// that the codes are gathered once however many copies of the column consume them; changing
    /// the name or the sorted flag of one gives it a cache of its own.
    materialized: Arc<OnceLock<Series>>,
}

impl DictionaryColumn {
    /// # Safety
    /// Every valid code must be in bounds for `dict`.
    pub unsafe fn new_unchecked(
        name: PlSmallStr,
        dict: Arc<SharedDictionary>,
        codes: UInt32Chunked,
    ) -> Self {
        debug_assert!(
            codes.max().is_none_or(|m| (m as usize) < dict.len()),
            "dictionary code out of bounds"
        );
        Self {
            name,
            dict,
            codes: codes.with_name(PlSmallStr::EMPTY),
            sorted: IsSorted::Not,
            materialized: Arc::default(),
        }
    }

    pub fn try_new(
        name: PlSmallStr,
        dict: Arc<SharedDictionary>,
        codes: UInt32Chunked,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            codes.max().is_none_or(|m| (m as usize) < dict.len()),
            ComputeError: "dictionary code out of bounds"
        );
        Ok(unsafe { Self::new_unchecked(name, dict, codes) })
    }

    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    /// A cache of its own, holding `series` if there is one.
    fn cache_of(series: Option<Series>) -> Arc<OnceLock<Series>> {
        let cache = OnceLock::new();
        if let Some(series) = series {
            _ = cache.set(series);
        }
        Arc::new(cache)
    }

    /// Applies `f` to the cached materialization, if any: in place when this column is its only
    /// owner, else on a copy that becomes this column's cache of its own.
    fn update_cache(&mut self, f: impl FnOnce(&mut Series)) {
        if let Some(cache) = Arc::get_mut(&mut self.materialized) {
            if let Some(s) = cache.get_mut() {
                f(s);
            }
            return;
        }
        self.materialized = Self::cache_of(self.materialized.get().map(|s| {
            let mut s = s.clone();
            f(&mut s);
            s
        }));
    }

    pub fn rename(&mut self, name: PlSmallStr) {
        self.update_cache(|s| {
            s.rename(name.clone());
        });
        self.name = name;
    }

    pub fn dtype(&self) -> &DataType {
        self.dict.dtype()
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    pub fn null_count(&self) -> usize {
        self.codes.null_count()
    }

    pub fn has_nulls(&self) -> bool {
        self.codes.has_nulls()
    }

    pub fn dict(&self) -> &Arc<SharedDictionary> {
        &self.dict
    }

    pub fn codes(&self) -> &UInt32Chunked {
        &self.codes
    }

    /// A column of the same dictionary with `codes`, which are not known to be sorted.
    fn with_codes(&self, codes: UInt32Chunked) -> Self {
        Self {
            name: self.name.clone(),
            dict: self.dict.clone(),
            codes,
            sorted: IsSorted::Not,
            materialized: Arc::default(),
        }
    }

    /// As [`Self::with_codes`], for codes that keep the order of this column's rows.
    fn with_order_kept(&self, codes: UInt32Chunked) -> Self {
        Self {
            sorted: self.sorted,
            ..self.with_codes(codes)
        }
    }

    pub fn is_sorted_flag(&self) -> IsSorted {
        self.sorted
    }

    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        self.update_cache(|s| {
            s.set_sorted_flag(sorted);
        });
        self.sorted = sorted;
    }

    /// Gathers the values of `codes`, which are in bounds, into a [`Series`] of this column's name.
    fn gather(&self, codes: &UInt32Chunked) -> Series {
        let values = self.dict.values_series(self.name.clone());
        #[cfg(feature = "bigidx")]
        let idx = codes.cast(&IDX_DTYPE).unwrap();
        #[cfg(feature = "bigidx")]
        let idx = idx.idx().unwrap();
        #[cfg(not(feature = "bigidx"))]
        let idx = codes;
        // SAFETY: the codes are in bounds (invariant).
        unsafe { values.take_unchecked(idx) }
    }

    /// Gathers the values into a [`Series`]: chunk by chunk in parallel when there are many rows
    /// in several chunks, as collected morsels are.
    pub fn to_series(&self) -> Series {
        /// Fewer rows are gathered on this thread.
        const MIN_ROWS_TO_PARALLELIZE: usize = 1 << 17;

        let n_chunks = self.codes.chunks().len();
        let mut s = if n_chunks > 1 && self.len() >= MIN_ROWS_TO_PARALLELIZE {
            let mut offset = 0i64;
            let slices: Vec<UInt32Chunked> = self
                .codes
                .chunk_lengths()
                .map(|len| {
                    let part = self.codes.slice(offset, len);
                    offset += len as i64;
                    part
                })
                .collect();
            let parts: Vec<Series> =
                RAYON.install(|| slices.par_iter().map(|c| self.gather(c)).collect());
            let mut parts = parts.into_iter();
            let mut acc = parts.next().unwrap();
            for part in parts {
                acc.append_owned(part).unwrap();
            }
            acc
        } else {
            self.gather(&self.codes)
        };
        s.set_sorted_flag(self.sorted);
        s
    }

    /// Get the [`DictionaryColumn`] as [`Series`] if it was already materialized.
    pub fn lazy_as_materialized_series(&self) -> Option<&Series> {
        self.materialized.get()
    }

    /// Get the [`DictionaryColumn`] as [`Series`]
    ///
    /// This needs to materialize upon the first call. Afterwards, this is cached.
    pub fn as_materialized_series(&self) -> &Series {
        self.materialized.get_or_init(|| self.to_series())
    }

    /// Take the [`DictionaryColumn`] and materialize as a [`Series`] if not already done.
    pub fn take_materialized_series(self) -> Series {
        self.as_materialized_series().clone()
    }

    /// `out`, a column of the same dictionary derived from this one, with the materialized values
    /// (if already there) derived by `f` in the same way.
    fn with_cache_from(&self, mut out: Self, f: impl FnOnce(&Series) -> Series) -> Self {
        if let Some(m) = self.materialized.get() {
            out.materialized = Self::cache_of(Some(f(m)));
        }
        out
    }

    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let out = self.with_order_kept(self.codes.slice(offset, length));
        self.with_cache_from(out, |m| m.slice(offset, length))
    }

    pub fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Self> {
        Ok(self.with_order_kept(self.codes.filter(filter)?))
    }

    /// # Safety
    /// The indices must be in bounds.
    pub unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        self.with_codes(unsafe { self.codes.take_unchecked(indices) })
    }

    /// # Safety
    /// The indices must be in bounds.
    pub unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Self {
        self.with_codes(unsafe { self.codes.take_unchecked(indices) })
    }

    pub fn rechunk(&self) -> Self {
        if self.codes.chunks().len() <= 1 {
            return self.clone();
        }
        self.with_order_kept(self.codes.rechunk().into_owned())
    }

    pub fn reverse(&self) -> Self {
        let mut out = self.with_codes(self.codes.reverse());
        out.sorted = self.sorted.reverse();
        self.with_cache_from(out, |m| {
            let mut r = m.reverse();
            r.set_sorted_flag(self.sorted.reverse());
            r
        })
    }

    pub fn clear(&self) -> Self {
        self.with_codes(self.codes.clear())
    }

    /// Appends `other`'s codes. With another dictionary, `other`'s values that this dictionary
    /// lacks are added to it (into a new dictionary, unless there are none) and `other`'s codes
    /// remapped, if the merged dictionary has no more values than the rows of both, as otherwise
    /// it costs more than materializing; returns false then.
    ///
    /// Values are matched by value, so appending the chunks of many row groups whose dictionaries
    /// share their values keeps one dictionary of the distinct values. Once the lookup table
    /// exists, remapping costs the size of `other`'s dictionary and codes. Extending also copies
    /// the existing dictionary's views, lookup and cached hashes, so this method returns false
    /// when that extension would copy more existing entries than `other` has rows.
    pub fn try_append(&mut self, other: &DictionaryColumn) -> bool {
        if self.dict.id != other.dict.id {
            // Only String and Binary values are merged by their bytes, and never one with the
            // other.
            let (Some(_), Some(other_bytes)) = (&self.dict.bytes, &other.dict.bytes) else {
                return false;
            };
            if self.dict.is_string() != other.dict.is_string() {
                return false;
            }
            let max_len = (self.len() + other.len()).min(u32::MAX as usize);
            if self.dict.len() > max_len {
                return false;
            }
            let mut added = Vec::new();
            let mut added_code_of = PlHashMap::<&[u8], u32>::new();
            let mut remap = Vec::with_capacity(other.dict.len());
            for v in other_bytes.values_iter() {
                let c = match self.dict.code_of(v) {
                    Some(c) => c,
                    None => {
                        let next = (self.dict.len() + added.len()) as u32;
                        *added_code_of.entry(v).or_insert_with(|| {
                            added.push(v);
                            next
                        })
                    },
                };
                remap.push(c);
                if self.dict.len() + added.len() > max_len {
                    return false;
                }
            }
            if !added.is_empty() && self.dict.len() > other.len() {
                return false;
            }
            // Null slots may hold any value, out of bounds too.
            let other_codes = other
                .codes
                .apply_values(|c| remap.get(c as usize).copied().unwrap_or(0));
            if !added.is_empty() {
                self.dict = self.dict.extended(&added);
            }
            self.codes.append(&other_codes).unwrap();
        } else {
            self.codes.append(&other.codes).unwrap();
        }
        self.sorted = IsSorted::Not;
        self.materialized = Arc::default();
        true
    }

    /// Concatenates `columns`, of one dtype, without gathering their values: the result's
    /// dictionary holds each distinct dictionary (by id) of the columns one after the other, and
    /// their codes, shifted by where their dictionary starts, are chunks of its codes.
    ///
    /// Values repeated across the dictionaries are not merged (codes of equal values may differ).
    /// The shifting runs in parallel, in place in the codes that nothing else holds (a fresh
    /// buffer costs more than the shift), and skips the first dictionary's columns.
    ///
    /// Gives the columns back for columns with a sorted flag (appending their values keeps it
    /// when the columns continue each other), if the dictionaries have more values than
    /// `rows / 8` (as materializing is cheaper then), or more than `u32` can number.
    ///
    /// The hashes of the merged dictionary's values are not cached.
    pub fn concat(columns: Vec<DictionaryColumn>) -> Result<Self, Vec<DictionaryColumn>> {
        let Some(first) = columns.first() else {
            return Err(columns);
        };
        let total_rows: usize = columns.iter().map(|c| c.len()).sum();
        let mut offset_of_id = PlHashMap::<u64, u32>::new();
        let mut dicts: Vec<&Arc<SharedDictionary>> = Vec::new();
        let mut offsets = Vec::with_capacity(columns.len());
        let mut merged_len = 0usize;
        for c in &columns {
            // A sorted flag is kept by appending values, which compares the boundaries.
            if c.dtype() != first.dtype() || c.sorted != IsSorted::Not {
                return Err(columns);
            }
            let offset = match offset_of_id.get(&c.dict.id) {
                Some(&o) => o,
                None => {
                    let o = merged_len;
                    merged_len += c.dict.len();
                    if merged_len > u32::MAX as usize {
                        return Err(columns);
                    }
                    offset_of_id.insert(c.dict.id, o as u32);
                    dicts.push(&c.dict);
                    o as u32
                },
            };
            offsets.push(offset);
        }
        let dict = if let [only] = dicts.as_slice() {
            (*only).clone()
        } else {
            if merged_len.saturating_mul(MIN_ROWS_PER_MERGED_VALUE) > total_rows {
                return Err(columns);
            }
            let mut values = dicts[0].values.clone();
            for d in &dicts[1..] {
                if values.append(&d.values).is_err() {
                    return Err(columns);
                }
            }
            match SharedDictionary::try_new(values) {
                Ok(dict) => dict,
                Err(_) => return Err(columns),
            }
        };
        let name = first.name.clone();

        let parts: Vec<UInt32Chunked> = RAYON.install(|| {
            columns
                .into_par_iter()
                .zip(offsets)
                .map(|(c, offset)| {
                    if offset == 0 {
                        c.codes
                    } else {
                        // Null slots may hold any code: they wrap, staying unused.
                        c.codes
                            .apply_in_place(move |code| code.wrapping_add(offset))
                    }
                })
                .collect()
        });
        let mut parts = parts.into_iter();
        let mut codes = parts.next().unwrap();
        for part in parts {
            codes.append_owned(part).unwrap();
        }
        // SAFETY: each valid code is in bounds for its dictionary, which starts at its offset in
        // the merged one.
        Ok(unsafe { Self::new_unchecked(name, dict, codes) })
    }

    /// The codes of this column shifted by `offset`, the start of its dictionary in a merged
    /// dictionary (see [`DictionaryMerger`]). Shifts in place in codes that nothing else holds.
    pub fn into_shifted_codes(self, offset: u32) -> UInt32Chunked {
        if offset == 0 {
            return self.codes;
        }
        // Null slots may hold any code: they wrap, staying unused.
        self.codes
            .apply_in_place(move |code| code.wrapping_add(offset))
    }

    /// Estimated size of the column, see [`Series::estimated_size`].
    ///
    /// The dictionary is shared by every slice of a column, so each counts the part of it its rows
    /// would take materialized, at most all of it; counting all of it for each slice would
    /// overstate the memory of the morsels of a row group many times over.
    pub fn estimated_size(&self) -> usize {
        let dict_size = self.dict.estimated_size();
        let dict_part = if self.len() >= self.dict.len() {
            dict_size
        } else {
            (dict_size as u128 * self.len() as u128 / self.dict.len() as u128) as usize
        };
        // Clones share the cache: each counts its share, so that the holders add up to the cache
        // once. A cached slice of a nested column counts all of its child values, however few
        // rows it has, so there the cache is bounded by what this column's rows would take at the
        // dictionary's average value size.
        let cache_size = self.materialized.get().map_or(0, |s| {
            let mut size = s.estimated_size();
            if s.dtype().is_nested() && !self.dict.is_empty() {
                let rows_size = (self.dict.estimated_size() as u128 * self.len() as u128
                    / self.dict.len() as u128) as usize;
                size = size.min(rows_size);
            }
            size / Arc::strong_count(&self.materialized)
        });
        self.codes
            .chunks()
            .iter()
            .map(|c| polars_arrow::compute::aggregate::estimated_bytes_size(c.as_ref()))
            .sum::<usize>()
            + dict_part
            + cache_size
    }

    /// The codes that valid rows use, in order of first use, and the position of each row's code
    /// among them (0 for a null row). `None` once more than `max_used` codes are used.
    ///
    /// The time this takes depends on the rows, not on the size of the dictionary, which can be
    /// far larger than the rows need after a filter or slice.
    pub fn used_codes(&self, max_used: usize) -> Option<(Vec<IdxSize>, Vec<u32>)> {
        /// A dense table of the dictionary's size is used up to this many entries per row.
        const MAX_DENSE_ENTRIES_PER_ROW: usize = 4;

        let codes = self.codes.rechunk();
        let codes = codes.downcast_as_array();
        let validity = codes.validity().filter(|v| v.unset_bits() > 0);

        // `slot_of(code, next)` is the position of `code`, `next` if it is new.
        fn assign(
            codes: &[u32],
            validity: Option<&polars_arrow::bitmap::Bitmap>,
            max_used: usize,
            mut slot_of: impl FnMut(u32, u32) -> u32,
        ) -> Option<(Vec<IdxSize>, Vec<u32>)> {
            let mut used: Vec<IdxSize> = Vec::new();
            let mut row_slots = Vec::with_capacity(codes.len());
            for (i, c) in codes.iter().enumerate() {
                if validity.is_some_and(|v| !unsafe { v.get_bit_unchecked(i) }) {
                    row_slots.push(0);
                    continue;
                }
                let next = used.len() as u32;
                let s = slot_of(*c, next);
                if s == next {
                    if used.len() >= max_used {
                        return None;
                    }
                    used.push(*c as IdxSize);
                }
                row_slots.push(s);
            }
            Some((used, row_slots))
        }

        if self.dict.len() <= MAX_DENSE_ENTRIES_PER_ROW.saturating_mul(self.len()) {
            let mut slot = vec![u32::MAX; self.dict.len()];
            assign(codes.values(), validity, max_used, |c, next| {
                // SAFETY: valid codes are in bounds (invariant).
                let s = unsafe { slot.get_unchecked_mut(c as usize) };
                if *s == u32::MAX {
                    *s = next;
                }
                *s
            })
        } else {
            let mut slot = polars_utils::aliases::PlHashMap::<u32, u32>::default();
            assign(codes.values(), validity, max_used, |c, next| {
                *slot.entry(c).or_insert(next)
            })
        }
    }

    /// Evaluates an elementwise `f` once per dictionary value that a valid row uses, and once on
    /// a null if a row is null, and spreads the results to the rows by code. A result of any dtype (but
    /// Null and Object) stays dictionary-coded, its dictionary the results. `f` gets a [`Series`] of this column's name and dtype and
    /// returns one value per input value.
    ///
    /// Only values that rows use are evaluated, so `f` errors exactly when it would on the rows.
    /// When that is not fewer values than rows, `f` is evaluated on the materialized column.
    pub fn apply_elementwise(
        &self,
        f: impl FnOnce(&Series) -> PolarsResult<Series>,
    ) -> PolarsResult<Column> {
        let max_evaluated = self.len() / 2;
        let Some((used, mut row_slots)) = self.used_codes(max_evaluated) else {
            return f(self.as_materialized_series()).map(Column::from);
        };
        let null_slot = used.len() as u32;

        let codes = self.codes.rechunk();
        let validity = codes
            .downcast_as_array()
            .validity()
            .filter(|v| v.unset_bits() > 0);
        if let Some(v) = validity {
            for (s, is_valid) in row_slots.iter_mut().zip(v.iter()) {
                if !is_valid {
                    *s = null_slot;
                }
            }
        }

        let values = self.dict.values_series(self.name.clone());
        let mut input = unsafe { values.take_unchecked(&IdxCa::from_vec(PlSmallStr::EMPTY, used)) };
        if validity.is_some() {
            input.append(&Series::full_null(PlSmallStr::EMPTY, 1, self.dtype()))?;
        }
        let out = f(&input)?;
        polars_ensure!(
            out.len() == input.len(),
            ComputeError: "elementwise function changed the length of a dictionary column"
        );

        if matches!(out.dtype(), DataType::String | DataType::Binary) {
            // Rows whose result is null get a null code.
            let name = out_name(&input, &out, &self.name);
            let (out, is_string) = match out.dtype() {
                DataType::String => (out.str().unwrap().as_binary(), true),
                _ => (out.binary().unwrap().clone(), false),
            };
            let out = out.rechunk();
            let out = out.downcast_as_array();
            let out_validity = out.validity().filter(|v| v.unset_bits() > 0);
            let row_validity = out_validity.map(|ov| {
                row_slots
                    .iter()
                    .map(|s| unsafe { ov.get_bit_unchecked(*s as usize) })
                    .collect::<polars_arrow::bitmap::Bitmap>()
            });
            let dict = SharedDictionary::with_values(out.clone().with_validity(None), is_string);
            let codes =
                UInt32Chunked::from_vec_validity(PlSmallStr::EMPTY, row_slots, row_validity);
            // SAFETY: every slot indexes `out`.
            return Ok(unsafe { DictionaryColumn::new_unchecked(name, dict, codes) }.into_column());
        }

        if can_be_dictionary_values(out.dtype()) {
            // The dictionary holds the valid results; rows whose result is null get a null code.
            let name = out_name(&input, &out, &self.name);
            let out = out.rechunk();
            let (out, row_slots, row_validity) = match out.rechunk_validity() {
                Some(ov) if ov.unset_bits() > 0 => {
                    let mut rank = Vec::with_capacity(ov.len());
                    let mut next = 0u32;
                    for is_valid in ov.iter() {
                        rank.push(next);
                        next += is_valid as u32;
                    }
                    let row_validity = row_slots
                        .iter()
                        .map(|s| unsafe { ov.get_bit_unchecked(*s as usize) })
                        .collect::<polars_arrow::bitmap::Bitmap>();
                    // A null row's code is never read: 0 keeps it in bounds when the dictionary
                    // has a value (`rank` of a null result can be one past the last).
                    let row_slots = row_slots
                        .into_iter()
                        .map(|s| {
                            let r = unsafe { *rank.get_unchecked(s as usize) };
                            if (r as usize) < next as usize { r } else { 0 }
                        })
                        .collect();
                    let out = out.filter(&BooleanChunked::with_chunk(
                        PlSmallStr::EMPTY,
                        polars_arrow::array::BooleanArray::from_data_default(ov, None),
                    ))?;
                    (out, row_slots, Some(row_validity))
                },
                _ => (out, row_slots, None),
            };
            let dict = SharedDictionary::try_new(out)?;
            let codes =
                UInt32Chunked::from_vec_validity(PlSmallStr::EMPTY, row_slots, row_validity);
            // SAFETY: every slot indexes the valid results.
            return Ok(unsafe { DictionaryColumn::new_unchecked(name, dict, codes) }.into_column());
        }

        #[cfg(feature = "bigidx")]
        let idx = IdxCa::from_vec(
            PlSmallStr::EMPTY,
            row_slots.into_iter().map(|s| s as IdxSize).collect(),
        );
        #[cfg(not(feature = "bigidx"))]
        let idx = IdxCa::from_vec(PlSmallStr::EMPTY, row_slots);
        // SAFETY: every slot indexes `out`.
        Ok(unsafe { out.take_unchecked(&idx) }.into_column())
    }

    /// Checks that the materialized values match the codes.
    pub fn check_invariants(&self) {
        let expected = self.to_series();
        if let Some(m) = self.materialized.get() {
            assert!(m.equals_missing(&expected));
        }
        assert_eq!(expected.len(), self.len());
    }
}

/// Whether a dictionary can hold values of `dtype`: any but Null, Object and an unknown one.
fn can_be_dictionary_values(dtype: &DataType) -> bool {
    !(dtype.is_null() || dtype.is_object() || dtype.is_unknown())
}

/// The name of `out`, which `f` computed from `input`: `name` unless `f` renamed it.
fn out_name(input: &Series, out: &Series, name: &PlSmallStr) -> PlSmallStr {
    if out.name() == input.name() {
        name.clone()
    } else {
        out.name().clone()
    }
}

impl IntoColumn for DictionaryColumn {
    #[inline(always)]
    fn into_column(self) -> Column {
        Column::Dictionary(self)
    }
}

impl From<DictionaryColumn> for Column {
    #[inline]
    fn from(value: DictionaryColumn) -> Self {
        value.into_column()
    }
}

/// Merges the dictionaries of the chunks of one column, as the chunks arrive: the merged dictionary
/// holds each distinct dictionary (by id) one after the other, so each chunk's codes only need
/// shifting by where its dictionary starts, which can be done as it arrives, in parallel with
/// other chunks arriving. Values repeated across the dictionaries are not merged.
///
/// The rows a value of the merged dictionary has to cover.
const MIN_ROWS_PER_MERGED_VALUE: usize = 8;

/// How far (in rows' worth) the merged dictionary may outgrow the rows seen while chunks arrive,
/// as chunks of a row group arrive before the rest of its rows.
pub const MAX_MERGED_SLACK_ROWS: usize = 1 << 24;

/// Chunks the merger cannot take (a sorted flag, another dtype, more values than `u32` numbers)
/// make it fail for good, see [`Self::register`].
#[derive(Default, Debug)]
pub struct DictionaryMerger {
    dtype: Option<DataType>,
    offset_of_id: PlHashMap<u64, u32>,
    dicts: Vec<Arc<SharedDictionary>>,
    /// Where each of `dicts` starts in the merged dictionary, ascending.
    offsets: Vec<u32>,
    merged_len: usize,
    /// The rows of the chunks registered or skipped so far.
    rows: usize,
    failed: bool,
}

impl DictionaryMerger {
    /// Whether no dictionary was registered.
    pub fn is_empty(&self) -> bool {
        self.dicts.is_empty()
    }

    /// Whether a column was refused: the others' codes are then not usable on their own.
    pub fn failed(&self) -> bool {
        self.failed
    }

    /// Refuses every column from now on, as for one that cannot be merged.
    pub fn fail(&mut self) {
        self.failed = true;
    }

    /// Counts `rows` rows that need no dictionary (a chunk of nulls).
    pub fn skip_rows(&mut self, rows: usize) {
        self.rows += rows;
    }

    /// Registers the dictionary of `column` and gives where it starts in the merged dictionary,
    /// for [`DictionaryColumn::into_shifted_codes`]. `None` (and every later call too) if the
    /// column cannot be merged, or if the merged dictionary outgrows the rows seen so far by
    /// more than `MAX_MERGED_SLACK_ROWS` (rows' worth), as every dictionary registered is held
    /// until the output is made, however few rows use it.
    pub fn register(&mut self, column: &DictionaryColumn) -> Option<u32> {
        if self.failed
            || column.sorted != IsSorted::Not
            || self.dtype.as_ref().is_some_and(|d| d != column.dtype())
        {
            self.failed = true;
            return None;
        }
        if let Some(&o) = self.offset_of_id.get(&column.dict.id) {
            self.rows += column.len();
            return Some(o);
        }
        let offset = self.merged_len;
        let merged_len = offset + column.dict.len();
        let rows = self.rows + column.len();
        if merged_len > u32::MAX as usize
            || merged_len.saturating_mul(MIN_ROWS_PER_MERGED_VALUE)
                > rows.saturating_add(MAX_MERGED_SLACK_ROWS)
        {
            self.failed = true;
            return None;
        }
        self.rows = rows;
        self.dtype = Some(column.dtype().clone());
        self.merged_len = merged_len;
        self.offset_of_id.insert(column.dict.id, offset as u32);
        self.dicts.push(column.dict.clone());
        self.offsets.push(offset as u32);
        Some(offset as u32)
    }

    /// The values of `codes`, the shifted codes of one registered chunk, gathered from the
    /// chunk's own dictionary (not the merged one).
    pub fn gather_chunk(&self, name: PlSmallStr, codes: UInt32Chunked) -> PolarsResult<Series> {
        let Some(first) = codes.iter().flatten().next() else {
            let dtype = self.dtype.clone().unwrap_or(DataType::Null);
            return Ok(Series::full_null(name, codes.len(), &dtype));
        };
        let i = self.offsets.partition_point(|&o| o <= first);
        polars_ensure!(i > 0, ComputeError: "dictionary code outside the merged dictionary");
        let (dict, offset) = (self.dicts[i - 1].clone(), self.offsets[i - 1]);
        let codes = if offset == 0 {
            codes
        } else {
            // Null slots may hold any code: they wrap back to what they were.
            codes.apply_in_place(move |code| code.wrapping_sub(offset))
        };
        let col = DictionaryColumn::try_new(name, dict, codes)?;
        Ok(col.to_series())
    }

    /// Whether the merged dictionary has at most `rows / 8` values, as it costs more than
    /// materializing the columns when it has more.
    pub fn is_dense_enough(&self, rows: usize) -> bool {
        self.dicts.len() <= 1 || self.merged_len.saturating_mul(MIN_ROWS_PER_MERGED_VALUE) <= rows
    }

    /// The merged dictionary: the only dictionary registered, if there is just one.
    pub fn dictionary(&self) -> PolarsResult<Arc<SharedDictionary>> {
        match self.dicts.as_slice() {
            [] => polars_bail!(ComputeError: "no dictionary was registered"),
            [only] => Ok(only.clone()),
            [first, rest @ ..] => {
                let mut values = first.values.clone();
                for d in rest {
                    values.append(&d.values)?;
                }
                SharedDictionary::try_new(values)
            },
        }
    }

    /// The column of `name` with `chunks`, the shifted codes of the registered columns in order.
    pub fn into_column(
        &self,
        name: PlSmallStr,
        chunks: impl IntoIterator<Item = UInt32Chunked>,
    ) -> PolarsResult<DictionaryColumn> {
        let dict = self.dictionary()?;
        let mut chunks = chunks.into_iter();
        let mut codes = chunks.next().expect("at least one chunk");
        for part in chunks {
            codes.append_owned(part)?;
        }
        // SAFETY: each valid code of a chunk is in bounds for its dictionary, which starts at the
        // offset the chunk was shifted by in the merged one.
        Ok(unsafe { DictionaryColumn::new_unchecked(name, dict, codes) })
    }
}

/// Concatenates `columns` (not empty, of one dtype) vertically. Dictionary columns are
/// concatenated by their codes ([`DictionaryColumn::concat`]) and, where that does not pay off,
/// materialized in parallel; other columns are appended one after the other.
pub fn concat_columns(columns: Vec<Column>) -> Column {
    try_concat_columns(columns).expect("columns of one dtype")
}

/// [`concat_columns`], an error if the dtypes differ.
pub fn try_concat_columns(mut columns: Vec<Column>) -> PolarsResult<Column> {
    assert!(!columns.is_empty());
    if columns.len() == 1 {
        return Ok(columns.pop().unwrap());
    }
    if columns.iter().all(|c| c.as_dictionary_column().is_some()) {
        let coded = columns.into_iter().map(|c| match c {
            Column::Dictionary(d) => d,
            _ => unreachable!(),
        });
        let columns: Vec<Column> = match DictionaryColumn::concat(coded.collect()) {
            Ok(out) => return Ok(out.into_column()),
            Err(coded) => coded.into_iter().map(Column::Dictionary).collect(),
        };
        let series: Vec<Series> = RAYON.install(|| {
            columns
                .par_iter()
                .map(|c| c.as_materialized_series().clone())
                .collect()
        });
        let mut series = series.into_iter();
        let mut acc = series.next().unwrap();
        for s in series {
            acc.append_owned(s)?;
        }
        return Ok(acc.into_column());
    }
    let mut columns = columns.into_iter();
    let mut acc = columns.next().unwrap();
    for c in columns {
        acc.append_owned(c)?;
    }
    Ok(acc)
}

/// Whether dictionary-coded columns are produced and used, `POLARS_DICT_CODES=0` turns them
/// off.
pub fn dictionary_codes_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("POLARS_DICT_CODES").as_deref() != Ok("0"))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::flags::StatisticsFlags;

    fn dict_column(dict: &[&str], codes: &[Option<u32>]) -> DictionaryColumn {
        let values = Utf8ViewArray::from_slice_values(dict);
        let codes = UInt32Chunked::from_iter_options(PlSmallStr::EMPTY, codes.iter().copied());
        DictionaryColumn::try_new("s".into(), SharedDictionary::new(values), codes).unwrap()
    }

    fn values(c: &DictionaryColumn) -> Vec<Option<String>> {
        c.to_series()
            .str()
            .unwrap()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    }

    fn binary_dict_column(dict: &[&[u8]], codes: &[Option<u32>]) -> DictionaryColumn {
        let values = BinaryViewArray::from_slice_values(dict);
        let codes = UInt32Chunked::from_iter_options(PlSmallStr::EMPTY, codes.iter().copied());
        DictionaryColumn::try_new("b".into(), SharedDictionary::new_binary(values), codes).unwrap()
    }

    #[test]
    fn binary_dictionary_materializes_bytes() {
        let invalid: &[u8] = &[0xff, 0xfe];
        let a = binary_dict_column(
            &[b"x", invalid],
            &[Some(1), None, Some(0), Some(0), Some(0), Some(0)],
        );
        assert_eq!(a.dtype(), &DataType::Binary);
        a.check_invariants();
        assert_eq!(
            a.to_series().binary().unwrap().iter().collect::<Vec<_>>(),
            [
                Some(invalid),
                None,
                Some(&b"x"[..]),
                Some(b"x"),
                Some(b"x"),
                Some(b"x")
            ]
        );
        let col = Column::from(a.clone());
        assert_eq!(col.dtype(), &DataType::Binary);
        assert_eq!(col.get(0).unwrap(), AnyValue::Binary(invalid));
        assert!(col.get(1).unwrap().is_null());
        // Casting to its own dtype keeps the codes; to String fails on invalid UTF-8 only if a
        // row uses it.
        assert!(
            col.cast(&DataType::Binary)
                .unwrap()
                .as_dictionary_column()
                .is_some()
        );
        assert!(col.strict_cast(&DataType::String).is_err());
        let valid = col.slice(2, 4).strict_cast(&DataType::String).unwrap();
        assert_eq!(valid.dtype(), &DataType::String);
        assert_eq!(valid.str().unwrap().get(0), Some("x"));
        assert!(valid.as_dictionary_column().is_some());
    }

    #[test]
    fn try_append_does_not_mix_string_and_binary() {
        let mut a = dict_column(&["x"], &[Some(0)]);
        let b = binary_dict_column(&[b"x"], &[Some(0)]);
        assert!(!a.try_append(&b));
        let mut b = b;
        b.try_append(&binary_dict_column(&[b"y", b"x"], &[Some(0), Some(1)]));
        assert_eq!(b.dict().len(), 2);
        assert_eq!(b.dtype(), &DataType::Binary);
        b.check_invariants();
    }

    #[test]
    fn binary_dictionary_compares_with_constant() {
        let lhs = Column::from(binary_dict_column(&[b"a", b"b"], &[Some(0), Some(1), None]));
        let rhs = Column::new_scalar(
            "c".into(),
            Scalar::new(DataType::Binary, AnyValue::BinaryOwned(b"a".to_vec())),
            1,
        );
        let out = lhs.equal(&rhs).unwrap();
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(true), Some(false), None]
        );
    }

    #[test]
    fn try_append_merges_by_value() {
        let mut a = dict_column(&["x", "y"], &[Some(0), Some(1), None, Some(0)]);
        let id = a.dict().id();
        // The same values in another order and another dictionary: nothing is added.
        let b = dict_column(&["y", "x"], &[Some(0), None, Some(1), Some(1)]);
        assert!(a.try_append(&b));
        assert_eq!(a.dict().id(), id);
        assert_eq!(a.dict().len(), 2);
        // One new value: it is added once, after the existing ones.
        let c = dict_column(&["z", "x"], &[Some(0), Some(1), Some(0), Some(0)]);
        assert!(a.try_append(&c));
        assert_eq!(a.dict().len(), 3);
        a.check_invariants();
        let s = |v: &str| Some(v.to_owned());
        assert_eq!(
            values(&a),
            [
                s("x"),
                s("y"),
                None,
                s("x"),
                s("y"),
                None,
                s("x"),
                s("x"),
                s("z"),
                s("x"),
                s("z"),
                s("z"),
            ]
        );
    }

    #[test]
    fn try_append_extends_lookup_and_hashes() {
        let rs = PlRandomState::default();
        let mut a = dict_column(&["x", "y"], &[Some(0), Some(1)]);
        a.dict().hashes(&rs).unwrap();
        assert_eq!(a.dict().code_of(b"y"), Some(1));
        let b = dict_column(
            &["z", "a-value-longer-than-a-view", "x"],
            &[Some(0), Some(1), Some(2)],
        );
        assert!(a.try_append(&b));
        let dict = a.dict();
        assert_eq!(dict.len(), 4);
        // The lookup and the hashes were carried into the extended dictionary.
        assert!(dict.lookup.get().is_some());
        assert_eq!(dict.code_of(b"a-value-longer-than-a-view"), Some(3));
        assert_eq!(dict.code_of(b"x"), Some(0));
        assert_eq!(dict.code_of(b"absent"), None);
        let cached = dict.cached_hashes(&rs).unwrap();
        let fresh = SharedDictionary::new_binary(dict.bytes().unwrap().clone())
            .hashes(&rs)
            .unwrap();
        assert_eq!(cached.as_slice(), fresh.as_slice());
        a.check_invariants();
    }

    #[test]
    fn compare_length_one_with_longer_constant_broadcasts() {
        let lhs = Column::from(dict_column(&["a", "b"], &[Some(0)]));
        let rhs = Column::new_scalar(
            "c".into(),
            Scalar::new(DataType::String, AnyValue::StringOwned("a".into())),
            5,
        );
        let out = lhs.equal(&rhs).unwrap();
        assert_eq!(out.len(), 5);
        assert!(out.all());
        let out = rhs.not_equal(&lhs).unwrap();
        assert_eq!(out.len(), 5);
        assert!(!out.any());
    }

    fn generic_dict_column(values: Series, codes: &[Option<u32>]) -> DictionaryColumn {
        let codes = UInt32Chunked::from_iter_options(PlSmallStr::EMPTY, codes.iter().copied());
        DictionaryColumn::try_new(
            "g".into(),
            SharedDictionary::try_new(values).unwrap(),
            codes,
        )
        .unwrap()
    }

    /// The flat column the dictionary column stands for.
    fn flat(c: &DictionaryColumn) -> Column {
        c.to_series().into_column()
    }

    fn int_dict_column(n: usize) -> DictionaryColumn {
        let values = Series::new("v".into(), [10i64, 20, 30]);
        let codes: Vec<Option<u32>> = (0..n)
            .map(|i| (i % 5 != 4).then_some((i % 3) as u32))
            .collect();
        generic_dict_column(values, &codes)
    }

    fn assert_coded(out: &Column, dict_len: usize) {
        let d = out.as_dictionary_column().expect("a dictionary column");
        assert_eq!(d.dict().len(), dict_len);
        d.check_invariants();
    }

    #[test]
    fn non_byte_dictionary_basics() {
        let a = int_dict_column(12);
        assert_eq!(a.dtype(), &DataType::Int64);
        assert!(a.dict().bytes().is_none());
        assert!(a.dict().hashes(&PlRandomState::default()).is_none());
        assert_eq!(a.dict().code_of(b"x"), None);
        a.check_invariants();
        let col = Column::from(a.clone());
        let expected = flat(&a);
        assert_eq!(col.dtype(), &DataType::Int64);
        assert_eq!(col.get(0).unwrap(), AnyValue::Int64(10));
        assert_eq!(col.get(1).unwrap(), AnyValue::Int64(20));
        assert!(col.get(4).unwrap().is_null());
        for i in 0..col.len() {
            assert_eq!(col.get(i).unwrap(), expected.get(i).unwrap());
        }
        assert!(col.slice(2, 5).equals_missing(&expected.slice(2, 5)));
        assert!(col.slice(2, 5).as_dictionary_column().is_some());
        let mask = BooleanChunked::from_iter((0..12).map(|i| i % 2 == 0));
        let filtered = col.filter(&mask).unwrap();
        assert!(filtered.as_dictionary_column().is_some());
        assert!(filtered.equals_missing(&expected.filter(&mask).unwrap()));
        assert!(col.rechunk().equals_missing(&expected));
        assert!(
            col.cast(&DataType::Int64)
                .unwrap()
                .as_dictionary_column()
                .is_some()
        );
        // The values are shared when their dtype changes by a cast.
        let as_string = col.cast(&DataType::String).unwrap();
        assert_coded(&as_string, 4);
        assert!(as_string.equals_missing(&expected.cast(&DataType::String).unwrap()));
        assert!(
            as_string
                .as_dictionary_column()
                .unwrap()
                .dict()
                .bytes()
                .is_some()
        );
    }

    #[test]
    fn non_byte_dictionary_rejects_unsupported_values() {
        assert!(SharedDictionary::try_new(Series::new_null("n".into(), 2)).is_err());
        assert!(SharedDictionary::try_new(Series::new("v".into(), [Some(1i64), None])).is_err());
        assert!(SharedDictionary::try_new(Series::new("v".into(), [1i64])).is_ok());
    }

    #[test]
    fn non_byte_dictionary_try_append() {
        let mut a = int_dict_column(6);
        let same = a.slice(1, 3);
        assert!(a.try_append(&same));
        assert_eq!(a.len(), 9);
        a.check_invariants();
        // Another dictionary, even of equal values, is not merged: the caller materializes.
        let other = int_dict_column(6);
        assert!(!a.try_append(&other));
        assert_eq!(a.len(), 9);
        // A String dictionary is no match for one of another dtype either way.
        let mut strings = dict_column(&["x"], &[Some(0)]);
        assert!(!strings.try_append(&int_dict_column(1)));
        let mut ints = int_dict_column(1);
        assert!(!ints.try_append(&strings));

        let mut col = Column::from(a.clone());
        let expected = {
            let mut e = flat(&a);
            e.append(&flat(&other)).unwrap();
            e
        };
        col.append(&Column::from(other)).unwrap();
        assert!(col.as_dictionary_column().is_none());
        assert!(col.equals_missing(&expected));
    }

    #[test]
    fn non_byte_dictionary_compares_with_constant() {
        let a = int_dict_column(10);
        let lhs = Column::from(a.clone());
        let rhs = Column::new_scalar(
            "c".into(),
            Scalar::new(DataType::Int64, AnyValue::Int64(20)),
            1,
        );
        let expected = flat(&a).equal(&rhs).unwrap();
        assert!(
            lhs.equal(&rhs)
                .unwrap()
                .into_series()
                .equals_missing(&expected.clone().into_series())
        );
        assert!(
            rhs.equal(&lhs)
                .unwrap()
                .into_series()
                .equals_missing(&expected.into_series())
        );
    }

    #[test]
    fn non_byte_dictionary_binary_op_with_constant_stays_coded() {
        let a = int_dict_column(10);
        let lhs = Column::from(a.clone());
        let rhs = Column::new_scalar(
            "c".into(),
            Scalar::new(DataType::Int64, AnyValue::Int64(3)),
            1,
        );
        let out = (&lhs * &rhs).unwrap();
        assert_coded(&out, 3);
        assert!(out.equals_missing(&(&flat(&a) * &rhs).unwrap()));
        let out = (&rhs - &lhs).unwrap();
        assert!(out.equals_missing(&(&rhs - &flat(&a)).unwrap()));
        // With a column of the same length, the work is not once per value.
        let other = Column::from(Series::new("o".into(), [1i64; 10]));
        assert!((&lhs + &other).unwrap().as_dictionary_column().is_none());
    }

    #[test]
    fn apply_elementwise_keeps_results_of_any_dtype_coded() {
        let a = int_dict_column(30);
        let expected = flat(&a);
        // A numeric result whose nulls are partly computed: values below 20 become null.
        let out = a
            .apply_elementwise(|v| {
                Ok(v.i64()
                    .unwrap()
                    .apply(|x| x.filter(|x| *x >= 20))
                    .into_series())
            })
            .unwrap();
        assert_coded(&out, 2);
        let want = Series::new(
            "".into(),
            expected
                .i64()
                .unwrap()
                .iter()
                .map(|x| x.filter(|x| *x >= 20))
                .collect::<Vec<_>>(),
        );
        assert!(out.as_materialized_series().equals_missing(&want));
        // A function that is null for a null input only.
        let out = a.apply_elementwise(|v| v.cast(&DataType::Float64)).unwrap();
        assert_coded(&out, 3);
        // Chains stay coded and share the codes.
        let step1 = Column::from(a.clone()).cast(&DataType::Float64).unwrap();
        let step2 = step1
            .as_dictionary_column()
            .unwrap()
            .apply_elementwise(|v| v.cast(&DataType::String))
            .unwrap();
        assert_coded(&step2, 4);
        assert!(
            step1
                .as_dictionary_column()
                .unwrap()
                .lazy_as_materialized_series()
                .is_none()
        );
        let flat_step2 = expected
            .cast(&DataType::Float64)
            .unwrap()
            .cast(&DataType::String)
            .unwrap();
        assert!(step2.equals_missing(&flat_step2));
        // Used values only: a selective slice does not map the whole dictionary.
        let sliced = a.slice(0, 2);
        let out = Column::from(sliced.clone())
            .cast(&DataType::String)
            .unwrap();
        assert!(out.equals_missing(&flat(&sliced).cast(&DataType::String).unwrap()));
    }

    #[test]
    fn non_byte_dictionary_nested_values() {
        let values = Series::new(
            "v".into(),
            &[
                Series::new("".into(), ["a", "b"]),
                Series::new("".into(), ["c"]),
            ],
        );
        let a = generic_dict_column(values, &[Some(1), None, Some(0), Some(1)]);
        assert!(matches!(a.dtype(), DataType::List(_)));
        a.check_invariants();
        let col = Column::from(a.clone());
        assert_eq!(col.get(0).unwrap().to_string(), "[\"c\"]");
        assert!(col.get(1).unwrap().is_null());
        assert!(col.slice(1, 3).equals_missing(&flat(&a).slice(1, 3)));
        assert!(col.rechunk().equals_missing(&flat(&a)));
    }

    #[test]
    fn clones_share_one_materialization() {
        let a = Column::from(int_dict_column(9));
        let b = a.clone();
        let c = a.clone();
        let first = b.as_materialized_series();
        assert!(
            a.as_dictionary_column()
                .unwrap()
                .lazy_as_materialized_series()
                .is_some()
        );
        // Same buffers: the codes were gathered once for all the copies.
        let ptr = |s: &Series| s.chunks()[0].as_ref() as *const dyn Array as *const u8;
        assert_eq!(ptr(first), ptr(a.as_materialized_series()));
        assert_eq!(ptr(first), ptr(c.as_materialized_series()));
        // Renaming one gives it a cache of its own.
        let mut d = a.clone();
        d.rename("other".into());
        assert_eq!(d.as_materialized_series().name().as_str(), "other");
        assert_eq!(a.as_materialized_series().name().as_str(), "g");
    }

    #[test]
    fn group_by_and_unique_materialize_a_non_byte_dictionary_once() {
        let n = 40;
        let d = int_dict_column(n);
        let coded = Column::from(d.clone());
        let vals = Column::new("x".into(), (0..n as i64).collect::<Vec<_>>());
        let df = DataFrame::new(n, vec![coded.clone(), vals.clone()]).unwrap();
        let plain = DataFrame::new(n, vec![flat(&d).with_name("g".into()), vals]).unwrap();

        let got = df
            .group_by_stable(["g"])
            .unwrap()
            .select(["x"])
            .count()
            .unwrap();
        let want = plain
            .group_by_stable(["g"])
            .unwrap()
            .select(["x"])
            .count()
            .unwrap();
        assert!(got.equals_missing(&want));
        // One gather served the frame, its clone and the original column.
        let cache = coded
            .as_dictionary_column()
            .unwrap()
            .lazy_as_materialized_series();
        assert!(cache.is_some());
        let ptr = |s: &Series| s.chunks()[0].as_ref() as *const dyn Array as *const u8;
        let before = ptr(cache.unwrap());

        let got = df
            .unique_stable(Some(&["g".to_string()]), UniqueKeepStrategy::First, None)
            .unwrap();
        let want = plain
            .unique_stable(Some(&["g".to_string()]), UniqueKeepStrategy::First, None)
            .unwrap();
        assert!(got.equals_missing(&want));
        assert_eq!(
            before,
            ptr(coded
                .as_dictionary_column()
                .unwrap()
                .lazy_as_materialized_series()
                .unwrap())
        );
    }

    #[cfg(feature = "dtype-decimal")]
    #[test]
    fn decimal_dictionary_maps_once_per_value() {
        let values = Int128Chunked::from_vec("v".into(), vec![150, 275, 9_999])
            .into_decimal_unchecked(10, 2)
            .into_series();
        let codes: Vec<Option<u32>> = (0..60).map(|i| (i % 7 != 0).then_some(i % 3)).collect();
        let a = generic_dict_column(values, &codes);
        let col = Column::from(a.clone());
        let expected = flat(&a);
        assert!(matches!(a.dtype(), DataType::Decimal(10, 2)));
        a.check_invariants();
        // cast to String
        let out = col.cast(&DataType::String).unwrap();
        assert_coded(&out, 4);
        assert!(out.equals_missing(&expected.cast(&DataType::String).unwrap()));
        assert!(out.strict_cast(&DataType::String).is_ok());
        // arithmetic with a literal
        let lit = Column::new_scalar(
            "c".into(),
            Scalar::new(DataType::Decimal(10, 2), AnyValue::Decimal(200, 10, 2)),
            1,
        );
        let out = (&col * &lit).unwrap();
        assert_coded(&out, 3);
        assert!(out.equals_missing(&(&expected * &lit).unwrap()));
    }

    #[cfg(feature = "dtype-i128")]
    #[test]
    fn int128_dictionary_slices_and_casts() {
        let values = Series::new("v".into(), [1i128, i128::MAX, -5]);
        let codes: Vec<Option<u32>> = (0..20).map(|i| (i % 6 != 0).then_some(i % 3)).collect();
        let a = generic_dict_column(values, &codes);
        a.check_invariants();
        let col = Column::from(a.clone());
        assert!(col.slice(3, 9).equals_missing(&flat(&a).slice(3, 9)));
        let out = col.cast(&DataType::String).unwrap();
        assert!(out.equals_missing(&flat(&a).cast(&DataType::String).unwrap()));
        assert!(out.as_dictionary_column().is_some());
    }

    #[cfg(feature = "timezones")]
    #[test]
    fn datetime_dictionary_maps_once_per_value() {
        use crate::datatypes::TimeUnit;
        let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
        let dtype = DataType::Datetime(TimeUnit::Microseconds, tz);
        let values = Series::new("v".into(), [0i64, 3_600_000_000 * 5, 3_600_000_000 * 30])
            .cast(&dtype)
            .unwrap();
        let codes: Vec<Option<u32>> = (0..50).map(|i| (i % 7 != 0).then_some(i % 3)).collect();
        let a = generic_dict_column(values, &codes);
        a.check_invariants();
        let col = Column::from(a.clone());
        let expected = flat(&a);
        assert_eq!(col.dtype(), &dtype);
        assert_eq!(
            col.get(1).unwrap().to_string(),
            expected.get(1).unwrap().to_string()
        );
        // strftime as a per-value function
        let fmt = |v: &Series| {
            Ok(v.datetime()
                .unwrap()
                .strftime("%Y-%m-%d %H:%M")
                .unwrap()
                .into_series())
        };
        let out = a.apply_elementwise(fmt).unwrap();
        assert_coded(&out, 4);
        assert!(
            out.as_materialized_series()
                .equals_missing(&fmt(expected.as_materialized_series()).unwrap())
        );
        // Cast to String formats too.
        let out = col.cast(&DataType::String).unwrap();
        assert_coded(&out, 4);
        assert!(out.equals_missing(&expected.cast(&DataType::String).unwrap()));
        // A chain: the dictionary of the mapped values is mapped again, never gathered.
        let shifted = |v: &Series| {
            let ca = v.datetime().unwrap();
            Ok(ca
                .phys
                .clone()
                .apply(|x| x.map(|x| x + 3_600_000_000))
                .into_datetime(TimeUnit::Microseconds, ca.time_zone().clone())
                .into_series())
        };
        let step1 = a.apply_elementwise(shifted).unwrap();
        assert_coded(&step1, 3);
        let step2 = step1
            .as_dictionary_column()
            .unwrap()
            .apply_elementwise(|v| v.cast(&DataType::String))
            .unwrap();
        assert_coded(&step2, 4);
        assert!(
            step1
                .as_dictionary_column()
                .unwrap()
                .lazy_as_materialized_series()
                .is_none()
        );
        let flat_step2 = shifted(expected.as_materialized_series())
            .unwrap()
            .cast(&DataType::String)
            .unwrap();
        assert!(step2.as_materialized_series().equals_missing(&flat_step2));
        // Only the column a flat consumer uses is gathered.
        assert!(
            step1
                .as_dictionary_column()
                .unwrap()
                .lazy_as_materialized_series()
                .is_none()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serializes_as_series() {
        let col = Column::from(dict_column(&["a", "b"], &[Some(1), None, Some(0)]));
        let series = Column::from(col.as_materialized_series().clone());
        let json = serde_json::to_string(&col).unwrap();
        assert_eq!(json, serde_json::to_string(&series).unwrap());
        let back: Column = serde_json::from_str(&json).unwrap();
        assert!(
            back.as_materialized_series()
                .equals_missing(&series.take_materialized_series())
        );
    }

    #[test]
    fn try_append_declines_a_large_dictionary() {
        let mut a = dict_column(&["a", "b"], &[Some(0), Some(1)]);
        let b = dict_column(&["c", "d", "e"], &[Some(0)]);
        assert!(!a.try_append(&b));
        assert_eq!(a.len(), 2);
        assert_eq!(a.dict().len(), 2);
    }

    #[test]
    fn try_append_bounds_dictionary_extension() {
        let a = dict_column(&["a", "b", "c"], &[Some(0), Some(1), Some(2)]);
        let b = dict_column(&["d"], &[Some(0), None, Some(0)]);
        let mut at_limit = a.clone();
        assert!(at_limit.try_append(&b));
        assert_eq!(at_limit.dict().len(), 4);
        let expected = values(&at_limit);

        let mut above_limit = a.clone();
        above_limit.set_sorted_flag(IsSorted::Ascending);
        above_limit.as_materialized_series();
        let smaller_b = b.slice(0, 2);
        assert!(!above_limit.try_append(&smaller_b));
        assert_eq!(above_limit.dict().id(), a.dict().id());
        assert_eq!(values(&above_limit), values(&a));
        assert_eq!(above_limit.is_sorted_flag(), IsSorted::Ascending);
        assert!(above_limit.lazy_as_materialized_series().is_some());

        let mut col = Column::from(above_limit);
        col.append(&Column::from(smaller_b)).unwrap();
        assert!(col.as_dictionary_column().is_none());
        assert_eq!(
            col.str().unwrap().iter().collect::<Vec<_>>(),
            expected[..5]
                .iter()
                .map(Option::as_deref)
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn try_append_reuses_dictionary_above_extension_bound() {
        let mut a = dict_column(&["a", "b", "c"], &[Some(0), Some(1), Some(2)]);
        let same_dict = a.slice(1, 1);
        let overlapping = dict_column(&["c"], &[Some(0)]);
        let id = a.dict().id();
        assert!(a.try_append(&same_dict));
        assert!(a.try_append(&overlapping));
        assert_eq!(a.dict().id(), id);
        assert_eq!(
            a.as_materialized_series()
                .str()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some("a"), Some("b"), Some("c"), Some("b"), Some("c")],
        );
    }

    #[test]
    fn sorted_flag() {
        let mut a = dict_column(&["b", "a"], &[Some(1), Some(1), Some(0)]);
        assert_eq!(a.is_sorted_flag(), IsSorted::Not);
        a.set_sorted_flag(IsSorted::Ascending);
        assert_eq!(a.slice(1, 2).is_sorted_flag(), IsSorted::Ascending);
        assert_eq!(a.rechunk().is_sorted_flag(), IsSorted::Ascending);
        assert_eq!(a.reverse().is_sorted_flag(), IsSorted::Descending);
        assert_eq!(
            a.as_materialized_series().is_sorted_flag(),
            IsSorted::Ascending
        );
        let idx = IdxCa::from_vec(PlSmallStr::EMPTY, vec![2, 0]);
        assert_eq!(
            unsafe { a.take_unchecked(&idx) }.is_sorted_flag(),
            IsSorted::Not
        );
        let b = a.clone();
        assert!(a.try_append(&b));
        assert_eq!(a.is_sorted_flag(), IsSorted::Not);

        let mut col = Column::from(dict_column(&["a"], &[Some(0)]));
        col.set_sorted_flag(IsSorted::Descending);
        assert_eq!(col.is_sorted_flag(), IsSorted::Descending);
        assert!(col.get_flags().contains(StatisticsFlags::IS_SORTED_DSC));
    }

    #[test]
    fn slice_and_reverse_keep_the_materialized_values() {
        let a = dict_column(&["x", "y", "z"], &[Some(0), Some(2), None, Some(1)]);
        assert!(a.lazy_as_materialized_series().is_none());
        assert!(a.slice(1, 2).lazy_as_materialized_series().is_none());
        a.as_materialized_series();
        let sliced = a.slice(1, 3);
        assert!(sliced.lazy_as_materialized_series().is_some());
        sliced.check_invariants();
        let reversed = a.reverse();
        assert!(reversed.lazy_as_materialized_series().is_some());
        reversed.check_invariants();
        // Filtering drops it.
        let mask = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[true, false, true, true]);
        assert!(
            a.filter(&mask)
                .unwrap()
                .lazy_as_materialized_series()
                .is_none()
        );
    }

    #[test]
    fn rename_and_sorted_flag_keep_the_cache_in_sync() {
        let mut a = dict_column(&["x", "y"], &[Some(0), Some(1)]);
        a.as_materialized_series();
        let shared = a.clone();
        a.rename("t".into());
        assert_eq!(a.lazy_as_materialized_series().unwrap().name(), "t");
        assert_eq!(shared.lazy_as_materialized_series().unwrap().name(), "s");
        // Only owner: renamed in place.
        let mut b = dict_column(&["x", "y"], &[Some(0), Some(1)]);
        b.as_materialized_series();
        b.rename("u".into());
        b.set_sorted_flag(IsSorted::Ascending);
        assert_eq!(b.lazy_as_materialized_series().unwrap().name(), "u");
        assert_eq!(
            b.lazy_as_materialized_series().unwrap().is_sorted_flag(),
            IsSorted::Ascending
        );
    }

    #[test]
    fn apply_elementwise_null_results_have_in_bounds_codes() {
        let a = dict_column(&["x", "y"], &[Some(0), Some(1), None, Some(0)]);
        // Null results come last, so the rank of a null row would be one past the values.
        let out = a
            .apply_elementwise(|v| {
                Ok(Series::new(
                    PlSmallStr::EMPTY,
                    &[Some(1i64), Some(2), None][..v.len()],
                ))
            })
            .unwrap();
        let d = out.as_dictionary_column().unwrap();
        assert!(
            d.codes()
                .downcast_iter()
                .all(|c| c.values().iter().all(|c| (*c as usize) < d.dict().len()))
        );
        d.check_invariants();
    }

    fn strings_of(c: &Column) -> Vec<Option<String>> {
        c.as_materialized_series()
            .str()
            .unwrap()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    }

    /// Concatenates clones, which share their codes' buffers: nothing is shifted in place.
    fn concat_of(cols: &[&DictionaryColumn]) -> Result<DictionaryColumn, Vec<DictionaryColumn>> {
        DictionaryColumn::concat(cols.iter().map(|c| (*c).clone()).collect())
    }

    #[test]
    fn concat_shifts_unshared_codes_in_place() {
        let rows =
            |n: usize, k: u32| -> Vec<Option<u32>> { (0..n as u32).map(|i| Some(i % k)).collect() };
        let make = || {
            vec![
                dict_column(&["a", "b"], &rows(40, 2)),
                dict_column(&["c", "d"], &rows(40, 2)),
                dict_column(&["e", "f"], &rows(40, 2)),
            ]
        };
        let expected: Vec<_> = make()
            .iter()
            .flat_map(|c| strings_of(&Column::from(c.clone())))
            .collect();
        let out = DictionaryColumn::concat(make()).unwrap();
        out.check_invariants();
        assert_eq!(out.dict().len(), 6);
        assert_eq!(strings_of(&Column::from(out)), expected);
    }

    #[test]
    fn concat_shifts_codes_by_dictionary_offset() {
        // Rows are many enough for the dictionaries to pay off.
        let rows =
            |n: usize, k: u32| -> Vec<Option<u32>> { (0..n as u32).map(|i| Some(i % k)).collect() };
        let a = dict_column(&["a", "b"], &rows(40, 2));
        let a_tail = a.slice(3, 10);
        let b = dict_column(&["b", "c", "d"], &rows(40, 3));
        let cols = [&a, &a_tail, &b, &a];
        let out = concat_of(&cols).unwrap();
        out.check_invariants();
        // Distinct dictionaries once each, in order (the repeated `a` reuses its offset).
        assert_eq!(out.dict().len(), 5);
        assert_eq!(out.len(), 40 + 10 + 40 + 40);
        assert_eq!(out.codes().chunks().len(), 4);
        let mut expected = Vec::new();
        for c in cols {
            expected.extend(strings_of(&Column::from(c.clone())));
        }
        assert_eq!(strings_of(&Column::from(out.clone())), expected);
        assert_eq!(out.dtype(), &DataType::String);
    }

    /// Merges `cols` as they would arrive, one chunk at a time.
    fn merge_of(cols: &[&DictionaryColumn]) -> Option<DictionaryColumn> {
        let mut merger = DictionaryMerger::default();
        let mut chunks = Vec::new();
        for c in cols {
            let offset = merger.register(c)?;
            chunks.push((*c).clone().into_shifted_codes(offset));
        }
        Some(merger.into_column("c".into(), chunks).unwrap())
    }

    #[test]
    fn merger_shifts_codes_as_chunks_arrive() {
        let rows = |n: usize, k: u32| -> Vec<Option<u32>> {
            (0..n as u32)
                .map(|i| (i % 7 != 6).then_some(i % k))
                .collect()
        };
        let a = dict_column(&["a", "b"], &rows(40, 2));
        let b = dict_column(&["b", "c", "d"], &rows(40, 3));
        let cols = [&a, &b, &a];
        let out = merge_of(&cols).unwrap();
        out.check_invariants();
        assert_eq!(out.dict().len(), 5);
        assert_eq!(out.codes().chunks().len(), 3);
        let expected: Vec<_> = cols
            .iter()
            .flat_map(|c| strings_of(&Column::from((*c).clone())))
            .collect();
        assert_eq!(strings_of(&Column::from(out)), expected);

        // One dictionary, however many chunks, stays that dictionary.
        let only = merge_of(&[&a, &a.slice(2, 5)]).unwrap();
        assert_eq!(only.dict().id(), a.dict().id());

        // Non-byte values.
        let x = int_dict_column(30);
        let y = int_dict_column(17);
        let out = merge_of(&[&x, &y]).unwrap();
        out.check_invariants();
        assert_eq!(out.dict().len(), 6);
        let mut expected = flat(&x).as_materialized_series().clone();
        expected.append(flat(&y).as_materialized_series()).unwrap();
        assert_eq!(
            out.as_materialized_series().clone().with_name("v".into()),
            expected.with_name("v".into())
        );
    }

    #[test]
    fn merger_refuses_sorted_and_other_dtypes() {
        let a = dict_column(&["a", "b"], &[Some(0), Some(1)]);
        let mut sorted = a.clone();
        sorted.set_sorted_flag(IsSorted::Ascending);
        let mut merger = DictionaryMerger::default();
        assert_eq!(merger.register(&a), Some(0));
        assert_eq!(merger.register(&int_dict_column(4)), None);
        assert!(merger.failed());
        assert_eq!(merger.register(&a), None, "failing is for good");
        let mut merger = DictionaryMerger::default();
        assert_eq!(merger.register(&sorted), None);
        assert!(merger.is_empty());
    }

    #[test]
    fn merger_dense_enough() {
        let a = dict_column(&["a", "b"], &[Some(0)]);
        let b = dict_column(&["c", "d"], &[Some(0)]);
        let mut merger = DictionaryMerger::default();
        merger.register(&a);
        assert!(merger.is_dense_enough(0), "one dictionary always pays off");
        merger.register(&b);
        assert!(!merger.is_dense_enough(31));
        assert!(merger.is_dense_enough(32));
    }

    #[test]
    fn concat_same_dictionary_keeps_it_and_null_codes() {
        let a = dict_column(&["x", "y"], &[Some(0), None, Some(1)]);
        let b = a.slice(1, 2);
        let out = concat_of(&[&a, &b]).unwrap();
        assert_eq!(out.dict().id(), a.dict().id());
        assert_eq!(out.null_count(), 2);
        out.check_invariants();
    }

    #[test]
    fn concat_wraps_out_of_range_null_codes() {
        let dict_a = SharedDictionary::new(Utf8ViewArray::from_slice([Some("a")]));
        let dict_b = SharedDictionary::new(Utf8ViewArray::from_slice([Some("b")]));
        // A null slot holding u32::MAX, which the shift must not overflow on.
        let mut raw = vec![0u32; 10];
        raw[1] = u32::MAX;
        let mut validity = polars_arrow::bitmap::MutableBitmap::from_len_set(10);
        validity.set(1, false);
        let array = polars_arrow::array::PrimitiveArray::<u32>::new(
            polars_arrow::datatypes::ArrowDataType::UInt32,
            raw.into(),
            Some(validity.into()),
        );
        let codes = UInt32Chunked::with_chunk(PlSmallStr::EMPTY, array);
        let a = unsafe { DictionaryColumn::new_unchecked("c".into(), dict_a, codes.clone()) };
        let b = unsafe { DictionaryColumn::new_unchecked("c".into(), dict_b, codes) };
        let out = concat_of(&[&a, &b]).unwrap();
        assert_eq!(out.null_count(), 2);
        let values = strings_of(&Column::from(out));
        assert_eq!(values[10], Some("b".to_owned()));
        assert_eq!(values[11], None);
    }

    #[test]
    fn concat_declines_sparse_dictionaries_and_other_dtypes() {
        let a = dict_column(&["a", "b"], &[Some(0), Some(1)]);
        let b = dict_column(&["c", "d"], &[Some(0), Some(1)]);
        // Four values for four rows.
        assert!(concat_of(&[&a, &b]).is_err());
        assert!(concat_of(&[&a, &int_dict_column(3)]).is_err());
        // The Column-level concatenation then materializes, with the same values.
        let out = concat_columns(vec![a.clone().into_column(), b.clone().into_column()]);
        assert!(out.as_dictionary_column().is_none());
        assert_eq!(
            strings_of(&out),
            ["a", "b", "c", "d"].map(|v| Some(v.to_owned()))
        );
    }

    #[test]
    fn concat_columns_non_byte_dictionaries() {
        let a = int_dict_column(64);
        let b = int_dict_column(64);
        let out = concat_columns(vec![a.clone().into_column(), b.clone().into_column()]);
        let d = out.as_dictionary_column().expect("kept as codes");
        d.check_invariants();
        let mut expected = flat(&a);
        expected.append(&flat(&b)).unwrap();
        assert!(out.equals_missing(&expected));
    }

    #[test]
    fn to_series_of_many_chunks_matches_one() {
        let n = 1 << 17;
        let codes: Vec<Option<u32>> = (0..n as u32)
            .map(|i| if i % 7 == 0 { None } else { Some(i % 3) })
            .collect();
        let a = dict_column(&["p", "q", "r"], &codes);
        let b = dict_column(&["p", "q", "r"], &codes);
        let out = concat_of(&[&a, &b]).unwrap();
        assert!(out.codes().chunks().len() > 1);
        let mut expected = a.to_series();
        expected.append(&b.to_series()).unwrap();
        assert!(out.to_series().equals_missing(&expected));
    }

    #[test]
    fn concat_declines_sorted_columns() {
        let mut a = dict_column(&["a", "b"], &[Some(0), Some(0), Some(1), Some(1)]);
        a.set_sorted_flag(IsSorted::Ascending);
        assert!(concat_of(&[&a, &a.slice(0, 2)]).is_err());
        let out = concat_columns(vec![a.clone().into_column(), a.clone().into_column()]);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn merger_stops_holding_dictionaries_that_outgrow_the_rows() {
        let mut merger = DictionaryMerger::default();
        let n = 1 << 20;
        let mut taken = 0;
        for _ in 0..200 {
            let values = Series::new("v".into(), (0..n as i64).collect::<Vec<_>>());
            let c = generic_dict_column(values, &[Some(0), Some(1)]);
            if merger.register(&c).is_some() {
                taken += 1;
            }
        }
        // 2^24 rows of slack allow 2^21 values: a few of the 2^20-value dictionaries.
        assert!(taken <= 3 && merger.failed(), "{taken}");
        assert!(merger.merged_len <= 3 << 20);
    }

    #[test]
    fn gather_chunk_uses_the_dictionary_of_the_chunk() {
        let a = dict_column(&["a", "b"], &[Some(1), None, Some(0)]);
        let b = dict_column(&["x", "y", "z"], &[Some(2), Some(0), None]);
        let mut merger = DictionaryMerger::default();
        let oa = merger.register(&a).unwrap();
        let ob = merger.register(&b).unwrap();
        let ca = a.into_shifted_codes(oa);
        let cb = b.into_shifted_codes(ob);
        let sa = merger.gather_chunk("s".into(), ca).unwrap();
        let sb = merger.gather_chunk("s".into(), cb).unwrap();
        assert_eq!(
            strings_of(&sa.into_column()),
            vec![Some("b".to_owned()), None, Some("a".to_owned())]
        );
        assert_eq!(
            strings_of(&sb.into_column()),
            vec![Some("z".to_owned()), Some("x".to_owned()), None]
        );
    }

    #[test]
    fn try_concat_columns_of_other_dtypes_is_an_error() {
        let s = dict_column(&["a"], &[Some(0)]).into_column();
        let i = int_dict_column(64).into_column();
        assert!(try_concat_columns(vec![s, i]).is_err());
    }
}
