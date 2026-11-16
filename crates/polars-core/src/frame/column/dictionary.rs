use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use polars_arrow::array::{Array, BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::aliases::{InitHashMaps, PlHashMap, PlRandomState};
use polars_utils::pl_str::PlSmallStr;

use super::{Column, DataType, IntoColumn, Series};
use crate::prelude::*;
use crate::series::IsSorted;

static NEXT_DICTIONARY_ID: AtomicU64 = AtomicU64::new(0);

/// How many hashers' hashes a dictionary keeps.
const MAX_CACHED_HASHERS: usize = 4;

static STRING_DTYPE: DataType = DataType::String;
static BINARY_DTYPE: DataType = DataType::Binary;

/// The values of a dictionary that [`DictionaryColumn`]s index into, such as the dictionary page
/// of a parquet column chunk. They are String values (valid UTF-8) or Binary values; the codes, the
/// hashes and the lookup table work on their bytes either way.
///
/// The values need not be unique: equal codes mean equal values, but different codes may also.
pub struct SharedDictionary {
    /// Process-unique id; caches in operators are keyed by it.
    id: u64,
    /// The dictionary values; materialized columns share its data buffers.
    values: BinaryViewArray,
    /// Whether the values are String (valid UTF-8) rather than Binary.
    is_string: bool,
    /// The hash of each value under recently used hashers, keyed by [`hasher_fingerprint`].
    hashes: Mutex<Vec<CachedHashes>>,
    /// The first code of each value, built on first use (see [`Self::code_of`]).
    lookup: OnceLock<CodeLookup>,
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

    fn with_values(values: BinaryViewArray, is_string: bool) -> Arc<Self> {
        assert!(
            values.validity().is_none_or(|v| v.unset_bits() == 0),
            "dictionary values must not be null"
        );
        let values = values.with_validity(None);
        Arc::new(Self {
            id: NEXT_DICTIONARY_ID.fetch_add(1, Ordering::Relaxed),
            values,
            is_string,
            hashes: Mutex::new(Vec::new()),
            lookup: OnceLock::new(),
        })
    }

    fn lookup(&self) -> &CodeLookup {
        self.lookup.get_or_init(|| {
            let random_state = PlRandomState::default();
            let mut codes = hashbrown::HashTable::with_capacity(self.len());
            for (c, v) in self.values.values_iter().enumerate() {
                let h = random_state.hash_one(v);
                // SAFETY: the codes in the table are in bounds.
                let value = |k: &u32| unsafe { self.values.value_unchecked(*k as usize) };
                if codes.find(h, |k| value(k) == v).is_none() {
                    codes.insert_unique(h, c as u32, |k| random_state.hash_one(value(k)));
                }
            }
            CodeLookup {
                random_state,
                codes,
            }
        })
    }

    /// The first code of `value`, if the dictionary holds it. The lookup table is built on the
    /// first call and kept with the dictionary.
    pub fn code_of(&self, value: &[u8]) -> Option<u32> {
        let lookup = self.lookup();
        lookup
            .codes
            .find(lookup.random_state.hash_one(value), |k| {
                // SAFETY: the codes in the table are in bounds.
                unsafe { self.values.value_unchecked(*k as usize) == value }
            })
            .copied()
    }

    /// A new dictionary of these values followed by `added`, none of which this one holds. It
    /// shares these values' buffers, and extends the lookup table and the cached hashes instead of
    /// computing them again.
    fn extended(&self, added: &[&[u8]]) -> Arc<Self> {
        debug_assert!(
            !self.is_string || added.iter().all(|v| std::str::from_utf8(v).is_ok()),
            "added values of a String dictionary must be valid UTF-8"
        );
        let old_len = self.len();
        let mut values = MutableBinaryViewArray::<[u8]>::with_capacity(old_len + added.len());
        for buffer in self.values.data_buffers().iter() {
            values.push_buffer(buffer.clone());
        }
        // SAFETY: the views index the buffers just pushed, in the same order.
        unsafe {
            values.views_mut().extend(self.values.views().iter());
            values.set_total_bytes_len(self.values.total_bytes_len());
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

        Arc::new(Self {
            id: NEXT_DICTIONARY_ID.fetch_add(1, Ordering::Relaxed),
            values,
            is_string: self.is_string,
            hashes: Mutex::new(hashes),
            lookup,
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// The values as bytes.
    pub fn values(&self) -> &BinaryViewArray {
        &self.values
    }

    /// Whether the values are String rather than Binary.
    pub fn is_string(&self) -> bool {
        self.is_string
    }

    pub fn dtype(&self) -> &DataType {
        if self.is_string {
            &STRING_DTYPE
        } else {
            &BINARY_DTYPE
        }
    }

    /// The values as a String or Binary [`Series`], sharing the buffers.
    pub fn values_series(&self, name: PlSmallStr) -> Series {
        if self.is_string {
            // SAFETY: a String dictionary holds valid UTF-8.
            let values = unsafe { self.values.to_utf8view_unchecked() };
            StringChunked::with_chunk(name, values).into_series()
        } else {
            BinaryChunked::with_chunk(name, self.values.clone()).into_series()
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The hashes [`Self::hashes`] gives, if they were already computed.
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
    /// `random_state`, computed once per hasher.
    pub fn hashes(&self, random_state: &PlRandomState) -> Buffer<u64> {
        if let Some(hashes) = self.cached_hashes(random_state) {
            return hashes;
        }
        let fingerprint = hasher_fingerprint(random_state);

        let buffers = self.values.data_buffers();
        let hashes: Buffer<u64> = self
            .values
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
        hashes
    }
}

/// A String or Binary [`Column`] held as codes into a [`SharedDictionary`]: row `i`, when valid, is
/// `dict.values()[codes[i]]`.
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

    /// A lazily materialized [`Series`] variant of this [`DictionaryColumn`].
    materialized: OnceLock<Series>,
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
            materialized: OnceLock::new(),
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

    pub fn rename(&mut self, name: PlSmallStr) {
        if let Some(s) = self.materialized.get_mut() {
            s.rename(name.clone());
        }
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
            materialized: OnceLock::new(),
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
        if let Some(s) = self.materialized.get_mut() {
            s.set_sorted_flag(sorted);
        }
        self.sorted = sorted;
    }

    /// Gathers the values into a String or Binary [`Series`].
    pub fn to_series(&self) -> Series {
        let values = self.dict.values_series(self.name.clone());
        #[cfg(feature = "bigidx")]
        let idx = self.codes.cast(&IDX_DTYPE).unwrap();
        #[cfg(feature = "bigidx")]
        let idx = idx.idx().unwrap();
        #[cfg(not(feature = "bigidx"))]
        let idx = &self.codes;
        // SAFETY: the codes are in bounds (invariant).
        let mut s = unsafe { values.take_unchecked(idx) };
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
        match self.materialized.into_inner() {
            Some(s) => s,
            None => Self {
                materialized: OnceLock::new(),
                ..self
            }
            .to_series(),
        }
    }

    pub fn slice(&self, offset: i64, length: usize) -> Self {
        self.with_order_kept(self.codes.slice(offset, length))
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
        out
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
            if self.dict.is_string != other.dict.is_string {
                return false;
            }
            let max_len = (self.len() + other.len()).min(u32::MAX as usize);
            if self.dict.len() > max_len {
                return false;
            }
            let mut added = Vec::new();
            let mut added_code_of = PlHashMap::<&[u8], u32>::new();
            let mut remap = Vec::with_capacity(other.dict.len());
            for v in other.dict.values.values_iter() {
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
        self.materialized = OnceLock::new();
        true
    }

    /// Estimated size of the column, see [`Series::estimated_size`].
    ///
    /// The dictionary is shared by every slice of a column, so each counts the part of it its rows
    /// would take materialized, at most all of it; counting all of it for each slice would
    /// overstate the memory of the morsels of a row group many times over.
    pub fn estimated_size(&self) -> usize {
        let dict_size = polars_arrow::compute::aggregate::estimated_bytes_size(&self.dict.values);
        let dict_part = if self.len() >= self.dict.len() {
            dict_size
        } else {
            (dict_size as u128 * self.len() as u128 / self.dict.len() as u128) as usize
        };
        self.codes
            .chunks()
            .iter()
            .map(|c| polars_arrow::compute::aggregate::estimated_bytes_size(c.as_ref()))
            .sum::<usize>()
            + dict_part
            + self.materialized.get().map_or(0, |s| s.estimated_size())
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
    /// a null if a row is null, and spreads the results to the rows by code. A String or Binary
    /// result stays dictionary-coded. `f` gets a String or Binary [`Series`] of this column's name and returns one value
    /// per input value.
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

/// Whether dictionary-coded String and Binary columns are produced and used, `POLARS_DICT_CODES=0` turns them
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
        a.dict().hashes(&rs);
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
        let fresh = SharedDictionary::new_binary(dict.values().clone()).hashes(&rs);
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
}
