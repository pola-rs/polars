use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use hashbrown::hash_table::{Entry, HashTable};
use polars_arrow::array::{
    BinaryViewArray, ListArray, MutableBinaryViewArray, MutableBooleanArray, MutablePrimitiveArray,
    View,
};
use polars_arrow::offset::Offsets;
use polars_arrow::pushable::Pushable;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::UnitVec;
use polars_utils::aliases::PlFixedStateQuality;
use polars_utils::total_ord::{TotalEq, TotalHash, TotalOrdWrap};

use super::*;

pub fn supports_unique_reduction(dtype: &DataType) -> bool {
    dtype.is_bool()
        || dtype.is_primitive_numeric()
        || dtype.is_temporal()
        || dtype.is_decimal()
        || dtype.is_categorical()
        || dtype.is_enum()
        || dtype.is_string()
        || dtype.is_binary()
}

/// The unique values of each group, as a list per group.
pub fn new_unique_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction::<false>(dtype)
}

/// The number of unique values of each group, null included.
pub fn new_n_unique_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction::<true>(dtype)
}

fn new_reduction<const COUNT: bool>(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    assert!(supports_unique_reduction(&dtype));
    match dtype {
        Boolean => Box::new(VGR::new(dtype, BoolUniqueReducer::<COUNT>)),
        String | Binary => Box::new(VGR::new(dtype, BinaryUniqueReducer::<COUNT>)),
        _ => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumUniqueReducer::<$T, COUNT>(PhantomData)))
            })
        },
    }
}

/// Groups with at most this many unique values are deduplicated by a linear scan.
const SMALL_LIMIT: usize = 16;

fn hash_one<T: Hash + ?Sized>(x: &T) -> u64 {
    PlFixedStateQuality::default().hash_one(x)
}

/// The distinct values of a group, as keys that may refer to memory outside of it.
#[derive(Clone)]
struct UniqueValues<K> {
    small: UnitVec<K>,
    large: Option<Box<HashTable<K>>>,
    has_null: bool,
}

impl<K: Copy> UniqueValues<K> {
    fn new() -> Self {
        Self {
            small: UnitVec::new(),
            large: None,
            has_null: false,
        }
    }

    /// Inserts the key `make` returns unless `eq` matches a key, and returns whether it did.
    /// `hash` is the hash of the new value and `rehash` that of a key.
    #[inline(always)]
    fn insert(
        &mut self,
        hash: impl FnOnce() -> u64,
        eq: impl Fn(&K) -> bool,
        rehash: impl Fn(&K) -> u64,
        make: impl FnOnce() -> K,
    ) -> bool {
        if let Some(table) = &mut self.large {
            match table.entry(hash(), eq, rehash) {
                Entry::Occupied(_) => false,
                Entry::Vacant(entry) => {
                    entry.insert(make());
                    true
                },
            }
        } else if self.small.iter().any(eq) {
            false
        } else if self.small.len() < SMALL_LIMIT {
            self.small.push(make());
            true
        } else {
            let mut table = HashTable::with_capacity(2 * SMALL_LIMIT);
            for k in std::mem::take(&mut self.small) {
                table.insert_unique(rehash(&k), k, &rehash);
            }
            table.insert_unique(hash(), make(), &rehash);
            self.large = Some(Box::new(table));
            true
        }
    }

    /// Makes room for the keys of `other` if both are tables.
    fn reserve_for(&mut self, other: &Self, rehash: impl Fn(&K) -> u64) {
        if let (Some(table), Some(other)) = (&mut self.large, &other.large) {
            table.reserve(other.len(), rehash);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &K> {
        self.small
            .iter()
            .chain(self.large.iter().flat_map(|t| t.iter()))
    }

    fn num_values(&self) -> usize {
        self.small.len() + self.large.as_ref().map_or(0, |t| t.len())
    }

    fn len(&self) -> usize {
        self.num_values() + self.has_null as usize
    }

    /// The heap size of the keys.
    fn heap_bytes(&self) -> usize {
        let small = if self.small.is_inline() {
            0
        } else {
            self.small.capacity() * size_of::<K>()
        };
        let large = self.large.as_ref().map_or(0, |t| {
            size_of::<HashTable<K>>() + t.capacity() * (size_of::<K>() + 1)
        });
        small + large
    }

    fn into_keys(self) -> impl Iterator<Item = K> {
        self.small
            .into_iter()
            .chain(self.large.into_iter().flat_map(|t| t.into_iter()))
    }
}

impl<T: TotalHash + TotalEq + Copy> UniqueValues<TotalOrdWrap<T>> {
    #[inline(always)]
    fn insert_value(&mut self, x: T) {
        let k = TotalOrdWrap(x);
        self.insert(|| hash_one(&k), |s| *s == k, hash_one, || k);
    }

    fn combine(&mut self, other: &Self) {
        self.has_null |= other.has_null;
        self.reserve_for(other, hash_one);
        for k in other.iter() {
            self.insert_value(k.0);
        }
    }
}

/// `heap_bytes` gives the heap memory of one group.
fn state_size<V>(
    v: &[V],
    len: impl Fn(&V) -> usize,
    heap_bytes: impl Fn(&V) -> usize,
) -> StateSize {
    let mut size = StateSize {
        num_values: 0,
        bytes: size_of_val(v),
    };
    for u in v {
        size.num_values += len(u);
        size.bytes += heap_bytes(u);
    }
    size
}

fn finish_counts<V>(v: &[V], len: impl Fn(&V) -> usize) -> Series {
    let counts = v.iter().map(|u| len(u) as IdxSize).collect();
    IdxCa::from_vec(PlSmallStr::EMPTY, counts).into_series()
}

struct NumUniqueReducer<T, const COUNT: bool>(PhantomData<T>);

impl<T, const COUNT: bool> Clone for NumUniqueReducer<T, COUNT> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: PolarsNumericType, const COUNT: bool> Reducer for NumUniqueReducer<T, COUNT> {
    type Dtype = T;
    type Value = UniqueValues<TotalOrdWrap<T::Native>>;

    fn init(&self) -> Self::Value {
        UniqueValues::new()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.combine(b);
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        match b {
            Some(x) => a.insert_value(x),
            None => a.has_null = true,
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.has_null |= ca.null_count() > 0;
        for x in ca
            .downcast_iter()
            .flat_map(|arr| arr.non_null_values_iter())
        {
            v.insert_value(x);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        if COUNT {
            return Ok(finish_counts(&v, UniqueValues::len));
        }

        let total_len = v.iter().map(|u| u.len()).sum();
        let mut out = MutablePrimitiveArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for u in v.into_iter() {
            offsets.try_push(u.len()).unwrap();
            let has_null = u.has_null;
            out.extend_values(u.into_keys().map(|x| x.0));
            if has_null {
                out.push_null();
            }
        }

        let values = out.freeze();
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        let phys_list_dtype = DataType::List(Box::new(dtype.to_physical()));
        let arr = ListArray::new(
            phys_list_dtype.to_arrow(CompatLevel::newest()),
            offsets.freeze(),
            values.boxed(),
            None,
        );
        let ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        let s = ca.into_series();
        unsafe { s.from_physical_unchecked(&list_dtype) }
    }

    fn state_size(&self, v: &[Self::Value]) -> Option<StateSize> {
        Some(state_size(v, UniqueValues::len, UniqueValues::heap_bytes))
    }
}

/// A value of a group. Values that do not fit inline are stored in the bytes of the group,
/// at offset `buffer_idx << 32 | offset`.
#[derive(Clone, Copy)]
struct Key {
    hash: u64,
    view: View,
}

/// The distinct values of a group, with the values that do not fit inline in one buffer.
#[derive(Clone)]
struct BinaryUniqueValues {
    values: UniqueValues<Key>,
    bytes: Vec<u8>,
}

impl BinaryUniqueValues {
    fn new() -> Self {
        Self {
            values: UniqueValues::new(),
            bytes: Vec::new(),
        }
    }

    #[inline(always)]
    fn get<'a>(bytes: &'a [u8], k: &'a Key) -> &'a [u8] {
        if k.view.is_inline() {
            // SAFETY: the view is inline.
            unsafe { k.view.get_inlined_slice_unchecked() }
        } else {
            let start = ((k.view.buffer_idx as u64) << 32 | k.view.offset as u64) as usize;
            &bytes[start..start + k.view.length as usize]
        }
    }

    #[inline(always)]
    fn insert_value(&mut self, x: &[u8]) {
        if x.len() <= View::MAX_INLINE_SIZE as usize {
            // SAFETY: the length was checked.
            let view = unsafe { View::new_inline_unchecked(x) };
            self.insert_inline(view, hash_one(&view.as_u128()));
        } else {
            self.insert_external(x, hash_one(x));
        }
    }

    #[inline(always)]
    fn insert_inline(&mut self, view: View, hash: u64) {
        self.values.insert(
            || hash,
            |k| k.view == view,
            |k| k.hash,
            || Key { hash, view },
        );
    }

    #[inline(always)]
    fn insert_external(&mut self, x: &[u8], hash: u64) {
        let bytes = &self.bytes;
        let start = bytes.len() as u64;
        let inserted = self.values.insert(
            || hash,
            |k| k.hash == hash && Self::get(bytes, k) == x,
            |k| k.hash,
            || Key {
                hash,
                // SAFETY: values that fit inline are inserted by `insert_inline`.
                view: unsafe {
                    View::new_noninline_unchecked(x, (start >> 32) as u32, start as u32)
                },
            },
        );
        if inserted {
            self.bytes.extend_from_slice(x);
        }
    }

    fn combine(&mut self, other: &Self) {
        self.values.has_null |= other.values.has_null;
        self.values.reserve_for(&other.values, |k| k.hash);
        for k in other.values.iter() {
            if k.view.is_inline() {
                self.insert_inline(k.view, k.hash);
            } else {
                self.insert_external(Self::get(&other.bytes, k), k.hash);
            }
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn heap_bytes(&self) -> usize {
        self.values.heap_bytes() + self.bytes.capacity()
    }
}

#[derive(Clone)]
struct BinaryUniqueReducer<const COUNT: bool>;

impl<const COUNT: bool> Reducer for BinaryUniqueReducer<COUNT> {
    type Dtype = BinaryType;
    type Value = BinaryUniqueValues;

    fn init(&self) -> Self::Value {
        BinaryUniqueValues::new()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.combine(b);
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, _seq_id: u64) {
        match b {
            Some(x) => a.insert_value(x),
            None => a.values.has_null = true,
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.values.has_null |= ca.null_count() > 0;
        for x in ca
            .downcast_iter()
            .flat_map(|arr| arr.non_null_values_iter())
        {
            v.insert_value(x);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        if COUNT {
            return Ok(finish_counts(&v, BinaryUniqueValues::len));
        }

        let total_len = v.iter().map(|u| u.len()).sum();
        let mut out = MutableBinaryViewArray::<[u8]>::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for u in v {
            offsets.try_push(u.len()).unwrap();
            for s in u.values.iter() {
                out.push_value(BinaryUniqueValues::get(&u.bytes, s));
            }
            if u.values.has_null {
                out.push_null();
            }
        }

        let values: BinaryViewArray = out.freeze();
        let values = if dtype.is_string() {
            // SAFETY: the values came from a String column.
            unsafe { values.to_utf8view_unchecked() }.boxed()
        } else {
            values.boxed()
        };
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        let arr = ListArray::new(
            list_dtype.to_arrow(CompatLevel::newest()),
            offsets.freeze(),
            values,
            None,
        );
        Ok(ListChunked::with_chunk(PlSmallStr::EMPTY, arr).into_series())
    }

    fn state_size(&self, v: &[Self::Value]) -> Option<StateSize> {
        Some(state_size(
            v,
            BinaryUniqueValues::len,
            BinaryUniqueValues::heap_bytes,
        ))
    }
}

#[derive(Clone, Copy, Default)]
struct BoolUniqueValues {
    has_true: bool,
    has_false: bool,
    has_null: bool,
}

impl BoolUniqueValues {
    fn len(&self) -> usize {
        self.has_true as usize + self.has_false as usize + self.has_null as usize
    }
}

#[derive(Clone)]
struct BoolUniqueReducer<const COUNT: bool>;

impl<const COUNT: bool> Reducer for BoolUniqueReducer<COUNT> {
    type Dtype = BooleanType;
    type Value = BoolUniqueValues;

    fn init(&self) -> Self::Value {
        BoolUniqueValues::default()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.has_true |= b.has_true;
        a.has_false |= b.has_false;
        a.has_null |= b.has_null;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<bool>, _seq_id: u64) {
        match b {
            Some(true) => a.has_true = true,
            Some(false) => a.has_false = true,
            None => a.has_null = true,
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.has_true |= ca.num_trues() > 0;
        v.has_false |= ca.num_falses() > 0;
        v.has_null |= ca.null_count() > 0;
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        if COUNT {
            return Ok(finish_counts(&v, BoolUniqueValues::len));
        }

        let total_len = v.iter().map(|u| u.len()).sum();
        let mut out = MutableBooleanArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for u in v.into_iter() {
            offsets.try_push(u.len()).unwrap();
            if u.has_true {
                out.push(Some(true));
            }
            if u.has_false {
                out.push(Some(false));
            }
            if u.has_null {
                out.push(None);
            }
        }

        let values = out.freeze();
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        let arr = ListArray::new(
            list_dtype.to_arrow(CompatLevel::newest()),
            offsets.freeze(),
            values.boxed(),
            None,
        );
        let ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        Ok(ca.into_series())
    }
}
