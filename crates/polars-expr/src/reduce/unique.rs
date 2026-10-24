#![allow(unsafe_op_in_unsafe_fn)]
use std::borrow::Borrow;
use std::hash::Hash;
use std::marker::PhantomData;

use hashbrown::Equivalent;
use polars_arrow::array::{
    BinaryViewArray, ListArray, MutableBinaryViewArray, MutableBooleanArray, MutablePrimitiveArray,
};
use polars_arrow::offset::Offsets;
use polars_arrow::pushable::Pushable;
use polars_core::with_match_physical_numeric_polars_type;
use polars_plan::prelude::{AExpr, IRFunctionExpr};
use polars_utils::UnitVec;
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::{Arena, Node};
use polars_utils::total_ord::TotalOrdWrap;

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

/// Returns `x` if `node` is `x.unique()` without `maintain_order`.
pub fn unordered_unique_input(node: Node, expr_arena: &Arena<AExpr>) -> Option<Node> {
    match expr_arena.get(node) {
        AExpr::Function {
            input,
            function: IRFunctionExpr::Unique(false),
            ..
        } => Some(input[0].node()),
        _ => None,
    }
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

#[derive(Clone)]
struct UniqueValues<K> {
    small: UnitVec<K>,
    large: Option<Box<PlHashSet<K>>>,
    has_null: bool,
}

impl<K: Hash + Eq + Clone> UniqueValues<K> {
    fn new() -> Self {
        Self {
            small: UnitVec::new(),
            large: None,
            has_null: false,
        }
    }

    #[inline(always)]
    fn insert<Q>(&mut self, value: &Q, to_owned: impl FnOnce(&Q) -> K)
    where
        Q: Hash + Equivalent<K> + PartialEq + ?Sized,
        K: Borrow<Q>,
    {
        if let Some(set) = &mut self.large {
            set.get_or_insert_with(value, to_owned);
        } else if !self.small.iter().any(|k| k.borrow() == value) {
            if self.small.len() < SMALL_LIMIT {
                self.small.push(to_owned(value));
            } else {
                let mut set = PlHashSet::with_capacity(2 * SMALL_LIMIT);
                set.extend(std::mem::take(&mut self.small));
                set.insert(to_owned(value));
                self.large = Some(Box::new(set));
            }
        }
    }

    fn combine(&mut self, other: &Self) {
        self.has_null |= other.has_null;
        for k in other.iter() {
            self.insert(k, K::clone);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &K> {
        self.small
            .iter()
            .chain(self.large.iter().flat_map(|s| s.iter()))
    }

    fn num_values(&self) -> usize {
        self.small.len() + self.large.as_ref().map_or(0, |s| s.len())
    }

    fn len(&self) -> usize {
        self.num_values() + self.has_null as usize
    }

    /// The heap size of the containers, without the heap memory of the values themselves.
    fn container_bytes(&self) -> usize {
        let small = if self.small.is_inline() {
            0
        } else {
            self.small.capacity() * size_of::<K>()
        };
        let large = self.large.as_ref().map_or(0, |s| {
            size_of::<PlHashSet<K>>() + s.capacity() * (size_of::<K>() + 1)
        });
        small + large
    }

    fn into_values(self) -> impl Iterator<Item = K> {
        self.small
            .into_iter()
            .chain(self.large.into_iter().flat_map(|s| s.into_iter()))
    }
}

/// `value_bytes` gives the heap memory of the values of one group.
fn state_size<K: Hash + Eq + Clone>(
    v: &[UniqueValues<K>],
    value_bytes: impl Fn(&UniqueValues<K>) -> usize,
) -> StateSize {
    let mut size = StateSize {
        num_values: 0,
        bytes: size_of_val(v),
    };
    for u in v {
        size.num_values += u.len();
        size.bytes += u.container_bytes() + value_bytes(u);
    }
    size
}

fn finish_counts<K: Hash + Eq + Clone>(v: Vec<UniqueValues<K>>) -> Series {
    let counts = v.iter().map(|u| u.len() as IdxSize).collect();
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
            Some(x) => a.insert(&TotalOrdWrap(x), |x| *x),
            None => a.has_null = true,
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.has_null |= ca.null_count() > 0;
        for x in ca
            .downcast_iter()
            .flat_map(|arr| arr.non_null_values_iter())
        {
            v.insert(&TotalOrdWrap(x), |x| *x);
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
            return Ok(finish_counts(v));
        }

        let total_len = v.iter().map(|u| u.len()).sum();
        let mut out = MutablePrimitiveArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for u in v.into_iter() {
            offsets.try_push(u.len()).unwrap();
            let has_null = u.has_null;
            out.extend_values(u.into_values().map(|x| x.0));
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
        Some(state_size(v, |_| 0))
    }
}

#[derive(Clone)]
struct BinaryUniqueReducer<const COUNT: bool>;

impl<const COUNT: bool> Reducer for BinaryUniqueReducer<COUNT> {
    type Dtype = BinaryType;
    type Value = UniqueValues<Vec<u8>>;

    fn init(&self) -> Self::Value {
        UniqueValues::new()
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
            Some(x) => a.insert(x, <[u8]>::to_vec),
            None => a.has_null = true,
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.has_null |= ca.null_count() > 0;
        for x in ca
            .downcast_iter()
            .flat_map(|arr| arr.non_null_values_iter())
        {
            v.insert(x, <[u8]>::to_vec);
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
            return Ok(finish_counts(v));
        }

        let total_len = v.iter().map(|u| u.len()).sum();
        let mut out = MutableBinaryViewArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for u in v.into_iter() {
            offsets.try_push(u.len()).unwrap();
            let has_null = u.has_null;
            out.extend_values(u.into_values());
            if has_null {
                out.push_null();
            }
        }

        let values: BinaryViewArray = out.freeze();
        let arrow_dtype = ArrowDataType::LargeList(Box::new(ArrowField::new(
            PlSmallStr::EMPTY,
            ArrowDataType::BinaryView,
            true,
        )));
        let arr = ListArray::new(arrow_dtype, offsets.freeze(), values.boxed(), None);
        let ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        ca.into_series().cast(&list_dtype)
    }

    fn state_size(&self, v: &[Self::Value]) -> Option<StateSize> {
        Some(state_size(v, |u| u.iter().map(Vec::capacity).sum()))
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
            let counts = v.iter().map(|u| u.len() as IdxSize).collect();
            return Ok(IdxCa::from_vec(PlSmallStr::EMPTY, counts).into_series());
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
