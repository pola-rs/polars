use std::cmp::Ordering;

use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
#[cfg(feature = "dtype-f16")]
use polars_core::datatypes::Float16Type;
#[cfg(feature = "dtype-i128")]
use polars_core::datatypes::Int128Type;
#[cfg(feature = "dtype-u128")]
use polars_core::datatypes::UInt128Type;
use polars_core::datatypes::{
    DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    PolarsNumericType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use polars_core::series::Series;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::IdxSize;
use polars_utils::total_ord::TotalOrd;

/// The split tree of one key type.
trait KeyTree: Send + Sync {
    fn gen_idxs_per_bucket(&self, keys: &Series, idxs_per_bucket: &mut [Vec<IdxSize>]);
}

/// Whether each split key equals the one before it.
fn equal_to_prev<K>(sorted: &[K], cmp: impl Fn(&K, &K) -> Ordering) -> Vec<bool> {
    (0..sorted.len())
        .map(|j| j > 0 && cmp(&sorted[j - 1], &sorted[j]).is_eq())
        .collect()
}

/// A branchless search tree over the `b - 1` split keys of `b` buckets.
///
/// The split keys are stored in Eytzinger (BFS) layout: tree node `i`
/// (`1 <= i < b`) lives at `keys[i - 1]`. A row passes a split key when it
/// compares greater than or equal to it, or strictly greater when the split key
/// equals its predecessor, and its bucket is the number of split keys it
/// passes. Without the strict comparison the bucket between two equal split
/// keys would be empty and those rows would share a bucket with larger keys.
struct SplitTree<K> {
    keys: Vec<K>,
    /// Minimum comparison result to pass a split key: 0 for `>=`, 1 for `>`.
    min_cmp: Vec<i8>,
    b: usize,
    depth: u32,
    descending: bool,
}

impl<K: Clone> SplitTree<K> {
    /// Builds the tree from the `b - 1` split keys in bucket order, with `b` a
    /// power of two.
    fn new(sorted: &[K], equal_to_prev: &[bool], descending: bool) -> Self {
        let b = sorted.len() + 1;
        assert!(b.is_power_of_two());

        // The position of tree node `i` in the in-order traversal.
        let sorted_idx = |i: usize| {
            let depth = i.ilog2();
            let pos = i - (1 << depth);
            (2 * pos + 1) * (b >> (depth + 1)) - 1
        };
        let keys = (1..b).map(|i| sorted[sorted_idx(i)].clone()).collect();
        let min_cmp = (1..b).map(|i| equal_to_prev[sorted_idx(i)] as i8).collect();

        Self {
            keys,
            min_cmp,
            b,
            depth: b.trailing_zeros(),
            descending,
        }
    }
}

impl<K> SplitTree<K> {
    #[inline(always)]
    fn classify<const DESCENDING: bool, Q: ?Sized>(
        &self,
        x: &Q,
        cmp: impl Fn(&Q, &K) -> Ordering,
    ) -> usize {
        let mut i = 1;
        for _ in 0..self.depth {
            // SAFETY: i < b throughout the descent, so i - 1 is in bounds.
            unsafe {
                let c = cmp(x, self.keys.get_unchecked(i - 1));
                let c = if DESCENDING { c.reverse() } else { c };
                let go_right = (c as i8) >= *self.min_cmp.get_unchecked(i - 1);
                i = 2 * i + go_right as usize;
            }
        }
        i - self.b
    }
}

/// Rows classified together, so that the tree descents of different rows
/// overlap.
const BATCH: usize = 16;

/// Appends the index of every row to the list of its bucket id.
fn fill_idxs_from_ids(ids: &[u32], idxs_per_bucket: &mut [Vec<IdxSize>]) {
    let mut counts = vec![0usize; idxs_per_bucket.len()];
    for &id in ids {
        counts[id as usize] += 1;
    }
    for (idxs, &count) in idxs_per_bucket.iter_mut().zip(&counts) {
        idxs.reserve(count);
    }
    for (i, &id) in ids.iter().enumerate() {
        idxs_per_bucket[id as usize].push(i as IdxSize);
    }
}

/// Whether each of the `b + 1` buckets has to be sorted. The null bucket and a
/// bucket bounded by two equal split keys hold a single key value.
fn needs_sort(equal_to_prev: &[bool]) -> Vec<bool> {
    let b = equal_to_prev.len() + 1;
    let mut needs_sort = vec![true; b + 1];
    needs_sort[b] = false;
    for j in 1..b.saturating_sub(1) {
        needs_sort[j] = !equal_to_prev[j];
    }
    needs_sort
}

fn gen_idxs<'a, K, Q: ?Sized + 'a>(
    tree: &SplitTree<K>,
    values: impl Iterator<Item = &'a Q>,
    validity: Option<&Bitmap>,
    cmp: impl Fn(&Q, &K) -> Ordering + Copy,
    idxs_per_bucket: &mut [Vec<IdxSize>],
) {
    if tree.descending {
        gen_idxs_impl::<K, Q, true>(tree, values, validity, cmp, idxs_per_bucket)
    } else {
        gen_idxs_impl::<K, Q, false>(tree, values, validity, cmp, idxs_per_bucket)
    }
}

fn gen_idxs_impl<'a, K, Q: ?Sized + 'a, const DESCENDING: bool>(
    tree: &SplitTree<K>,
    values: impl Iterator<Item = &'a Q>,
    validity: Option<&Bitmap>,
    cmp: impl Fn(&Q, &K) -> Ordering + Copy,
    idxs_per_bucket: &mut [Vec<IdxSize>],
) {
    let null_bucket = tree.b;
    match validity.filter(|v| v.unset_bits() > 0) {
        None => {
            for (i, v) in values.enumerate() {
                let bucket = tree.classify::<DESCENDING, Q>(v, cmp);
                // SAFETY: the bucket index is in 0..num_buckets.
                unsafe { idxs_per_bucket.get_unchecked_mut(bucket).push(i as IdxSize) };
            }
        },
        Some(validity) => {
            for (i, (v, is_valid)) in values.zip(validity.iter()).enumerate() {
                let bucket = if is_valid {
                    tree.classify::<DESCENDING, Q>(v, cmp)
                } else {
                    null_bucket
                };
                // SAFETY: the bucket index is in 0..=num_buckets.
                unsafe { idxs_per_bucket.get_unchecked_mut(bucket).push(i as IdxSize) };
            }
        },
    }
}

/// An order-preserving map to `u64`, consistent with `TotalOrd`.
trait ToOrdU64: Copy {
    fn to_ord_u64(self) -> u64;
}

macro_rules! impl_ord_unsigned {
    ($($t:ty),*) => {$(impl ToOrdU64 for $t {
        #[inline(always)]
        fn to_ord_u64(self) -> u64 { self as u64 }
    })*};
}
macro_rules! impl_ord_signed {
    ($($t:ty),*) => {$(impl ToOrdU64 for $t {
        #[inline(always)]
        fn to_ord_u64(self) -> u64 { (self as i64 as u64) ^ (1 << 63) }
    })*};
}
impl_ord_unsigned!(u8, u16, u32, u64);
impl_ord_signed!(i8, i16, i32, i64);

impl ToOrdU64 for f64 {
    #[inline(always)]
    fn to_ord_u64(self) -> u64 {
        let bits = polars_utils::total_ord::canonical_f64(self).to_bits();
        if bits >> 63 == 1 {
            !bits
        } else {
            bits | (1 << 63)
        }
    }
}

impl ToOrdU64 for f32 {
    #[inline(always)]
    fn to_ord_u64(self) -> u64 {
        let bits = polars_utils::total_ord::canonical_f32(self).to_bits();
        (if bits >> 31 == 1 {
            !bits
        } else {
            bits | (1 << 31)
        }) as u64
    }
}

/// A split tree over the keys mapped with [`ToOrdU64`], inverted when
/// descending, so that classifying is a plain `>=` per level.
struct FastNumericTree<T: PolarsNumericType> {
    keys: Vec<u64>,
    flip: u64,
    b: usize,
    depth: u32,
    _pd: std::marker::PhantomData<fn() -> T>,
}

impl<T: PolarsNumericType> FastNumericTree<T>
where
    T::Native: ToOrdU64,
{
    fn new(tree: &SplitTree<T::Native>) -> Option<Self> {
        let flip = if tree.descending { u64::MAX } else { 0 };
        let keys = tree
            .keys
            .iter()
            .zip(&tree.min_cmp)
            .map(|(k, &strict)| (k.to_ord_u64() ^ flip).checked_add(strict as u64))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            keys,
            flip,
            b: tree.b,
            depth: tree.depth,
            _pd: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    fn classify<const N: usize>(&self, xs: [u64; N]) -> [usize; N] {
        let mut idx = [1usize; N];
        for _ in 0..self.depth {
            for j in 0..N {
                // SAFETY: idx[j] < b throughout the descent.
                let go_right = xs[j] >= unsafe { *self.keys.get_unchecked(idx[j] - 1) };
                idx[j] = 2 * idx[j] + go_right as usize;
            }
        }
        idx.map(|i| i - self.b)
    }
}

impl<T: PolarsNumericType> KeyTree for FastNumericTree<T>
where
    T::Native: ToOrdU64,
{
    fn gen_idxs_per_bucket(&self, keys: &Series, idxs_per_bucket: &mut [Vec<IdxSize>]) {
        let arr = keys.unpack::<T>().unwrap().downcast_as_array();
        let values = arr.values().as_slice();
        let mut ids: Vec<u32> = Vec::with_capacity(values.len());
        let mut it = values.chunks_exact(BATCH);
        for c in &mut it {
            let xs: [u64; BATCH] = core::array::from_fn(|j| c[j].to_ord_u64() ^ self.flip);
            ids.extend(self.classify(xs).map(|b| b as u32));
        }
        for &v in it.remainder() {
            ids.push(self.classify([v.to_ord_u64() ^ self.flip])[0] as u32);
        }
        if let Some(validity) = arr.validity().filter(|v| v.unset_bits() > 0) {
            for (id, is_valid) in ids.iter_mut().zip(validity.iter()) {
                if !is_valid {
                    *id = self.b as u32;
                }
            }
        }
        fill_idxs_from_ids(&ids, idxs_per_bucket);
    }
}

fn numeric_keys<T: PolarsNumericType>(split_keys: &Series) -> Vec<T::Native> {
    split_keys
        .unpack::<T>()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

/// The tree and the buckets to sort for numeric split keys.
fn numeric_tree<T: PolarsNumericType>(
    split_keys: &Series,
    descending: bool,
) -> (Box<dyn KeyTree>, Vec<bool>) {
    let keys = numeric_keys::<T>(split_keys);
    let eq = equal_to_prev(&keys, |a: &T::Native, b: &T::Native| a.tot_cmp(b));
    let tree = NumericTree::<T>(SplitTree::new(&keys, &eq, descending));
    (Box::new(tree), needs_sort(&eq))
}

/// Like [`numeric_tree`], but with a [`FastNumericTree`] where possible.
fn fast_numeric_tree<T: PolarsNumericType>(
    split_keys: &Series,
    descending: bool,
) -> (Box<dyn KeyTree>, Vec<bool>)
where
    T::Native: ToOrdU64,
{
    let keys = numeric_keys::<T>(split_keys);
    let eq = equal_to_prev(&keys, |a: &T::Native, b: &T::Native| a.tot_cmp(b));
    let tree = SplitTree::new(&keys, &eq, descending);
    let tree: Box<dyn KeyTree> = match FastNumericTree::<T>::new(&tree) {
        Some(fast) => Box::new(fast),
        None => Box::new(NumericTree::<T>(tree)),
    };
    (tree, needs_sort(&eq))
}

struct NumericTree<T: PolarsNumericType>(SplitTree<T::Native>);

impl<T: PolarsNumericType> KeyTree for NumericTree<T> {
    fn gen_idxs_per_bucket(&self, keys: &Series, idxs_per_bucket: &mut [Vec<IdxSize>]) {
        let arr = keys.unpack::<T>().unwrap().downcast_as_array();
        gen_idxs(
            &self.0,
            arr.values().iter(),
            arr.validity(),
            |a: &T::Native, b: &T::Native| a.tot_cmp(b),
            idxs_per_bucket,
        );
    }
}

impl KeyTree for SplitTree<Box<[u8]>> {
    fn gen_idxs_per_bucket(&self, keys: &Series, idxs_per_bucket: &mut [Vec<IdxSize>]) {
        if keys.dtype() == &DataType::BinaryOffset {
            let arr = keys.binary_offset().unwrap().downcast_as_array();
            gen_idxs(
                self,
                arr.values_iter(),
                arr.validity(),
                |a: &[u8], b| a.cmp(b),
                idxs_per_bucket,
            );
        } else {
            let arr = keys.binary().unwrap().downcast_as_array();
            gen_idxs(
                self,
                arr.values_iter(),
                arr.validity(),
                |a: &[u8], b| a.cmp(b),
                idxs_per_bucket,
            );
        }
    }
}

/// Assigns rows to buckets by key range.
pub struct BucketClassifier {
    tree: Box<dyn KeyTree>,
    num_buckets: usize,
    needs_sort: Vec<bool>,
}

impl BucketClassifier {
    /// Builds the classifier for a key column from its split keys.
    ///
    /// `split_keys` holds the `b - 1` non-null split keys in bucket order, that
    /// is descending when `descending` is set, and has the dtype of the key
    /// column as given to [`Self::gen_idxs_per_bucket`].
    pub fn new(split_keys: &Series, descending: bool) -> Self {
        let (tree, needs_sort): (Box<dyn KeyTree>, _) = match split_keys.dtype() {
            DataType::Binary | DataType::BinaryOffset => {
                let keys: Vec<Box<[u8]>> = if split_keys.dtype() == &DataType::BinaryOffset {
                    let ca = split_keys.binary_offset().unwrap();
                    ca.downcast_iter()
                        .flat_map(|arr| arr.values_iter())
                        .map(Box::from)
                        .collect()
                } else {
                    let ca = split_keys.binary().unwrap();
                    ca.downcast_iter()
                        .flat_map(|arr| arr.values_iter())
                        .map(Box::from)
                        .collect()
                };
                let eq = equal_to_prev(&keys, <Box<[u8]> as Ord>::cmp);
                (
                    Box::new(SplitTree::new(&keys, &eq, descending)),
                    needs_sort(&eq),
                )
            },
            DataType::UInt8 => fast_numeric_tree::<UInt8Type>(split_keys, descending),
            DataType::UInt16 => fast_numeric_tree::<UInt16Type>(split_keys, descending),
            DataType::UInt32 => fast_numeric_tree::<UInt32Type>(split_keys, descending),
            DataType::UInt64 => fast_numeric_tree::<UInt64Type>(split_keys, descending),
            DataType::Int8 => fast_numeric_tree::<Int8Type>(split_keys, descending),
            DataType::Int16 => fast_numeric_tree::<Int16Type>(split_keys, descending),
            DataType::Int32 => fast_numeric_tree::<Int32Type>(split_keys, descending),
            DataType::Int64 => fast_numeric_tree::<Int64Type>(split_keys, descending),
            DataType::Float32 => fast_numeric_tree::<Float32Type>(split_keys, descending),
            DataType::Float64 => fast_numeric_tree::<Float64Type>(split_keys, descending),
            dt => {
                with_match_physical_numeric_polars_type!(dt, |$T| numeric_tree::<$T>(split_keys, descending))
            },
        };

        Self {
            tree,
            num_buckets: split_keys.len() + 1,
            needs_sort,
        }
    }

    /// The number of value buckets. Nulls go to bucket `num_buckets()`.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Whether each of the `num_buckets() + 1` buckets has to be sorted.
    pub fn needs_sort(&self) -> &[bool] {
        &self.needs_sort
    }

    /// Appends the index of every row of `keys` to the bucket it belongs to.
    ///
    /// `keys` is the key column of a rechunked frame and `idxs_per_bucket` has
    /// length `num_buckets() + 1`.
    pub fn gen_idxs_per_bucket(&self, keys: &Series, idxs_per_bucket: &mut [Vec<IdxSize>]) {
        assert_eq!(idxs_per_bucket.len(), self.num_buckets + 1);
        if !keys.is_empty() {
            self.tree.gen_idxs_per_bucket(keys, idxs_per_bucket);
        }
    }
}
