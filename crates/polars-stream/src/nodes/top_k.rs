use std::any::Any;
use std::collections::BinaryHeap;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use polars_async::executor::TaskMetricAggregator;
use polars_core::prelude::row_encode::{_get_rows_encoded, get_row_encoding_context};
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::with_match_physical_numeric_polars_type;
use polars_io::predicates::RuntimeRange;
use polars_plan::plans::{DynamicPred, PredicateExpr};
use polars_row::RowEncodingOptions;
use polars_utils::IdxSize;
use polars_utils::priority::Priority;
use polars_utils::sort::ReorderWithNulls;
use polars_utils::total_ord::TotalOrdWrap;
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap, new_key_type};

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::nodes::in_memory_source::InMemorySourceNode;

new_key_type! {
    struct DfsKey;
    struct RowIdxKey;
}

/// Minimum number of rows a gather must remove from a subset to be worthwhile.
const MIN_GATHER_REMOVED_ROWS: usize = 16;

/// Up to this `k` several pipelines share their best `k` keys, so the published optimum is
/// the k-th key over all pipelines instead of the best k-th key of one pipeline.
const MAX_SHARED_K: usize = 128;

/// Represents a subset of a dataframe.
struct DfSubset {
    df: DataFrame,
    rows: Vec<RowIdxKey>,
    subset_len: usize,
}

impl DfSubset {
    /// Gather this subset into a contiguous DataFrame, updating the relevant row indices.
    pub fn gather(
        &mut self,
        row_idxs: &mut SlotMap<RowIdxKey, IdxSize>,
        gather_idx_buf: &mut Vec<IdxSize>,
    ) {
        if self.subset_len == self.df.height() {
            return;
        }

        gather_idx_buf.clear();
        let mut new_idx = 0;
        self.rows.retain(|row_idx_key| {
            let row_idx = &mut row_idxs[*row_idx_key];
            if *row_idx != IdxSize::MAX {
                gather_idx_buf.push(*row_idx);
                *row_idx = new_idx;
                new_idx += 1;
                true
            } else {
                row_idxs.remove(*row_idx_key);
                false
            }
        });

        unsafe { self.df = self.df.take_slice_unchecked_impl(gather_idx_buf, false) }
    }
}

struct BottomKWithPayload<P> {
    k: usize,
    heap: BinaryHeap<Priority<P, (DfsKey, RowIdxKey)>>,
    df_subsets: SlotMap<DfsKey, DfSubset>,
    row_idxs: SlotMap<RowIdxKey, IdxSize>,
    marked_for_pruning: SecondaryMap<DfsKey, ()>,
    to_prune: Vec<DfsKey>,
    gather_idxs: Vec<IdxSize>,
    shared_optimum: Arc<RwLock<Option<P>>>,
    /// The best `k` keys of all pipelines, when they share them.
    shared_keys: Option<Arc<Mutex<BinaryHeap<P>>>>,
}

impl<P: Ord + Clone> BottomKWithPayload<P> {
    pub fn new(k: usize, num_pipelines: usize) -> Self {
        let share_keys = num_pipelines > 1 && k <= MAX_SHARED_K;
        Self::with_shared(
            k,
            Arc::default(),
            share_keys.then(|| Arc::new(Mutex::new(BinaryHeap::with_capacity(k + 1)))),
        )
    }

    fn with_shared(
        k: usize,
        shared_optimum: Arc<RwLock<Option<P>>>,
        shared_keys: Option<Arc<Mutex<BinaryHeap<P>>>>,
    ) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
            df_subsets: SlotMap::with_key(),
            row_idxs: SlotMap::with_key(),
            marked_for_pruning: SecondaryMap::new(),
            to_prune: Vec::new(),
            gather_idxs: Vec::new(),
            shared_optimum,
            shared_keys,
        }
    }

    /// An empty one that shares the optimum with `self`.
    pub fn new_sharing(&self) -> Self {
        Self::with_shared(
            self.k,
            self.shared_optimum.clone(),
            self.shared_keys.clone(),
        )
    }

    pub fn add_df<Q>(
        &mut self,
        df: DataFrame,
        keys: impl IntoIterator<Item = Q>,
        is_less: impl Fn(&Q, &P) -> bool,
        is_less_owned: impl Fn(&P, &P) -> bool,
        to_owned: impl Fn(Q) -> P,
    ) {
        let dfs_key = self.df_subsets.insert(DfSubset {
            df,
            rows: Vec::new(),
            subset_len: 0,
        });

        let mut new_optimum = false;
        for (row_idx, key) in keys.into_iter().enumerate() {
            new_optimum |= self.add_one(
                dfs_key,
                row_idx.try_into().unwrap(),
                key,
                &is_less,
                &to_owned,
            );
        }
        self.mark_if_sparse(dfs_key);
        self.prune();

        let optimum = match &self.shared_keys {
            Some(shared_keys) => {
                // The keys of `df` still in the heap, below the current optimum: only
                // those can lower it.
                if self.df_subsets.get(dfs_key).is_none() {
                    return;
                }
                let shared_optimum = self.shared_optimum.read().clone();
                let new_keys: Vec<P> = self
                    .heap
                    .iter()
                    .filter(|p| {
                        p.1.0 == dfs_key && shared_optimum.as_ref().is_none_or(|v| p.0 < *v)
                    })
                    .map(|p| p.0.clone())
                    .collect();
                if new_keys.is_empty() {
                    return;
                }
                let mut shared_keys = shared_keys.lock();
                for key in new_keys {
                    if shared_keys.len() < self.k {
                        shared_keys.push(key);
                    } else if is_less_owned(&key, shared_keys.peek().unwrap()) {
                        *shared_keys.peek_mut().unwrap() = key;
                    }
                }
                if shared_keys.len() < self.k {
                    return;
                }
                shared_keys.peek().unwrap().clone()
            },
            None => {
                if !new_optimum || self.heap.len() < self.k {
                    return;
                }
                self.peek_optimum().unwrap().clone()
            },
        };

        let mut shared_optimum = self.shared_optimum.write();
        if shared_optimum
            .as_ref()
            .is_none_or(|v| is_less_owned(&optimum, v))
        {
            *shared_optimum = Some(optimum);
        }
    }

    fn add_one<Q>(
        &mut self,
        dfs_key: DfsKey,
        row_idx: IdxSize,
        key: Q,
        is_less: impl Fn(&Q, &P) -> bool,
        to_owned: impl Fn(Q) -> P,
    ) -> bool {
        // We use a max-heap for our bottom k. This means the top element in our heap (peek())
        // is the first to be replaced.
        if !(self.heap.len() < self.k || is_less(&key, self.peek_optimum().unwrap())) {
            return false;
        }
        let opt = self.heap.peek().map(|p| p.1);
        let row_idx_key = self.row_idxs.insert(row_idx);
        let df_subset = &mut self.df_subsets[dfs_key];
        df_subset.subset_len += 1;
        df_subset.rows.push(row_idx_key);
        self.heap
            .push(Priority(to_owned(key), (dfs_key, row_idx_key)));

        if self.heap.len() > self.k {
            let (dfs_key, row_idx_key) = self.heap.pop().unwrap().1;
            self.row_idxs[row_idx_key] = IdxSize::MAX;
            self.df_subsets[dfs_key].subset_len -= 1;
            self.mark_if_sparse(dfs_key);
        }

        opt != self.heap.peek().map(|p| p.1)
    }

    /// Schedules a subset for pruning once none of its rows are in the heap, or once at most half
    /// of its rows are in the heap and a gather would remove enough rows to be worthwhile.
    fn mark_if_sparse(&mut self, dfs_key: DfsKey) {
        let df_subset = &self.df_subsets[dfs_key];
        let in_heap = df_subset.subset_len;
        let removed = df_subset.df.height() - in_heap;
        let should_prune =
            in_heap == 0 || (removed >= in_heap && removed >= MIN_GATHER_REMOVED_ROWS);
        if should_prune && self.marked_for_pruning.insert(dfs_key, ()).is_none() {
            self.to_prune.push(dfs_key);
        }
    }

    pub fn prune(&mut self) {
        for dfs_key in self.to_prune.drain(..) {
            self.marked_for_pruning.remove(dfs_key);
            if self.df_subsets[dfs_key].subset_len == 0 {
                let df_subset = self.df_subsets.remove(dfs_key).unwrap();
                for row_idx in df_subset.rows {
                    self.row_idxs.remove(row_idx);
                }
            } else {
                self.df_subsets[dfs_key].gather(&mut self.row_idxs, &mut self.gather_idxs);
            }
        }
    }

    pub fn combine(&mut self, other: &BottomKWithPayload<P>) {
        let mut new_df_keys =
            SecondaryMap::<DfsKey, DfsKey>::with_capacity(other.df_subsets.capacity());
        for (dfs_key, dfs) in &other.df_subsets {
            if dfs.subset_len > 0 {
                let subset = DfSubset {
                    df: dfs.df.clone(),
                    rows: Vec::new(),
                    subset_len: 0,
                };
                new_df_keys.insert(dfs_key, self.df_subsets.insert(subset));
            }
        }
        for prio in &other.heap {
            let (dfs_key, row_idx_key) = prio.1;
            self.add_one(
                new_df_keys[dfs_key],
                other.row_idxs[row_idx_key],
                prio.0.clone(),
                |l, r| l < r,
                |x| x,
            );
        }
        for &dfs_key in new_df_keys.values() {
            self.mark_if_sparse(dfs_key);
        }
        self.prune();
    }

    pub fn finalize(&mut self) -> Option<DataFrame> {
        let mut gather_idx_buf = Vec::new();
        if self.df_subsets.is_empty() {
            return None;
        }
        let ret = accumulate_dataframes_vertical(self.df_subsets.drain().map(|(_k, mut df)| {
            df.gather(&mut self.row_idxs, &mut gather_idx_buf);
            df.df
        }));
        self.heap.clear();
        self.row_idxs.clear();
        self.marked_for_pruning.clear();
        self.to_prune.clear();
        Some(ret.unwrap())
    }

    fn peek_optimum(&self) -> Option<&P> {
        self.heap.peek().map(|x| &x.0)
    }
}

trait DfByKeyReducer: Any + Send + 'static {
    fn new_empty(&self) -> Box<dyn DfByKeyReducer>;
    fn new_pred(&self) -> Arc<dyn PredicateExpr>;
    fn add(&mut self, df: DataFrame, keys: DataFrame);
    fn combine(&mut self, other: &dyn DfByKeyReducer);
    fn finalize(self: Box<Self>) -> Option<DataFrame>;
}

struct PrimitiveBottomK<T: PolarsNumericType, const REVERSE: bool, const NULLS_LAST: bool> {
    inner: BottomKWithPayload<
        ReorderWithNulls<TotalOrdWrap<T::Physical<'static>>, REVERSE, NULLS_LAST>,
    >,
    dtype: DataType,
}

impl<T: PolarsNumericType, const REVERSE: bool, const NULLS_LAST: bool>
    PrimitiveBottomK<T, REVERSE, NULLS_LAST>
{
    fn new(k: usize, num_pipelines: usize, dtype: DataType) -> Self {
        Self {
            inner: BottomKWithPayload::new(k, num_pipelines),
            dtype,
        }
    }
}

impl<T: PolarsNumericType, const REVERSE: bool, const NULLS_LAST: bool> DfByKeyReducer
    for PrimitiveBottomK<T, REVERSE, NULLS_LAST>
{
    fn new_empty(&self) -> Box<dyn DfByKeyReducer> {
        Box::new(Self {
            inner: self.inner.new_sharing(),
            dtype: self.dtype.clone(),
        })
    }

    fn new_pred(&self) -> Arc<dyn PredicateExpr> {
        Arc::new(PrimitiveBottomKPredicate::<T, REVERSE, NULLS_LAST> {
            shared_optimum: self.inner.shared_optimum.clone(),
            dtype: self.dtype.clone(),
        })
    }

    fn add(&mut self, df: DataFrame, keys: DataFrame) {
        assert!(keys.width() == 1);
        let keys = keys.columns()[0].as_materialized_series();
        let key_ca: &ChunkedArray<T> = keys.as_phys_any().downcast_ref().unwrap();
        self.inner.add_df(
            df,
            key_ca
                .iter()
                .map(|opt_x| ReorderWithNulls(opt_x.map(TotalOrdWrap))),
            |l, r| l < r,
            |l, r| l < r,
            |x| x,
        );
    }

    fn combine(&mut self, other: &dyn DfByKeyReducer) {
        let other: &Self = (other as &dyn Any).downcast_ref().unwrap();
        self.inner.combine(&other.inner);
    }

    fn finalize(mut self: Box<Self>) -> Option<DataFrame> {
        self.inner.finalize()
    }
}

struct PrimitiveBottomKPredicate<T: PolarsNumericType, const REVERSE: bool, const NULLS_LAST: bool>
{
    #[allow(clippy::type_complexity)]
    shared_optimum: Arc<
        RwLock<Option<ReorderWithNulls<TotalOrdWrap<T::Physical<'static>>, REVERSE, NULLS_LAST>>>,
    >,
    dtype: DataType,
}

impl<T: PolarsNumericType, const REVERSE: bool, const NULLS_LAST: bool> PredicateExpr
    for PrimitiveBottomKPredicate<T, REVERSE, NULLS_LAST>
{
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>> {
        let Some(v) = self.shared_optimum.read().clone() else {
            return Ok(None);
        };

        if columns[0].is_empty() {
            return Ok(Some(Column::new_empty(
                PlSmallStr::EMPTY,
                &DataType::Boolean,
            )));
        }

        if columns[0].dtype().is_null() || matches!(columns[0], Column::Scalar(_)) {
            let cv = columns[0]
                .get(0)?
                .null_to_none()
                // Physical: e.g. decimals extract as their (truncated) logical value.
                .map(|v| TotalOrdWrap(v.to_physical().try_extract().unwrap()));
            let keep = ReorderWithNulls(cv) < v;
            let s = Scalar::new(DataType::Boolean, AnyValue::Boolean(keep));
            Ok(Some(Column::new_scalar(
                PlSmallStr::EMPTY,
                s,
                columns[0].len(),
            )))
        } else {
            let keys = columns[0].as_materialized_series();
            let key_ca: &ChunkedArray<T> = keys.as_phys_any().downcast_ref().unwrap();
            let pred: BooleanChunked = key_ca
                .iter()
                .map(|opt_x| ReorderWithNulls(opt_x.map(TotalOrdWrap)) < v)
                .collect_ca(PlSmallStr::EMPTY);
            Ok(Some(Column::from(pred.into_series())))
        }
    }

    fn filters_rows(&self) -> bool {
        self.shared_optimum.read().is_some()
    }

    fn runtime_range(&self) -> RuntimeRange {
        // Floats order NaN apart from statistics, enums by their categories.
        if !(self.dtype.is_integer() || self.dtype.is_temporal() || self.dtype.is_decimal()) {
            return RuntimeRange::Disabled;
        }
        let Some(ReorderWithNulls(Some(TotalOrdWrap(threshold)))) = *self.shared_optimum.read()
        else {
            return RuntimeRange::Pending;
        };
        let physical = ChunkedArray::<T>::from_slice(PlSmallStr::EMPTY, &[threshold]);
        let Ok(value) = (unsafe { physical.into_series().from_physical_unchecked(&self.dtype) })
            .and_then(|s| s.get(0).map(|v| v.into_static()))
        else {
            return RuntimeRange::Disabled;
        };
        threshold_range(Scalar::new(self.dtype.clone(), value), REVERSE, NULLS_LAST)
    }
}

/// The values a bottom-k with this threshold can still take, as a range.
fn threshold_range(threshold: Scalar, descending: bool, nulls_last: bool) -> RuntimeRange {
    let (lo, hi) = if descending {
        (Some(threshold), None)
    } else {
        (None, Some(threshold))
    };
    RuntimeRange::Range {
        lo,
        hi,
        nulls_match: !nulls_last,
    }
}

struct BinaryBottomK<const REVERSE: bool, const NULLS_LAST: bool> {
    inner: BottomKWithPayload<ReorderWithNulls<Vec<u8>, REVERSE, NULLS_LAST>>,
    dtype: DataType,
}

impl<const REVERSE: bool, const NULLS_LAST: bool> BinaryBottomK<REVERSE, NULLS_LAST> {
    fn new(k: usize, num_pipelines: usize, dtype: DataType) -> Self {
        Self {
            inner: BottomKWithPayload::new(k, num_pipelines),
            dtype,
        }
    }
}

impl<const REVERSE: bool, const NULLS_LAST: bool> DfByKeyReducer
    for BinaryBottomK<REVERSE, NULLS_LAST>
{
    fn new_empty(&self) -> Box<dyn DfByKeyReducer> {
        Box::new(Self {
            inner: self.inner.new_sharing(),
            dtype: self.dtype.clone(),
        })
    }

    fn new_pred(&self) -> Arc<dyn PredicateExpr> {
        Arc::new(BinaryBottomKPredicate {
            shared_optimum: self.inner.shared_optimum.clone(),
            dtype: self.dtype.clone(),
        })
    }

    fn add(&mut self, df: DataFrame, keys: DataFrame) {
        assert!(keys.width() == 1);
        let key_ca = if let Ok(ca_str) = keys[0].str() {
            ca_str.as_binary()
        } else {
            keys[0].binary().unwrap().clone()
        };
        self.inner.add_df(
            df,
            key_ca
                .iter()
                .map(ReorderWithNulls::<_, REVERSE, NULLS_LAST>),
            |l, r| l < &r.as_deref(),
            |l, r| l < r,
            |x| ReorderWithNulls(x.0.map(<[u8]>::to_vec)),
        );
    }

    fn combine(&mut self, other: &dyn DfByKeyReducer) {
        let other: &Self = (other as &dyn Any).downcast_ref().unwrap();
        self.inner.combine(&other.inner);
    }

    fn finalize(mut self: Box<Self>) -> Option<DataFrame> {
        self.inner.finalize()
    }
}

struct BinaryBottomKPredicate<const REVERSE: bool, const NULLS_LAST: bool> {
    shared_optimum: Arc<RwLock<Option<ReorderWithNulls<Vec<u8>, REVERSE, NULLS_LAST>>>>,
    dtype: DataType,
}

impl<const REVERSE: bool, const NULLS_LAST: bool> PredicateExpr
    for BinaryBottomKPredicate<REVERSE, NULLS_LAST>
{
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>> {
        let Some(v) = self.shared_optimum.read().clone() else {
            return Ok(None);
        };

        if columns[0].is_empty() {
            return Ok(Some(Column::new_empty(
                PlSmallStr::EMPTY,
                &DataType::Boolean,
            )));
        }

        if columns[0].dtype().is_null() || matches!(columns[0], Column::Scalar(_)) {
            let scalar = columns[0].get(0)?;
            let cv = match &scalar {
                AnyValue::Null => None,
                AnyValue::String(s) => Some(s.as_bytes()),
                AnyValue::StringOwned(s) => Some(s.as_bytes()),
                AnyValue::Binary(b) => Some(*b),
                AnyValue::BinaryOwned(b) => Some(b.as_slice()),
                _ => unreachable!(),
            };
            let keep = ReorderWithNulls(cv) < v.as_deref();
            let s = Scalar::new(DataType::Boolean, AnyValue::Boolean(keep));
            Ok(Some(Column::new_scalar(
                PlSmallStr::EMPTY,
                s,
                columns[0].len(),
            )))
        } else {
            let keys = columns[0].as_materialized_series();
            let key_ca = if let Ok(ca_str) = keys.str() {
                ca_str.as_binary()
            } else {
                keys.binary().unwrap().clone()
            };
            let pred: BooleanChunked = key_ca
                .iter()
                .map(|opt_x| ReorderWithNulls(opt_x) < v.as_deref())
                .collect_ca(PlSmallStr::EMPTY);
            Ok(Some(Column::from(pred.into_series())))
        }
    }

    fn filters_rows(&self) -> bool {
        self.shared_optimum.read().is_some()
    }

    fn runtime_range(&self) -> RuntimeRange {
        let Some(ReorderWithNulls(Some(threshold))) = self.shared_optimum.read().clone() else {
            return RuntimeRange::Pending;
        };
        let value = match self.dtype {
            DataType::String => match String::from_utf8(threshold) {
                Ok(s) => AnyValue::StringOwned(s.into()),
                Err(_) => return RuntimeRange::Disabled,
            },
            _ => AnyValue::BinaryOwned(threshold),
        };
        threshold_range(Scalar::new(self.dtype.clone(), value), REVERSE, NULLS_LAST)
    }
}

struct RowEncodedBottomK {
    inner: BottomKWithPayload<Vec<u8>>,
    reverse: Vec<bool>,
    nulls_last: Vec<bool>,
    first_key_dtype: DataType,
}

impl RowEncodedBottomK {
    fn new(
        k: usize,
        num_pipelines: usize,
        reverse: Vec<bool>,
        nulls_last: Vec<bool>,
        first_key_dtype: DataType,
    ) -> Self {
        Self {
            inner: BottomKWithPayload::new(k, num_pipelines),
            reverse,
            nulls_last,
            first_key_dtype,
        }
    }
}

impl DfByKeyReducer for RowEncodedBottomK {
    fn new_empty(&self) -> Box<dyn DfByKeyReducer> {
        Box::new(Self {
            inner: self.inner.new_sharing(),
            reverse: self.reverse.clone(),
            nulls_last: self.nulls_last.clone(),
            first_key_dtype: self.first_key_dtype.clone(),
        })
    }

    fn new_pred(&self) -> Arc<dyn PredicateExpr> {
        Arc::new(FirstKeyBottomKPredicate {
            shared_optimum: self.inner.shared_optimum.clone(),
            descending: self.reverse[0],
            nulls_last: self.nulls_last[0],
            dtype: self.first_key_dtype.clone(),
        })
    }

    fn add(&mut self, df: DataFrame, keys: DataFrame) {
        let keys_encoded = _get_rows_encoded(keys.columns(), &self.reverse, &self.nulls_last)
            .unwrap()
            .into_array();
        self.inner.add_df(
            df,
            keys_encoded.values_iter(),
            |l, r| *l < r.as_slice(),
            |l, r| l < r,
            |x| x.to_vec(),
        );
    }

    fn combine(&mut self, other: &dyn DfByKeyReducer) {
        let other: &Self = (other as &dyn Any).downcast_ref().unwrap();
        self.inner.combine(&other.inner);
    }

    fn finalize(mut self: Box<Self>) -> Option<DataFrame> {
        self.inner.finalize()
    }
}

/// Keeps the values of the first key that can still be in a row-encoded bottom-k: those
/// up to and including the first key of the threshold row.
struct FirstKeyBottomKPredicate {
    shared_optimum: Arc<RwLock<Option<Vec<u8>>>>,
    descending: bool,
    nulls_last: bool,
    dtype: DataType,
}

impl PredicateExpr for FirstKeyBottomKPredicate {
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>> {
        let Some(threshold) = self.shared_optimum.read().clone() else {
            return Ok(None);
        };
        // The encoding of a value is never a prefix of another one, so a value equal to the
        // first key of the threshold is a prefix of it and compares below it.
        let encoded = _get_rows_encoded(&columns[0..1], &[self.descending], &[self.nulls_last])?;
        let pred: BooleanChunked = encoded
            .iter()
            .map(|row| row < threshold.as_slice())
            .collect_ca(PlSmallStr::EMPTY);
        Ok(Some(Column::from(pred.into_series())))
    }

    fn filters_rows(&self) -> bool {
        self.shared_optimum.read().is_some()
    }

    fn runtime_range(&self) -> RuntimeRange {
        // Floats order NaN apart from statistics, enums and categoricals by their categories.
        if !(self.dtype.is_integer()
            || self.dtype.is_temporal()
            || self.dtype.is_decimal()
            || matches!(self.dtype, DataType::String | DataType::Binary))
        {
            return RuntimeRange::Disabled;
        }
        let Some(threshold) = self.shared_optimum.read().clone() else {
            return RuntimeRange::Pending;
        };
        let arrow_dtype = self.dtype.to_physical().to_arrow(CompatLevel::newest());
        let opt = RowEncodingOptions::new_sorted(self.descending, self.nulls_last);
        let ctxt = get_row_encoding_context(&self.dtype);
        let [first_key] = unsafe {
            polars_row::decode::decode_rows(
                &mut [threshold.as_slice()],
                &[opt],
                &[ctxt],
                &[arrow_dtype],
            )
        }
        .try_into()
        .unwrap();
        let Ok(value) = Series::try_from((PlSmallStr::EMPTY, first_key))
            .and_then(|s| unsafe { s.from_physical_unchecked(&self.dtype) })
            .and_then(|s| s.get(0).map(|v| v.into_static()))
        else {
            return RuntimeRange::Disabled;
        };
        if value.is_null() {
            return RuntimeRange::Pending;
        }
        threshold_range(
            Scalar::new(self.dtype.clone(), value),
            self.descending,
            self.nulls_last,
        )
    }
}

fn new_top_k_reducer(
    k: usize,
    num_pipelines: usize,
    reverse: &[bool],
    nulls_last: &[bool],
    key_schema: &Schema,
) -> Box<dyn DfByKeyReducer> {
    if key_schema.len() == 1 {
        let (_name, dt) = key_schema.get_at_index(0).unwrap();
        match dt {
            dt if dt.is_primitive_numeric() | dt.is_temporal() | dt.is_decimal() | dt.is_enum() => {
                return with_match_physical_numeric_polars_type!(dt.to_physical(), |$T| {
                    match (reverse[0], nulls_last[0]) {
                        (false, false) => Box::new(PrimitiveBottomK::<$T, true, false>::new(k, num_pipelines, dt.clone())),
                        (false, true) => Box::new(PrimitiveBottomK::<$T, true, true>::new(k, num_pipelines, dt.clone())),
                        (true, false) => Box::new(PrimitiveBottomK::<$T, false, false>::new(k, num_pipelines, dt.clone())),
                        (true, true) => Box::new(PrimitiveBottomK::<$T, false, true>::new(k, num_pipelines, dt.clone())),
                    }
                });
            },

            DataType::String | DataType::Binary => {
                return match (reverse[0], nulls_last[0]) {
                    (false, false) => Box::new(BinaryBottomK::<true, false>::new(
                        k,
                        num_pipelines,
                        dt.clone(),
                    )),
                    (false, true) => Box::new(BinaryBottomK::<true, true>::new(
                        k,
                        num_pipelines,
                        dt.clone(),
                    )),
                    (true, false) => Box::new(BinaryBottomK::<false, false>::new(
                        k,
                        num_pipelines,
                        dt.clone(),
                    )),
                    (true, true) => Box::new(BinaryBottomK::<false, true>::new(
                        k,
                        num_pipelines,
                        dt.clone(),
                    )),
                };
            },

            // TODO: categorical single-key.
            _ => {},
        }
    }

    let reverse = reverse.iter().map(|r| !r).collect();
    let first_key_dtype = key_schema.get_at_index(0).unwrap().1.clone();
    Box::new(RowEncodedBottomK::new(
        k,
        num_pipelines,
        reverse,
        nulls_last.to_vec(),
        first_key_dtype,
    ))
}

enum TopKState {
    WaitingForK(InMemorySinkNode),

    Sink {
        key_selectors: Vec<StreamExpr>,
        reducers: Vec<Box<dyn DfByKeyReducer>>,
    },

    Source(InMemorySourceNode),

    Done,
}

pub struct TopKNode {
    reverse: Vec<bool>,
    nulls_last: Vec<bool>,
    key_schema: Arc<Schema>,
    key_selectors: Vec<StreamExpr>,
    state: TopKState,
    dyn_pred: Option<DynamicPred>,
}

impl TopKNode {
    pub fn new(
        k_schema: Arc<Schema>,
        reverse: Vec<bool>,
        nulls_last: Vec<bool>,
        key_schema: Arc<Schema>,
        key_selectors: Vec<StreamExpr>,
        dyn_pred: Option<DynamicPred>,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        Self {
            reverse,
            nulls_last,
            key_schema,
            key_selectors,
            state: TopKState::WaitingForK(InMemorySinkNode::new(k_schema, task_metrics)),
            dyn_pred,
        }
    }
}

impl ComputeNode for TopKNode {
    fn name(&self) -> &str {
        if self.reverse.iter().all(|r| *r) {
            "bottom-k"
        } else {
            "top-k"
        }
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                self.state = TopKState::Done;
            },
            // We've received k, transition to being a sink.
            TopKState::WaitingForK(inner) if recv[1] == PortState::Done => {
                let k_frame = inner.get_output()?.unwrap();
                polars_ensure!(k_frame.height() == 1, ComputeError: "got more than one value for 'k' in top_k");
                let k_item = k_frame.columns()[0].get(0)?;
                let k = k_item.extract::<usize>().ok_or_else(
                    || polars_err!(ComputeError: "invalid value of 'k' in top_k: {:?}", k_item),
                )?;

                if k > 0 {
                    let reducer = new_top_k_reducer(
                        k,
                        state.num_pipelines,
                        &self.reverse,
                        &self.nulls_last,
                        &self.key_schema,
                    );
                    if let Some(dyn_pred) = &self.dyn_pred {
                        dyn_pred.set(reducer.new_pred());
                    }
                    let reducers = (0..state.num_pipelines)
                        .map(|_| reducer.new_empty())
                        .collect();
                    self.state = TopKState::Sink {
                        key_selectors: core::mem::take(&mut self.key_selectors),
                        reducers,
                    };
                } else {
                    self.state = TopKState::Done;
                }
            },
            // Input is done, transition to being a source.
            TopKState::Sink { reducers, .. } if recv[0] == PortState::Done => {
                let reducer = RAYON.install(|| {
                    core::mem::take(reducers)
                        .into_par_iter()
                        .with_max_len(1)
                        .reduce_with(|mut l, r| {
                            l.combine(&*r);
                            l
                        })
                        .unwrap()
                });
                if let Some(df) = reducer.finalize() {
                    self.state = TopKState::Source(InMemorySourceNode::new(
                        Arc::new(df),
                        MorselSeq::default(),
                    ));
                } else {
                    self.state = TopKState::Done;
                }
            },
            // Nothing to change.
            _ => {},
        }

        // Communicate our state.
        match &mut self.state {
            TopKState::WaitingForK(inner) => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Blocked;
                inner.update_state(&mut recv[1..2], &mut [], state)?;
            },
            TopKState::Sink { .. } => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
                recv[1] = PortState::Done;
            },
            TopKState::Source(src) => {
                src.update_state(&mut [], send, state)?;
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
            },
            TopKState::Done => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        match &self.state {
            TopKState::WaitingForK(_) | TopKState::Sink { .. } | TopKState::Done => {
                NodeMemoryUsage::Bounded
            },
            TopKState::Source(src) => src.memory_usage(),
        }
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);
        match &mut self.state {
            TopKState::WaitingForK(inner) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[0].is_none());
                inner.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
            },
            TopKState::Sink {
                key_selectors,
                reducers,
            } => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports[1].is_none());
                let receivers = recv_ports[0].take().unwrap().parallel();

                for (mut recv, reducer) in receivers.into_iter().zip(reducers) {
                    let key_selectors = &*key_selectors;
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = recv.recv().await {
                            let df = morsel.into_df().await;
                            let mut key_columns = Vec::new();
                            for selector in key_selectors {
                                let s = selector.evaluate(&df, &state.in_memory_exec_state).await?;
                                key_columns.push(s.into_column());
                            }
                            let keys = unsafe {
                                DataFrame::new_unchecked_with_broadcast(df.height(), key_columns)?
                            };

                            reducer.add(df, keys);
                        }

                        Ok(())
                    }));
                }
            },

            TopKState::Source(src) => {
                assert!(recv_ports[0].is_none());
                assert!(recv_ports[1].is_none());
                src.spawn(scope, &mut [], send_ports, state, join_handles);
            },

            TopKState::Done => unreachable!(),
        }
    }
}
