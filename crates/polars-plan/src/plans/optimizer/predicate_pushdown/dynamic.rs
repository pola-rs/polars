use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{RwLock, Weak};

use polars_io::predicates::{DynamicPredicateSource, RuntimeRange};
use polars_utils::unique_id::UniqueId;
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use super::*;

pub trait PredicateExpr: Send + Sync + Any {
    // Invariant: output column must be of type `Boolean`. If true a value is
    // included, if false it is filtered out. If None is returned it is assumed
    // all values are needed.
    fn evaluate(&self, _columns: &[Column]) -> PolarsResult<Option<Column>> {
        Ok(None)
    }

    // Whether a batch with these per-column `min`, `max` and `null_count`
    // statistics can be skipped entirely. True skips the batch. None means the
    // statistics do not settle it.
    fn evaluate_stats(
        &self,
        _min: &Column,
        _max: &Column,
        _null_count: &Column,
    ) -> PolarsResult<Option<Column>> {
        Ok(None)
    }

    // A range every matching value lies in, for a reader that skips batches by
    // their statistics. `Disabled` when the predicate gives none.
    fn runtime_range(&self) -> RuntimeRange {
        RuntimeRange::Disabled
    }

    // Whether `evaluate` can reject rows. A predicate that cannot is not
    // evaluated per row.
    fn filters_rows(&self) -> bool {
        true
    }

    // Whether a reader may stop evaluating the predicate when it rejects too
    // little: the predicate stays as it is once set, and its producer checks
    // every row again. A predicate that tightens over time must be kept.
    fn can_bypass(&self) -> bool {
        false
    }
}

pub struct TrivialPredicateExpr;

impl PredicateExpr for TrivialPredicateExpr {
    fn filters_rows(&self) -> bool {
        false
    }
}

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
struct Inner {
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    pred: RwLock<Option<Arc<dyn PredicateExpr>>>,
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    is_set: AtomicBool,
    id: UniqueId,
}

#[derive(Clone)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct DynamicPred {
    inner: Arc<Inner>,
}

#[derive(Clone)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct DynamicPredWeakRef {
    inner: Weak<Inner>,
}

impl Debug for DynamicPred {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dynamic_pred: {:}", self.id())
    }
}

impl Debug for DynamicPredWeakRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(id) = self.id() {
            write!(f, "dynamic_pred: {:}", id)
        } else {
            write!(f, "dynamic_pred: dropped")
        }
    }
}

impl PartialEq for DynamicPred {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl PartialEq for DynamicPredWeakRef {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Hash for DynamicPred {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.id.hash(state);
    }
}

impl DynamicPred {
    fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                pred: Default::default(),
                is_set: Default::default(),
                id: UniqueId::new(),
            }),
        }
    }

    fn downgrade(&self) -> DynamicPredWeakRef {
        DynamicPredWeakRef {
            inner: Arc::downgrade(&self.inner),
        }
    }
}

impl DynamicPred {
    pub fn id(&self) -> &UniqueId {
        &self.inner.id
    }

    pub fn set(&self, pred: Arc<dyn PredicateExpr>) {
        {
            let mut guard = self.inner.pred.write().unwrap();
            *guard = Some(pred);
        }
        self.inner.is_set.store(true, Ordering::Release);
    }

    pub fn is_set(&self) -> bool {
        self.inner.is_set.load(Ordering::Acquire)
    }
}

impl DynamicPredWeakRef {
    pub fn id(&self) -> Option<UniqueId> {
        Some(self.inner.upgrade()?.id)
    }

    pub fn evaluate(&self, columns: &[Column]) -> PolarsResult<Column> {
        if let Some(inner) = self.inner.upgrade()
            && inner.is_set.load(Ordering::Acquire)
        {
            let guard = inner.pred.read().unwrap();
            let dyn_func = guard.as_ref().unwrap();
            if let Some(pred) = dyn_func.evaluate(columns)? {
                return Ok(pred);
            }
        }

        Ok(all_of(columns[0].name().clone(), columns[0].len(), true))
    }

    pub fn evaluate_stats(
        &self,
        min: &Column,
        max: &Column,
        null_count: &Column,
    ) -> PolarsResult<Column> {
        if let Some(inner) = self.inner.upgrade()
            && inner.is_set.load(Ordering::Acquire)
        {
            let guard = inner.pred.read().unwrap();
            let dyn_func = guard.as_ref().unwrap();
            if let Some(skip) = dyn_func.evaluate_stats(min, max, null_count)? {
                return Ok(skip);
            }
        }

        Ok(all_of(min.name().clone(), min.len(), false))
    }
}

impl DynamicPredicateSource for DynamicPredWeakRef {
    fn runtime_range(&self) -> RuntimeRange {
        let Some(inner) = self.inner.upgrade() else {
            return RuntimeRange::Disabled;
        };
        if !inner.is_set.load(Ordering::Acquire) {
            return RuntimeRange::Pending;
        }
        let guard = inner.pred.read().unwrap();
        guard.as_ref().unwrap().runtime_range()
    }

    fn filters_rows(&self) -> bool {
        self.with_set(|pred| pred.filters_rows())
    }

    fn can_bypass(&self) -> bool {
        self.with_set(|pred| pred.can_bypass())
    }
}

impl DynamicPredWeakRef {
    /// `f` on the predicate once it is set, `false` before.
    fn with_set(&self, f: impl FnOnce(&dyn PredicateExpr) -> bool) -> bool {
        let Some(inner) = self.inner.upgrade() else {
            return false;
        };
        if !inner.is_set.load(Ordering::Acquire) {
            return false;
        }
        let guard = inner.pred.read().unwrap();
        f(guard.as_ref().unwrap().as_ref())
    }
}

fn all_of(name: PlSmallStr, len: usize, value: bool) -> Column {
    Column::new_scalar(name, Scalar::from(value), len)
}

/// A predicate over `node` whose value a producer sets at run time, evaluated
/// per row wherever it lands.
pub fn new_dynamic_pred(node: Node, arena: &mut Arena<AExpr>) -> (Node, DynamicPred) {
    dynamic_pred_node(node, false, arena)
}

/// A predicate over `node` whose value a producer sets at run time, which a scan
/// only uses to skip batches by their statistics and never evaluates per row.
pub fn new_batch_only_dynamic_pred(node: Node, arena: &mut Arena<AExpr>) -> (Node, DynamicPred) {
    dynamic_pred_node(node, true, arena)
}

fn dynamic_pred_node(
    node: Node,
    batch_only: bool,
    arena: &mut Arena<AExpr>,
) -> (Node, DynamicPred) {
    let pred = DynamicPred::new();
    let function = IRFunctionExpr::DynamicPred {
        pred: pred.downgrade(),
        batch_only,
    };
    let options = function.function_options();
    let aexpr = AExpr::Function {
        input: vec![ExprIR::from_node(node, arena)],
        function,
        options,
    };

    (arena.add(aexpr), pred)
}
