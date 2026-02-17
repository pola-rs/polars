use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

use polars_core::frame::column::ScalarColumn;
use polars_utils::unique_id::UniqueId;
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use super::*;

pub trait PredicateExpr: Send + Sync + Any {
    // Invariant: output column must be of type `Boolean`. If true a value is
    // included, if false it is filtered out. If None is returned it is assumed
    // all values are needed.
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>>;
}

pub struct TrivialPredicateExpr;

impl PredicateExpr for TrivialPredicateExpr {
    fn evaluate(&self, _columns: &[Column]) -> PolarsResult<Option<Column>> {
        Ok(None)
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

impl Debug for DynamicPred {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dynamic_pred: {:}", self.id())
    }
}

impl PartialEq for DynamicPred {
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

    pub fn evaluate(&self, columns: &[Column]) -> PolarsResult<Column> {
        if self.inner.is_set.load(Ordering::Acquire) {
            let guard = self.inner.pred.read().unwrap();
            let dyn_func = guard.as_ref().unwrap();
            if let Some(pred) = dyn_func.evaluate(columns)? {
                return Ok(pred);
            }
        }

        let s = Scalar::new(DataType::Boolean, AnyValue::Boolean(true));
        Ok(Column::Scalar(ScalarColumn::new(
            columns[0].name().clone(),
            s,
            columns[0].len(),
        )))
    }
}

pub fn new_dynamic_pred(node: Node, arena: &mut Arena<AExpr>) -> (Node, DynamicPred) {
    let pred = DynamicPred::new();
    let function = IRFunctionExpr::DynamicPred { pred: pred.clone() };
    let options = function.function_options();
    let aexpr = AExpr::Function {
        input: vec![ExprIR::from_node(node, arena)],
        function,
        options,
    };

    (arena.add(aexpr), pred)
}
