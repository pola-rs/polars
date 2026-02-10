use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;

use polars_core::frame::column::ScalarColumn;
use polars_utils::unique_id::UniqueId;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

pub trait DynamicExpr: Send + Sync {
    // Invariant: Output Column must be of type `Boolean`.
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Column>;
}

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
struct Inner {
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    pred: RwLock<Option<Box<dyn DynamicExpr>>>,
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

    pub fn set(&self, pred: Box<dyn DynamicExpr>) {
        {
            let mut guard = self.inner.pred.write().unwrap();
            *guard = Some(pred);
        }
        self.inner
            .is_set
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn evaluate(&self, columns: &[Column]) -> PolarsResult<Column> {
        let h = columns[0].len();

        // Can be relaxed, worst thing that can happen is that we read
        // more data than strictly needed.
        if self.inner.is_set.load(std::sync::atomic::Ordering::Relaxed) {
            let guard = self.inner.pred.read().unwrap();
            let dyn_func = guard.as_ref().unwrap();
            dyn_func.evaluate(columns)
        } else {
            let s = Scalar::new(DataType::Boolean, AnyValue::Boolean(true));
            Ok(Column::Scalar(ScalarColumn::new(
                columns[0].name().clone(),
                s,
                1,
            )))
        }
    }
}

pub fn new_dynamic_pred(node: Node, arena: &mut Arena<AExpr>) -> (Node, DynamicPred) {
    let pred = DynamicPred::new();
    let function = IRFunctionExpr::DynamicExpr { pred: pred.clone() };
    let options = function.function_options();
    let aexpr = AExpr::Function {
        input: vec![ExprIR::from_node(node, arena)],
        function,
        options,
    };

    (arena.add(aexpr), pred)
}
