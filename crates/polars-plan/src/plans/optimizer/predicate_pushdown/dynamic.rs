use std::fmt::{Debug, Formatter};
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;

use polars_utils::unique_id::UniqueId;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
struct Inner {
    pred: RwLock<Option<Node>>,
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

impl DynamicPred {
    pub fn id(&self) -> &UniqueId {
        &self.inner.id
    }

    fn set(&self, node: Node) {
        let mut guard = self.inner.pred.write().unwrap();
        *guard = Some(node);
    }
}
