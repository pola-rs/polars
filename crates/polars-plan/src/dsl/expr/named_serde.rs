use std::sync::{Arc, LazyLock, RwLock};

use super::{ColumnsUdf, FunctionOutputField};

// Can be used to have named anonymous functions.
// The receiver must have implemented this registry and map the names to the proper UDFs.
pub trait ExprRegistry: Sync + Send {
    fn get_function(&self, name: &str, payload: &[u8]) -> Option<Arc<dyn ColumnsUdf>>;

    fn get_output(&self, name: &str) -> Option<Arc<dyn FunctionOutputField>>;
}

pub(super) static NAMED_SERDE_REGISTRY_EXPR: LazyLock<RwLock<Option<Arc<dyn ExprRegistry>>>> =
    LazyLock::new(Default::default);

pub fn set_named_serde_registry(reg: Arc<dyn ExprRegistry>) {
    let mut lock = NAMED_SERDE_REGISTRY_EXPR.write().unwrap();
    *lock = Some(reg);
}
