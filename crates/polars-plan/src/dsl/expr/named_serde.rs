use std::sync::{Arc, LazyLock, RwLock};

use super::ColumnsUdf;

pub trait ExprRegistry: Sync + Send {
    fn get_function(&self, name: &str) -> Option<Arc<dyn ColumnsUdf>>;
}

pub(super) static NAMED_SERDE_REGISTRY_EXPR: LazyLock<RwLock<Option<Box<dyn ExprRegistry>>>> =
    LazyLock::new(Default::default);
