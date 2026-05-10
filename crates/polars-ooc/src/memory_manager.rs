use std::sync::{Arc, LazyLock, RwLock, Weak};

use crate::SpillContext;

static MEMORY_MANAGER: LazyLock<MemoryManager> = LazyLock::new(MemoryManager::new);

/// Return a reference to the global [`MemoryManager`].
pub fn memory_manager() -> &'static MemoryManager {
    &MEMORY_MANAGER
}

pub struct MemoryManager {
    contexts: RwLock<Vec<Weak<dyn SpillContext>>>,
}

impl MemoryManager {
    fn new() -> Self {
        Self {
            contexts: RwLock::new(Vec::new()),
        }
    }

    pub fn register_ctx<C: SpillContext>(&self, ctx: &Arc<C>) {
        let weak = Arc::downgrade(ctx);
        self.contexts.write().unwrap().push(weak);
    }

    pub async fn spill(&self) {}

    pub fn spill_blocking(&self) {}
}
