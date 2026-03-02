// TODO: Only hands out the next query ID. Polars has no other query ID
// concept — this could move to polars-stream.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::memory_manager::mm;

static QUERY_MANAGER: LazyLock<QueryManager> = LazyLock::new(QueryManager::default);

pub fn qm() -> &'static QueryManager {
    &QUERY_MANAGER
}

#[derive(Default)]
pub struct QueryManager {
    next_query_id: AtomicU64,
}

impl QueryManager {
    fn next_query_id(&self) -> u64 {
        self.next_query_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Begin a new query. Returns a [`QueryGuard`] that cleans up the query's
    /// spill directory when dropped.
    pub fn begin_query(&self) -> QueryGuard {
        let query_id = self.next_query_id();
        mm().spiller.begin_query(query_id);
        QueryGuard { query_id }
    }
}

/// RAII guard returned by [`QueryManager::begin_query`].
/// Cleans up the query's spill directory on drop.
pub struct QueryGuard {
    pub(crate) query_id: u64,
}

impl Drop for QueryGuard {
    fn drop(&mut self) {
        mm().spiller.delete_query_dir_background(self.query_id);
    }
}
