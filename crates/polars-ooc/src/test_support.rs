//! Helpers for tests, here and in crates that spill, which put the memory manager over budget.

use polars_config::config;

/// Tests that change global spill or allocator state run one at a time.
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

const OVER_BUDGET_VARS: [(&str, &str); 1] = [("POLARS_OOC_MEMORY_BUDGET_MB", "1")];
const OVER_BUDGET_USAGE: i64 = 16 << 20;

/// Puts the memory manager over budget until dropped, also if the test panics. Hold
/// [`test_lock`] meanwhile.
pub struct OverBudget;

impl OverBudget {
    pub fn new() -> Self {
        for (var, val) in OVER_BUDGET_VARS {
            config().set_env_var_override(var, Some(val));
        }
        crate::global_alloc::add_test_usage(OVER_BUDGET_USAGE);
        OverBudget
    }
}

impl Default for OverBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for OverBudget {
    fn drop(&mut self) {
        crate::global_alloc::add_test_usage(-OVER_BUDGET_USAGE);
        for (var, _) in OVER_BUDGET_VARS {
            config().reload_env_var(var);
        }
    }
}
