use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::async_primitives::wait_group::WaitToken;

/// The outcome of a phase in a task.
///
/// This indicates whether a task finished (and does not need to be started again) or has stopped
/// prematurely. When this is dropped without calling `stop`, it is assumed that the task is
/// finished (most likely because it errored).
pub struct PhaseOutcome {
    // This is used to see when phase is finished.
    #[expect(unused)]
    consume_token: WaitToken,

    outcome_token: PhaseOutcomeToken,
}

impl PhaseOutcome {
    pub fn new_shared_wait(consume_token: WaitToken) -> (PhaseOutcomeToken, Self) {
        let outcome_token = PhaseOutcomeToken::new();
        (
            outcome_token.clone(),
            Self {
                consume_token,
                outcome_token,
            },
        )
    }

    /// Phase ended before the task finished and needs to be called again.
    pub fn stopped(self) {
        self.outcome_token.stop();
    }
}

/// Token that contains the outcome of a phase.
///
/// Namely, this indicates whether a phase finished completely or whether it was stopped before
/// that.
#[derive(Clone)]
pub struct PhaseOutcomeToken {
    /// - `false` -> finished / panicked
    /// - `true` -> stopped before finishing
    stop: Arc<AtomicBool>,
}

impl PhaseOutcomeToken {
    pub fn new() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Indicate that the phase was stopped before finishing.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Returns whether the phase was stopped before finishing.
    pub fn was_stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Returns whether the phase was finished completely.
    pub fn did_finish(&self) -> bool {
        !self.was_stopped()
    }
}
