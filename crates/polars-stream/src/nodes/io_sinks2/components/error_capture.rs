use std::any::Any;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use polars_error::{PolarsError, PolarsResult};

/// Utility to capture errors and propagate them to an associated [`ErrorHandle`].
#[derive(Clone)]
pub struct ErrorCapture {
    tx: tokio::sync::mpsc::Sender<ErrorMessage>,
}

impl ErrorCapture {
    pub fn new() -> (Self, ErrorHandle) {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        (Self { tx }, ErrorHandle { rx })
    }

    /// Wraps a future such that its error result is sent to the associated [`ErrorHandle`].
    pub async fn wrap_future<F, O>(self, fut: F)
    where
        F: Future<Output = PolarsResult<O>>,
    {
        let err: Result<(), tokio::sync::mpsc::error::TrySendError<ErrorMessage>> =
            match AssertUnwindSafe(fut).catch_unwind().await {
                Ok(Ok(_)) => return,
                Ok(Err(err)) => self.tx.try_send(ErrorMessage::Error(err)),
                Err(panic) => self.tx.try_send(ErrorMessage::Panic(panic)),
            };
        drop(err);
    }
}

enum ErrorMessage {
    Error(PolarsError),
    Panic(Box<dyn Any + Send + 'static>),
}

/// Handle to await the completion of multiple tasks. Propagates error results
/// and resumes unwinds when joined.
pub struct ErrorHandle {
    rx: tokio::sync::mpsc::Receiver<ErrorMessage>,
}

impl ErrorHandle {
    pub fn has_errored(&self) -> bool {
        !self.rx.is_empty()
    }

    /// Block until either an error is received, or all [`ErrorCapture`]s associated with this
    /// handle are dropped (i.e. successful completion of all wrapped futures).
    ///
    /// # Panics
    /// If a panic is received, this will resume unwinding.
    pub async fn join(self) -> PolarsResult<()> {
        let ErrorHandle { mut rx } = self;

        match rx.recv().await {
            None => Ok(()),
            Some(ErrorMessage::Error(e)) => Err(e),
            Some(ErrorMessage::Panic(panic)) => std::panic::resume_unwind(panic),
        }
    }
}
