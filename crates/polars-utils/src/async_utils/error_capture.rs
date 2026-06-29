use std::any::Any;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;

/// Utility to capture errors and propagate them to an associated [`ErrorHandle`].
pub struct ErrorCapture<ErrorT> {
    tx: tokio::sync::mpsc::Sender<ErrorMessage<ErrorT>>,
}

impl<ErrorT> Clone for ErrorCapture<ErrorT> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<ErrorT> ErrorCapture<ErrorT> {
    pub fn new() -> (Self, ErrorHandle<ErrorT>) {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        (Self { tx }, ErrorHandle { rx })
    }

    /// Wraps a future such that its error result is sent to the associated [`ErrorHandle`].
    pub async fn wrap_future<F, O>(self, fut: F)
    where
        F: Future<Output = Result<O, ErrorT>>,
    {
        let err: Result<(), tokio::sync::mpsc::error::TrySendError<ErrorMessage<ErrorT>>> =
            match AssertUnwindSafe(fut).catch_unwind().await {
                Ok(Ok(_)) => return,
                Ok(Err(err)) => self.tx.try_send(ErrorMessage::Error(err)),
                Err(panic) => self.tx.try_send(ErrorMessage::Panic(panic)),
            };
        drop(err);
    }
}

enum ErrorMessage<ErrorT> {
    Error(ErrorT),
    Panic(Box<dyn Any + Send + 'static>),
}

/// Handle to await the completion of multiple tasks. Propagates error results
/// and resumes unwinds when joined.
pub struct ErrorHandle<ErrorT> {
    rx: tokio::sync::mpsc::Receiver<ErrorMessage<ErrorT>>,
}

impl<ErrorT> ErrorHandle<ErrorT> {
    pub fn has_errored(&self) -> bool {
        !self.rx.is_empty()
    }

    /// Block until either an error is received, or all [`ErrorCapture`]s associated with this
    /// handle are dropped (i.e. successful completion of all wrapped futures).
    ///
    /// # Panics
    /// If a panic is received, this will resume unwinding.
    pub async fn join(self) -> Result<(), ErrorT> {
        let ErrorHandle { mut rx } = self;

        match rx.recv().await {
            None => Ok(()),
            Some(ErrorMessage::Error(e)) => Err(e),
            Some(ErrorMessage::Panic(panic)) => std::panic::resume_unwind(panic),
        }
    }
}
