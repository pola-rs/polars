use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Calls [`tokio::task::JoinHandle::abort`] on the join handle when dropped.
pub struct AbortOnDropHandle<T>(pub tokio::task::JoinHandle<T>);

impl<T> Future for AbortOnDropHandle<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx)
    }
}

impl<T> Drop for AbortOnDropHandle<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}
