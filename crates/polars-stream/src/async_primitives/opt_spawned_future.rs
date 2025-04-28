use pin_project_lite::pin_project;

use crate::async_executor::{AbortOnDropHandle, TaskPriority, spawn};

pin_project! {
    /// Represents a potentially spawned future
    #[project = OptSpawnedFutureProj]
    pub enum OptSpawnedFuture<F, O> {
        Local { #[pin] fut: F },
        Spawned { #[pin] handle: AbortOnDropHandle<O> }
    }
}

impl<F, O> Future for OptSpawnedFuture<F, O>
where
    F: Future<Output = O>,
{
    type Output = O;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match self.project() {
            OptSpawnedFutureProj::Local { fut } => fut.poll(cx),
            OptSpawnedFutureProj::Spawned { handle } => handle.poll(cx),
        }
    }
}

/// Parallelizes an iterator of futures. The first future does not get spawned, and instead runs
/// on the current thread.
pub fn parallelize<I, F, O>(
    futures_iter: I,
    futures_iter_length: usize,
) -> Vec<OptSpawnedFuture<F, O>>
where
    I: IntoIterator<Item = F>,
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    if futures_iter_length == 0 {
        return vec![];
    }

    let mut futures = Vec::with_capacity(futures_iter_length);
    let mut futures_iter = futures_iter.into_iter();

    // The local future must come first to ensure we don't block polling it.
    futures.push(OptSpawnedFuture::Local {
        fut: futures_iter.next().unwrap(),
    });

    futures.extend((1..futures_iter_length).map(|_| OptSpawnedFuture::Spawned {
        handle: AbortOnDropHandle::new(spawn(TaskPriority::Low, futures_iter.next().unwrap())),
    }));

    futures
}
