use pin_project_lite::pin_project;

use crate::async_executor::{AbortOnDropHandle, TaskPriority, spawn};

pin_project! {
    /// Represents a potentially spawned future.
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

/// Parallelizes an iterator of futures across the computational async runtime.
///
/// As an optimization for cache access, the first future is kept on the current thread. If there
/// is only 1 future, then all data is kept on the current thread and spawn is not called at all.
///
/// Note this means the first future in the returned Vec does not run until polled.
///
/// Note that dropping the Vec will call abort on all spawned futures, as this is intended to be
/// used for compute.
///
/// # Panics
/// Panics if the iterator has less than `futures_iter_length` items.
pub fn parallelize_first_to_local<I, F, O>(mut futures_iter: I) -> Vec<OptSpawnedFuture<F, O>>
where
    I: Iterator<Item = F>,
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    let mut futures = Vec::with_capacity(futures_iter.size_hint().1.unwrap_or(0));

    let Some(first_fut) = futures_iter.next() else {
        return futures;
    };

    // The local future must come first to ensure we don't block polling it.
    futures.push(OptSpawnedFuture::Local { fut: first_fut });

    futures.extend(futures_iter.map(|fut| OptSpawnedFuture::Spawned {
        handle: AbortOnDropHandle::new(spawn(TaskPriority::Low, fut)),
    }));

    futures
}
