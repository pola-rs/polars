use pin_project_lite::pin_project;
use polars_utils::enum_unit_vec::EnumUnitVec;

use crate::async_executor::{AbortOnDropHandle, TaskPriority, spawn};

pin_project! {
    /// Represents a future that may either be local or spawned.
    #[project = LocalOrSpawnedFutureProj]
    pub enum LocalOrSpawnedFuture<F, O> {
        Local { #[pin] fut: F },
        Spawned { #[pin] handle: AbortOnDropHandle<O> }
    }
}

impl<F, O> LocalOrSpawnedFuture<F, O>
where
    F: Future<Output = O>,
{
    /// Wraps the future in a `Local` variant.
    pub fn new_local(fut: F) -> Self {
        LocalOrSpawnedFuture::Local { fut }
    }
}

impl<F, O> LocalOrSpawnedFuture<F, O>
where
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    /// Spawns the future onto the async executor.
    pub fn spawn(fut: F) -> Self {
        LocalOrSpawnedFuture::Spawned {
            handle: AbortOnDropHandle::new(spawn(TaskPriority::Low, fut)),
        }
    }
}

impl<F, O> Future for LocalOrSpawnedFuture<F, O>
where
    F: Future<Output = O>,
{
    type Output = O;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match self.project() {
            LocalOrSpawnedFutureProj::Local { fut } => fut.poll(cx),
            LocalOrSpawnedFutureProj::Spawned { handle } => handle.poll(cx),
        }
    }
}

/// Parallelizes futures across the computational async runtime.
///
/// As an optimization for cache access, the first future is kept on the current thread. If there
/// is only 1 future, then all data is kept on the current thread and spawn is not called at all.
///
/// Note this means the first future in the returned iterator does not run until polled.
///
/// Note that dropping the iterator will call abort on all spawned futures, as this is intended to be
/// used for compute.
pub fn parallelize_first_to_local<I, F, O>(
    futures_iter: I,
) -> impl ExactSizeIterator<Item = impl Future<Output = O> + Send + 'static>
where
    I: Iterator<Item = F>,
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    parallelize_first_to_local_impl(futures_iter).into_iter()
}

fn parallelize_first_to_local_impl<I, F, O>(
    mut futures_iter: I,
) -> EnumUnitVec<LocalOrSpawnedFuture<F, O>>
where
    I: Iterator<Item = F>,
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    let Some(first_fut) = futures_iter.next() else {
        return EnumUnitVec::new();
    };

    let first_fut = LocalOrSpawnedFuture::new_local(first_fut);

    let Some(second_fut) = futures_iter.next() else {
        return EnumUnitVec::new_single(first_fut);
    };

    let mut futures = Vec::with_capacity(2 + futures_iter.size_hint().0);

    // Note:
    // * The local future must come first to ensure we don't block polling it.
    // * Remaining futures must all be spawned upfront into the Vec for them to run parallel.
    futures.extend([first_fut, LocalOrSpawnedFuture::spawn(second_fut)]);
    futures.extend(futures_iter.map(LocalOrSpawnedFuture::spawn));

    EnumUnitVec::from(futures)
}
