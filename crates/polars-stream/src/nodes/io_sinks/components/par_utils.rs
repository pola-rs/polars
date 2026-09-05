use polars_async::executor::TaskPriority;
use polars_async::primitives::opt_spawned_future::parallelize_first_to_local;
use polars_core::prelude::Column;

/// Parallel rechunk of each column over the computational async executor.
pub async fn rechunk_par(columns: &mut [Column]) {
    for fut in parallelize_first_to_local(
        TaskPriority::Low,
        columns.iter_mut().enumerate().filter_map(|(i, c)| {
            (c.n_chunks() > 1).then(|| {
                let c = std::mem::take(c);
                async move { (i, c.rechunk()) }
            })
        }),
    ) {
        let (i, c) = fut.await;
        columns[i] = c;
    }
}
