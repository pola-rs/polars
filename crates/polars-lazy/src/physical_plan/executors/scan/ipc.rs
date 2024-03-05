use std::path::PathBuf;

use polars_core::config::env_force_async;
use polars_io::cloud::build_object_store;
use polars_io::is_cloud_url;
use polars_io::pl_async::get_runtime;

use super::*;

pub struct IpcExec {
    pub(crate) path: PathBuf,
    pub(crate) schema: SchemaRef,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) options: IpcScanOptions,
    pub(crate) file_options: FileScanOptions,
}

impl IpcExec {
    fn read(&mut self, verbose: bool) -> PolarsResult<DataFrame> {
        if is_cloud_url(&self.path) || env_force_async() {
            #[cfg(not(feature = "cloud"))]
            {
                panic!("activate cloud feature")
            }

            #[cfg(feature = "cloud")]
            {
                // TODO: This will block the current thread until the data has been
                // loaded. No other (compute) work can be done on this thread.
                //
                // What is our work scheduling strategy? From what I have heard, we do
                // not intend to make the entire library `async`. However, what do we do
                // instead?
                return get_runtime().block_on(async move {
                    let (location, store) =
                        build_object_store(self.path.to_str().unwrap(), None).await?;
                    debug_assert!(
                        location.expansion.is_none(),
                        "path wildcards should have been expanded"
                    );
                    read_ipc_async(&store, location.prefix.as_str(), IpcReadOptions::default()
                        .column_names(self.file_options.with_columns.as_deref().cloned())
                        .row_index(self.file_options.row_index.clone())
                        .row_limit(self.file_options.n_rows),
                    )
                        .await
                });
            }
        }

        let (projection, predicate) = prepare_scan_args(
            self.predicate.clone(),
            &mut self.file_options.with_columns,
            &mut self.schema,
            self.file_options.row_index.is_some(),
            None,
        );

        let file = std::fs::File::open(&self.path)?;
        IpcReader::new(file)
            .with_n_rows(self.file_options.n_rows)
            .with_row_index(std::mem::take(&mut self.file_options.row_index))
            .set_rechunk(self.file_options.rechunk)
            .with_projection(projection)
            .memory_mapped(self.options.memmap)
            .finish_with_scan_ops(predicate, verbose)
    }
}

impl Executor for IpcExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let finger_print = FileFingerPrint {
            paths: Arc::new([self.path.clone()]),
            #[allow(clippy::useless_asref)]
            predicate: self
                .predicate
                .as_ref()
                .map(|ae| ae.as_expression().unwrap().clone()),
            slice: (0, self.file_options.n_rows),
        };

        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.path.to_string_lossy().into()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("ipc".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                state
                    .file_cache
                    .read(finger_print, self.file_options.file_counter, &mut || {
                        self.read(state.verbose())
                    })
            },
            profile_name,
        )
    }
}
