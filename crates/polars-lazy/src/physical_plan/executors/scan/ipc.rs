use std::path::PathBuf;

use polars_core::config::env_force_async;
#[cfg(feature = "cloud")]
use polars_io::cloud::CloudOptions;
use polars_io::is_cloud_url;

use super::*;

pub struct IpcExec {
    pub(crate) path: PathBuf,
    pub(crate) schema: SchemaRef,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) options: IpcScanOptions,
    pub(crate) file_options: FileScanOptions,
    #[cfg(feature = "cloud")]
    pub(crate) cloud_options: Option<CloudOptions>,
    pub(crate) metadata: Option<arrow::io::ipc::read::FileMetadata>,
}

impl IpcExec {
    fn read(&mut self, verbose: bool) -> PolarsResult<DataFrame> {
        let is_cloud = is_cloud_url(&self.path);
        let force_async = env_force_async();

        let mut out = if is_cloud || force_async {
            #[cfg(not(feature = "cloud"))]
            {
                panic!("activate cloud feature")
            }

            #[cfg(feature = "cloud")]
            {
                if !is_cloud && verbose {
                    eprintln!("ASYNC READING FORCED");
                }

                polars_io::pl_async::get_runtime()
                    .block_on_potential_spawn(self.read_async(verbose))?
            }
        } else {
            self.read_sync(verbose)?
        };

        if self.file_options.rechunk {
            out.as_single_chunk_par();
        }

        Ok(out)
    }

    fn read_sync(&mut self, verbose: bool) -> PolarsResult<DataFrame> {
        let file = std::fs::File::open(&self.path)?;
        let (projection, predicate) = prepare_scan_args(
            self.predicate.clone(),
            &mut self.file_options.with_columns,
            &mut self.schema,
            self.file_options.row_index.is_some(),
            None,
        );
        IpcReader::new(file)
            .with_n_rows(self.file_options.n_rows)
            .with_row_index(std::mem::take(&mut self.file_options.row_index))
            .set_rechunk(self.file_options.rechunk)
            .with_projection(projection)
            .memory_mapped(self.options.memmap)
            .finish_with_scan_ops(predicate, verbose)
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self, verbose: bool) -> PolarsResult<DataFrame> {
        let predicate = self.predicate.clone().map(phys_expr_to_io_expr);

        let reader =
            IpcReaderAsync::from_uri(self.path.to_str().unwrap(), self.cloud_options.as_ref())
                .await?;
        reader
            .data(
                self.metadata.as_ref(),
                IpcReadOptions::default()
                    .with_row_limit(self.file_options.n_rows)
                    .with_row_index(self.file_options.row_index.clone())
                    .with_projection(self.file_options.with_columns.as_deref().cloned())
                    .with_predicate(predicate),
                verbose,
            )
            .await
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
