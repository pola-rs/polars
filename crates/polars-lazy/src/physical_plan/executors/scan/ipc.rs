use std::path::PathBuf;

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
}

impl Executor for IpcExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let finger_print = FileFingerPrint {
            paths: Arc::new([self.path.clone()]),
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
