use super::*;
use crate::prelude::file_caching::FileFingerPrint;

pub struct IpcExec {
    pub(crate) path: PathBuf,
    pub(crate) schema: SchemaRef,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) aggregate: Vec<ScanAggregation>,
    pub(crate) options: IpcScanOptionsInner,
}

impl IpcExec {
    fn read(&mut self, verbose: bool) -> Result<DataFrame> {
        let (file, projection, n_rows, aggregate, predicate) = prepare_scan_args(
            &self.path,
            &self.predicate,
            &mut self.options.with_columns,
            &mut self.schema,
            self.options.n_rows,
            &self.aggregate,
        );
        IpcReader::new(file)
            .with_n_rows(n_rows)
            .with_row_count(std::mem::take(&mut self.options.row_count))
            .set_rechunk(self.options.rechunk)
            .with_projection(projection)
            .memory_mapped(self.options.memmap)
            .finish_with_scan_ops(predicate, aggregate, verbose)
    }
}

impl Executor for IpcExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        let finger_print = FileFingerPrint {
            path: self.path.clone(),
            predicate: self
                .predicate
                .as_ref()
                .map(|ae| ae.as_expression().unwrap().clone()),
            slice: (0, self.options.n_rows),
        };
        state
            .file_cache
            .read(finger_print, self.options.file_counter, &mut || {
                self.read(state.verbose())
            })
    }
}
