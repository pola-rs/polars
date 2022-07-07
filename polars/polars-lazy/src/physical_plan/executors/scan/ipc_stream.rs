use super::*;
use crate::prelude::file_caching::FileFingerPrint;

pub struct IpcStreamExec {
    pub(crate) path: PathBuf,
    pub(crate) schema: SchemaRef,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) aggregate: Vec<ScanAggregation>,
    pub(crate) options: IpcStreamScanOptionsInner,
}

impl IpcStreamExec {
    fn read(&mut self) -> Result<DataFrame> {
        let (file, projection, n_rows, aggregate, predicate) = prepare_scan_args(
            &self.path,
            &self.predicate,
            &mut self.options.with_columns,
            &mut self.schema,
            self.options.n_rows,
            &self.aggregate,
        );
        IpcStreamReader::new(file)
            .with_n_rows(n_rows)
            .with_row_count(std::mem::take(&mut self.options.row_count))
            .set_rechunk(self.options.rechunk)
            .finish_with_scan_ops(predicate, aggregate, projection)
    }
}

impl Executor for IpcStreamExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let finger_print = FileFingerPrint {
            path: self.path.clone(),
            predicate: self.predicate.as_ref().map(|ae| ae.as_expression().clone()),
            slice: (0, self.options.n_rows),
        };
        state
            .file_cache
            .read(finger_print, self.options.file_counter, &mut || self.read())
    }
}
