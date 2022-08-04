use super::*;
use crate::mmap::MmapBytesReader;

impl<R: MmapBytesReader> IpcReader<R> {

    fn finish_memmapped(&self,
                     predicate: Option<Arc<dyn PhysicalIoExpr>>,
                     aggregate: Option<&[ScanAggregation]>,
    )

}