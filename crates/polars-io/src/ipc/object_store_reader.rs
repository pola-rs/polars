use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use polars_core::frame::DataFrame;
use polars_error::{to_compute_err, PolarsResult};

use crate::mmap::MmapBytesReader;
use crate::pl_async::with_concurrency_budget;
use crate::predicates::PhysicalIoExpr;
use crate::prelude::IpcReader;
use crate::{RowIndex, SerReader};

#[derive(Debug, Clone)]
enum Columns {
    Names(Vec<String>),
    Indices(Vec<usize>),
}

#[derive(Default, Clone)]
pub struct IpcReadOptions {
    columns: Option<Columns>,
    row_limit: Option<usize>,
    row_index: Option<RowIndex>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
}

impl IpcReadOptions {
    pub fn column_names(mut self, names: impl Into<Option<Vec<String>>>) -> Self {
        self.columns = names.into().map(Columns::Names);
        self
    }

    pub fn column_indices(mut self, indices: impl Into<Option<Vec<usize>>>) -> Self {
        self.columns = indices.into().map(Columns::Indices);
        self
    }

    pub fn row_limit(mut self, row_limit: impl Into<Option<usize>>) -> Self {
        self.row_limit = row_limit.into();
        self
    }

    pub fn row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.row_index = row_index.into();
        self
    }

    pub fn predicate(mut self, predicate: impl Into<Option<Arc<dyn PhysicalIoExpr>>>) -> Self {
        self.predicate = predicate.into();
        self
    }
}

pub async fn read_ipc_metadata_async<O: ObjectStore>(
    store: &O,
    path: &str,
    options: IpcReadOptions,
) -> PolarsResult<DataFrame> {
    let path = ObjectPath::parse(path).map_err(to_compute_err)?;

    let object_metadata = store.head(&path).await.map_err(to_compute_err)?;

    object_metadata.size;

    // TODO: Load only what is needed, rather than everything.
    let file_bytes = with_concurrency_budget(1, || read_bytes(store, &path)).await?;
    
    read_ipc(std::io::Cursor::new(file_bytes), options)
}

async fn read_bytes<O: ObjectStore>(store: &O, path: &ObjectPath) -> PolarsResult<Bytes> {
    // TODO: Is `to_compute_err` appropriate? It is used in the Parquet
    // reader as well but I am not sure it is what we want.
    let get_result = store.get(path).await.map_err(to_compute_err)?;

    // TODO: Perhaps use the streaming interface?
    let file_bytes = get_result.bytes().await.map_err(to_compute_err)?;

    PolarsResult::Ok(file_bytes)
}

fn read_ipc<R: MmapBytesReader>(reader: R, options: IpcReadOptions) -> PolarsResult<DataFrame> {
    let mut reader = IpcReader::new(reader);

    if let Some(columns) = options.columns {
        reader = match columns {
            Columns::Names(names) => reader.with_columns(Some(names)),
            Columns::Indices(indices) => reader.with_projection(Some(indices)),
        };
    }

    reader
        .with_n_rows(options.row_limit)
        .with_row_index(options.row_index)
        .finish_with_scan_ops(options.predicate, false)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use polars_core::df;

    use super::*;
    use crate::pl_async::get_runtime;
    use crate::prelude::IpcWriter;
    use crate::SerWriter;

    fn to_ipc_bytes(df: &mut DataFrame) -> Vec<u8> {
        let mut writer: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        IpcWriter::new(&mut writer).finish(df).unwrap();
        writer.into_inner()
    }

    fn write_ipc<T: ObjectStore>(store: &T, path: &str, df: &mut DataFrame) {
        get_runtime()
            .block_on(store.put(&ObjectPath::parse(path).unwrap(), to_ipc_bytes(df).into()))
            .unwrap();
    }

    #[test]
    fn read_ipc() {
        let mut df = df!("a" => [1], "b" => [2], "c" => [3]).unwrap();

        let store = object_store::memory::InMemory::new();
        write_ipc(&store, "data.ipc", &mut df);

        let actual_df = get_runtime()
            .block_on(read_ipc_metadata_async(
                &store,
                "data.ipc",
                IpcReadOptions::default().column_names(vec!["c".to_string(), "b".to_string()]),
            ))
            .unwrap();
        let expected_df = df!("c" => [3], "b" => [2]).unwrap();
        assert!(
            actual_df.equals(&expected_df),
            "expected {actual_df:?}\nto equal {expected_df:?}"
        );
    }
}
