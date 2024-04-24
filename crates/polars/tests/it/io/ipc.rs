use std::io::{Read, Seek, SeekFrom};

#[cfg(feature = "flight")]
use arrow::io::ipc::read::FlightAsyncRawReader;
use arrow::io::ipc::read::FlightStreamReader;
use futures::io::Cursor;
use futures::{Stream, StreamExt};
use polars::prelude::*;
use polars_core::assert_df_eq;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

#[test]
fn test_ipc_compression_variadic_buffers() {
    let mut df = df![
        "foo" => std::iter::repeat("Home delivery vat 24 %").take(3).collect::<Vec<_>>()
    ]
    .unwrap();

    let mut file = std::io::Cursor::new(vec![]);
    IpcWriter::new(&mut file)
        .with_compression(Some(IpcCompression::LZ4))
        .with_pl_flavor(true)
        .finish(&mut df)
        .unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();
    let out = IpcReader::new(file).finish().unwrap();

    assert_eq!(out.shape(), (3, 1));
}

#[cfg(feature = "dtype-categorical")]
fn create_multi_chunk_df() -> DataFrame {
    let s1 = Series::new(
        "foo",
        std::iter::repeat("Home delivery vat 24 %")
            .take(3)
            .collect::<Vec<_>>(),
    );
    let cat_builder = CategoricalChunkedBuilder::new("bar", 10, CategoricalOrdering::Physical);
    let s2 = cat_builder
        .drain_iter_and_finish(vec![Some("B"), Some("C"), Some("B")])
        .into_series();

    let df = DataFrame::new(vec![s1, s2]).unwrap();
    df.vstack(&df.clone()).unwrap()
}

#[cfg(all(feature = "flight", feature = "dtype-categorical"))]
#[tokio::test]
async fn test_ipc_flight_round_trip() -> PolarsResult<()> {
    // Roundtrip test DF -> IPC -> Flight Async Reader -> Flight Stream Adapter -> DF

    let mut df = create_multi_chunk_df();
    let mut file = std::io::Cursor::new(vec![]);
    IpcWriter::new(&mut file)
        .with_pl_flavor(true)
        .finish(&mut df)
        .unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();

    let mut ipc_mem = vec![];
    let _ = file.read_to_end(&mut ipc_mem)?;

    let async_cursor = Cursor::new(ipc_mem);

    // Read in raw bytes in streaming format
    let stream = FlightAsyncRawReader::stream(async_cursor);
    tokio::pin!(stream);

    let mut flight_stream = FlightStreamReader::default();
    let mut dfs = Vec::with_capacity(stream.size_hint().0);
    let mut schema: Option<ArrowSchema> = None;

    while let Some(message) = stream.next().await {
        let message = message?;

        let chunk = flight_stream.parse(message)?;

        // First one should be a schema
        if schema.is_none() {
            polars_ensure!(chunk.is_none(), ComputeError: "First message should be a schema, but parse returned a record batch");
            schema = Some(flight_stream.get_metadata()?.clone());
        } else if let Some(chunk) = chunk {
            let fields = schema.as_ref().unwrap().fields.as_slice();
            dfs.push(DataFrame::try_from((chunk, fields))?);
        }
    }

    let df_out = accumulate_dataframes_vertical_unchecked(dfs);
    assert_df_eq!(&df, &df_out);
    Ok(())
}
