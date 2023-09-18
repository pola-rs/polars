use super::{AsyncReader, ByteRecord};
use ahash::AHashSet;

use crate::datatypes::{DataType, Field};
use crate::error::Result;
use crate::io::csv::utils::merge_schema;

use futures::{AsyncRead, AsyncSeek};

/// Infers the [`Field`]s of a CSV file by reading through the first n records up to `max_rows`.
/// Seeks back to the begining of the file _after_ the header
pub async fn infer_schema<R, F>(
    reader: &mut AsyncReader<R>,
    max_rows: Option<usize>,
    has_header: bool,
    infer: &F,
) -> Result<(Vec<Field>, usize)>
where
    R: AsyncRead + AsyncSeek + Unpin + Send + Sync,
    F: Fn(&[u8]) -> DataType,
{
    // get or create header names
    // when has_header is false, creates default column names with column_ prefix
    let headers: Vec<String> = if has_header {
        reader
            .headers()
            .await?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        let first_record_count = &reader.headers().await?.len();
        (0..*first_record_count)
            .map(|i| format!("column_{}", i + 1))
            .collect()
    };

    // save the csv reader position after reading headers
    let position = reader.position().clone();

    let header_length = headers.len();
    // keep track of inferred field types
    let mut column_types: Vec<AHashSet<DataType>> = vec![AHashSet::new(); header_length];

    let mut records_count = 0;

    let mut record = ByteRecord::new();
    let max_records = max_rows.unwrap_or(usize::MAX);
    while records_count < max_records {
        if !reader.read_byte_record(&mut record).await? {
            break;
        }
        records_count += 1;

        for (i, column) in column_types.iter_mut().enumerate() {
            if let Some(string) = record.get(i) {
                column.insert(infer(string));
            }
        }
    }

    let fields = merge_schema(&headers, &mut column_types);

    // return the reader seek back to the start
    reader.seek(position).await?;

    Ok((fields, records_count))
}
