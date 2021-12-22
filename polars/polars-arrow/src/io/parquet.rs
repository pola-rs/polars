use arrow::datatypes::Schema;
use arrow::io::parquet::read::FileMetaData;
use arrow::{array::ArrayRef, error::Result, io::parquet::read};
use std::borrow::Cow;
use std::io::{Read, Seek};
use std::sync::Arc;

pub fn read_parquet<R: Read + Seek>(
    mut reader: R,
    limit: usize,
    projection: Option<&[usize]>,
    schema: &Schema,
    metadata: Option<FileMetaData>,
) -> Result<Vec<Vec<ArrayRef>>> {
    let metadata = metadata
        .map(Ok)
        .unwrap_or_else(|| read::read_metadata(&mut reader))?;
    let row_group_len = metadata.row_groups.len();

    let projection = projection
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned((0usize..schema.fields().len()).collect::<Vec<_>>()));

    let mut rb = Vec::with_capacity(row_group_len);

    let mut buf_1 = Vec::with_capacity(1024);
    let mut buf_2 = Vec::with_capacity(1024);

    let mut remaining_rows = limit;

    for rg in 0..row_group_len {
        let arrs = projection
            .clone()
            .iter()
            .map(|column_i| {
                let b1 = std::mem::take(&mut buf_1);
                let b2 = std::mem::take(&mut buf_2);

                // the get_column_iterator is an iterator of columns, each column contains compressed pages.
                // get_column_iterator yields `Vec<Vec<CompressedPage>>`:
                // outer `Vec` is len 1 for primitive types,
                // inner `Vec` is whatever number of pages the chunk contains.
                let column_iter =
                    read::get_column_iterator(&mut reader, &metadata, rg, *column_i, None, b1);
                let fld = schema.field(*column_i);
                let (mut array, b1, b2) = read::column_iter_to_array(column_iter, fld, b2)?;

                if array.len() > remaining_rows {
                    array = array.slice(0, remaining_rows);
                }

                buf_1 = b1;
                buf_2 = b2;

                Ok(Arc::from(array))
            })
            .collect::<Result<Vec<_>>>()?;

        remaining_rows = metadata.row_groups[rg].num_rows() as usize;
        rb.push(arrs)
    }

    Ok(rb)
}
