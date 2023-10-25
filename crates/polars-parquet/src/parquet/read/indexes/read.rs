use std::convert::TryInto;
use std::io::{Cursor, Read, Seek, SeekFrom};

use parquet_format_safe::thrift::protocol::TCompactInputProtocol;
use parquet_format_safe::{ColumnChunk, OffsetIndex, PageLocation};

use super::deserialize::deserialize;
use crate::parquet::error::Error;
use crate::parquet::indexes::Index;
use crate::parquet::metadata::ColumnChunkMetaData;

fn prepare_read<F: Fn(&ColumnChunk) -> Option<i64>, G: Fn(&ColumnChunk) -> Option<i32>>(
    chunks: &[ColumnChunkMetaData],
    get_offset: F,
    get_length: G,
) -> Result<(u64, Vec<usize>), Error> {
    // c1: [start, length]
    // ...
    // cN: [start, length]

    let first_chunk = if let Some(chunk) = chunks.first() {
        chunk
    } else {
        return Ok((0, vec![]));
    };
    let metadata = first_chunk.column_chunk();

    let offset: u64 = if let Some(offset) = get_offset(metadata) {
        offset.try_into()?
    } else {
        return Ok((0, vec![]));
    };

    let lengths = chunks
        .iter()
        .map(|x| get_length(x.column_chunk()))
        .map(|maybe_length| {
            let index_length = maybe_length.ok_or_else(|| {
                Error::oos("The column length must exist if column offset exists")
            })?;

            Ok(index_length.try_into()?)
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok((offset, lengths))
}

fn prepare_column_index_read(chunks: &[ColumnChunkMetaData]) -> Result<(u64, Vec<usize>), Error> {
    prepare_read(chunks, |x| x.column_index_offset, |x| x.column_index_length)
}

fn prepare_offset_index_read(chunks: &[ColumnChunkMetaData]) -> Result<(u64, Vec<usize>), Error> {
    prepare_read(chunks, |x| x.offset_index_offset, |x| x.offset_index_length)
}

fn deserialize_column_indexes(
    chunks: &[ColumnChunkMetaData],
    data: &[u8],
    lengths: Vec<usize>,
) -> Result<Vec<Box<dyn Index>>, Error> {
    let mut start = 0;
    let data = lengths.into_iter().map(|length| {
        let r = &data[start..start + length];
        start += length;
        r
    });

    chunks
        .iter()
        .zip(data)
        .map(|(chunk, data)| {
            let primitive_type = chunk.descriptor().descriptor.primitive_type.clone();
            deserialize(data, primitive_type)
        })
        .collect()
}

/// Reads the column indexes of all [`ColumnChunkMetaData`] and deserializes them into [`Index`].
/// Returns an empty vector if indexes are not available
pub fn read_columns_indexes<R: Read + Seek>(
    reader: &mut R,
    chunks: &[ColumnChunkMetaData],
) -> Result<Vec<Box<dyn Index>>, Error> {
    let (offset, lengths) = prepare_column_index_read(chunks)?;

    let length = lengths.iter().sum::<usize>();

    reader.seek(SeekFrom::Start(offset))?;

    let mut data = vec![];
    data.try_reserve(length)?;
    reader.by_ref().take(length as u64).read_to_end(&mut data)?;

    deserialize_column_indexes(chunks, &data, lengths)
}

fn deserialize_page_locations(
    data: &[u8],
    column_number: usize,
) -> Result<Vec<Vec<PageLocation>>, Error> {
    let len = data.len() * 2 + 1024;
    let mut reader = Cursor::new(data);

    (0..column_number)
        .map(|_| {
            let mut prot = TCompactInputProtocol::new(&mut reader, len);
            let offset = OffsetIndex::read_from_in_protocol(&mut prot)?;
            Ok(offset.page_locations)
        })
        .collect()
}

/// Read [`PageLocation`]s from the [`ColumnChunkMetaData`]s.
/// Returns an empty vector if indexes are not available
pub fn read_pages_locations<R: Read + Seek>(
    reader: &mut R,
    chunks: &[ColumnChunkMetaData],
) -> Result<Vec<Vec<PageLocation>>, Error> {
    let (offset, lengths) = prepare_offset_index_read(chunks)?;

    let length = lengths.iter().sum::<usize>();

    reader.seek(SeekFrom::Start(offset))?;

    let mut data = vec![];
    data.try_reserve(length)?;
    reader.by_ref().take(length as u64).read_to_end(&mut data)?;

    deserialize_page_locations(&data, chunks.len())
}
