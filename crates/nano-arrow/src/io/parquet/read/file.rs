use std::io::{Read, Seek};

use parquet2::indexes::FilteredPage;

use crate::array::Array;
use crate::chunk::Chunk;
use crate::datatypes::Schema;
use crate::error::Result;
use crate::io::parquet::read::read_columns_many;

use super::{RowGroupDeserializer, RowGroupMetaData};

/// An iterator of [`Chunk`]s coming from row groups of a parquet file.
///
/// This can be thought of a flatten chain of [`Iterator<Item=Chunk>`] - each row group is sequentially
/// mapped to an [`Iterator<Item=Chunk>`] and each iterator is iterated upon until either the limit
/// or the last iterator ends.
/// # Implementation
/// This iterator is single threaded on both IO-bounded and CPU-bounded tasks, and mixes them.
pub struct FileReader<R: Read + Seek> {
    row_groups: RowGroupReader<R>,
    remaining_rows: usize,
    current_row_group: Option<RowGroupDeserializer>,
}

impl<R: Read + Seek> FileReader<R> {
    /// Returns a new [`FileReader`].
    pub fn new(
        reader: R,
        row_groups: Vec<RowGroupMetaData>,
        schema: Schema,
        chunk_size: Option<usize>,
        limit: Option<usize>,
        page_indexes: Option<Vec<Vec<Vec<Vec<FilteredPage>>>>>,
    ) -> Self {
        let row_groups =
            RowGroupReader::new(reader, schema, row_groups, chunk_size, limit, page_indexes);

        Self {
            row_groups,
            remaining_rows: limit.unwrap_or(usize::MAX),
            current_row_group: None,
        }
    }

    fn next_row_group(&mut self) -> Result<Option<RowGroupDeserializer>> {
        let result = self.row_groups.next().transpose()?;

        // If current_row_group is None, then there will be no elements to remove.
        if self.current_row_group.is_some() {
            self.remaining_rows = self.remaining_rows.saturating_sub(
                result
                    .as_ref()
                    .map(|x| x.num_rows())
                    .unwrap_or(self.remaining_rows),
            );
        }
        Ok(result)
    }

    /// Returns the [`Schema`] associated to this file.
    pub fn schema(&self) -> &Schema {
        &self.row_groups.schema
    }
}

impl<R: Read + Seek> Iterator for FileReader<R> {
    type Item = Result<Chunk<Box<dyn Array>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_rows == 0 {
            // reached the limit
            return None;
        }

        if let Some(row_group) = &mut self.current_row_group {
            match row_group.next() {
                // no more chunks in the current row group => try a new one
                None => match self.next_row_group() {
                    Ok(Some(row_group)) => {
                        self.current_row_group = Some(row_group);
                        // new found => pull again
                        self.next()
                    }
                    Ok(None) => {
                        self.current_row_group = None;
                        None
                    }
                    Err(e) => Some(Err(e)),
                },
                other => other,
            }
        } else {
            match self.next_row_group() {
                Ok(Some(row_group)) => {
                    self.current_row_group = Some(row_group);
                    self.next()
                }
                Ok(None) => {
                    self.current_row_group = None;
                    None
                }
                Err(e) => Some(Err(e)),
            }
        }
    }
}

/// An [`Iterator<Item=RowGroupDeserializer>`] from row groups of a parquet file.
///
/// # Implementation
/// Advancing this iterator is IO-bounded - each iteration reads all the column chunks from the file
/// to memory and attaches [`RowGroupDeserializer`] to them so that they can be iterated in chunks.
pub struct RowGroupReader<R: Read + Seek> {
    reader: R,
    schema: Schema,
    row_groups: std::vec::IntoIter<RowGroupMetaData>,
    chunk_size: Option<usize>,
    remaining_rows: usize,
    page_indexes: Option<std::vec::IntoIter<Vec<Vec<Vec<FilteredPage>>>>>,
}

impl<R: Read + Seek> RowGroupReader<R> {
    /// Returns a new [`RowGroupReader`]
    pub fn new(
        reader: R,
        schema: Schema,
        row_groups: Vec<RowGroupMetaData>,
        chunk_size: Option<usize>,
        limit: Option<usize>,
        page_indexes: Option<Vec<Vec<Vec<Vec<FilteredPage>>>>>,
    ) -> Self {
        if let Some(pages) = &page_indexes {
            assert_eq!(pages.len(), row_groups.len())
        }
        Self {
            reader,
            schema,
            row_groups: row_groups.into_iter(),
            chunk_size,
            remaining_rows: limit.unwrap_or(usize::MAX),
            page_indexes: page_indexes.map(|pages| pages.into_iter()),
        }
    }

    #[inline]
    fn _next(&mut self) -> Result<Option<RowGroupDeserializer>> {
        if self.schema.fields.is_empty() {
            return Ok(None);
        }
        if self.remaining_rows == 0 {
            // reached the limit
            return Ok(None);
        }

        let row_group = if let Some(row_group) = self.row_groups.next() {
            row_group
        } else {
            return Ok(None);
        };

        let pages = self.page_indexes.as_mut().and_then(|iter| iter.next());

        // the number of rows depends on whether indexes are selected or not.
        let num_rows = pages
            .as_ref()
            .map(|x| {
                // first field, first column within that field
                x[0][0]
                    .iter()
                    .map(|page| {
                        page.selected_rows
                            .iter()
                            .map(|interval| interval.length)
                            .sum::<usize>()
                    })
                    .sum()
            })
            .unwrap_or_else(|| row_group.num_rows());

        let column_chunks = read_columns_many(
            &mut self.reader,
            &row_group,
            self.schema.fields.clone(),
            self.chunk_size,
            Some(self.remaining_rows),
            pages,
        )?;

        let result = RowGroupDeserializer::new(column_chunks, num_rows, Some(self.remaining_rows));
        self.remaining_rows = self.remaining_rows.saturating_sub(num_rows);
        Ok(Some(result))
    }
}

impl<R: Read + Seek> Iterator for RowGroupReader<R> {
    type Item = Result<RowGroupDeserializer>;

    fn next(&mut self) -> Option<Self::Item> {
        self._next().transpose()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.row_groups.size_hint()
    }
}
