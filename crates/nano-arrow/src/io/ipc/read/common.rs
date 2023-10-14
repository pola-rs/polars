use std::collections::VecDeque;
use std::io::{Read, Seek};

use ahash::AHashMap;
use arrow_format;
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::deserialize::{read, skip};
use super::Dictionaries;
use crate::array::*;
use crate::chunk::Chunk;
use crate::datatypes::{DataType, Field};
use crate::io::ipc::read::OutOfSpecKind;
use crate::io::ipc::{IpcField, IpcSchema};

#[derive(Debug, Eq, PartialEq, Hash)]
enum ProjectionResult<A> {
    Selected(A),
    NotSelected(A),
}

/// An iterator adapter that will return `Some(x)` or `None`
/// # Panics
/// The iterator panics iff the `projection` is not strictly increasing.
struct ProjectionIter<'a, A, I: Iterator<Item = A>> {
    projection: &'a [usize],
    iter: I,
    current_count: usize,
    current_projection: usize,
}

impl<'a, A, I: Iterator<Item = A>> ProjectionIter<'a, A, I> {
    /// # Panics
    /// iff `projection` is empty
    pub fn new(projection: &'a [usize], iter: I) -> Self {
        Self {
            projection: &projection[1..],
            iter,
            current_count: 0,
            current_projection: projection[0],
        }
    }
}

impl<'a, A, I: Iterator<Item = A>> Iterator for ProjectionIter<'a, A, I> {
    type Item = ProjectionResult<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.iter.next() {
            let result = if self.current_count == self.current_projection {
                if !self.projection.is_empty() {
                    assert!(self.projection[0] > self.current_projection);
                    self.current_projection = self.projection[0];
                    self.projection = &self.projection[1..];
                } else {
                    self.current_projection = 0 // a value that most likely already passed
                };
                Some(ProjectionResult::Selected(item))
            } else {
                Some(ProjectionResult::NotSelected(item))
            };
            self.current_count += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Returns a [`Chunk`] from a reader.
/// # Panic
/// Panics iff the projection is not in increasing order (e.g. `[1, 0]` nor `[0, 1, 1]` are valid)
#[allow(clippy::too_many_arguments)]
pub fn read_record_batch<R: Read + Seek>(
    batch: arrow_format::ipc::RecordBatchRef,
    fields: &[Field],
    ipc_schema: &IpcSchema,
    projection: Option<&[usize]>,
    limit: Option<usize>,
    dictionaries: &Dictionaries,
    version: arrow_format::ipc::MetadataVersion,
    reader: &mut R,
    block_offset: u64,
    file_size: u64,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Chunk<Box<dyn Array>>> {
    assert_eq!(fields.len(), ipc_schema.fields.len());
    let buffers = batch
        .buffers()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferBuffers(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageBuffers))?;
    let mut buffers: VecDeque<arrow_format::ipc::BufferRef> = buffers.iter().collect();

    // check that the sum of the sizes of all buffers is <= than the size of the file
    let buffers_size = buffers
        .iter()
        .map(|buffer| {
            let buffer_size: u64 = buffer
                .length()
                .try_into()
                .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
            Ok(buffer_size)
        })
        .sum::<PolarsResult<u64>>()?;
    if buffers_size > file_size {
        return Err(polars_err!(
            oos = OutOfSpecKind::InvalidBuffersLength {
                buffers_size,
                file_size,
            }
        ));
    }

    let field_nodes = batch
        .nodes()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferNodes(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageNodes))?;
    let mut field_nodes = field_nodes.iter().collect::<VecDeque<_>>();

    let columns = if let Some(projection) = projection {
        let projection =
            ProjectionIter::new(projection, fields.iter().zip(ipc_schema.fields.iter()));

        projection
            .map(|maybe_field| match maybe_field {
                ProjectionResult::Selected((field, ipc_field)) => Ok(Some(read(
                    &mut field_nodes,
                    field,
                    ipc_field,
                    &mut buffers,
                    reader,
                    dictionaries,
                    block_offset,
                    ipc_schema.is_little_endian,
                    batch.compression().map_err(|err| {
                        polars_err!(oos = OutOfSpecKind::InvalidFlatbufferCompression(err))
                    })?,
                    limit,
                    version,
                    scratch,
                )?)),
                ProjectionResult::NotSelected((field, _)) => {
                    skip(&mut field_nodes, &field.data_type, &mut buffers)?;
                    Ok(None)
                },
            })
            .filter_map(|x| x.transpose())
            .collect::<PolarsResult<Vec<_>>>()?
    } else {
        fields
            .iter()
            .zip(ipc_schema.fields.iter())
            .map(|(field, ipc_field)| {
                read(
                    &mut field_nodes,
                    field,
                    ipc_field,
                    &mut buffers,
                    reader,
                    dictionaries,
                    block_offset,
                    ipc_schema.is_little_endian,
                    batch.compression().map_err(|err| {
                        polars_err!(oos = OutOfSpecKind::InvalidFlatbufferCompression(err))
                    })?,
                    limit,
                    version,
                    scratch,
                )
            })
            .collect::<PolarsResult<Vec<_>>>()?
    };
    Chunk::try_new(columns)
}

fn find_first_dict_field_d<'a>(
    id: i64,
    data_type: &'a DataType,
    ipc_field: &'a IpcField,
) -> Option<(&'a Field, &'a IpcField)> {
    use DataType::*;
    match data_type {
        Dictionary(_, inner, _) => find_first_dict_field_d(id, inner.as_ref(), ipc_field),
        List(field) | LargeList(field) | FixedSizeList(field, ..) | Map(field, ..) => {
            find_first_dict_field(id, field.as_ref(), &ipc_field.fields[0])
        },
        Union(fields, ..) | Struct(fields) => {
            for (field, ipc_field) in fields.iter().zip(ipc_field.fields.iter()) {
                if let Some(f) = find_first_dict_field(id, field, ipc_field) {
                    return Some(f);
                }
            }
            None
        },
        _ => None,
    }
}

fn find_first_dict_field<'a>(
    id: i64,
    field: &'a Field,
    ipc_field: &'a IpcField,
) -> Option<(&'a Field, &'a IpcField)> {
    if let Some(field_id) = ipc_field.dictionary_id {
        if id == field_id {
            return Some((field, ipc_field));
        }
    }
    find_first_dict_field_d(id, &field.data_type, ipc_field)
}

pub(crate) fn first_dict_field<'a>(
    id: i64,
    fields: &'a [Field],
    ipc_fields: &'a [IpcField],
) -> PolarsResult<(&'a Field, &'a IpcField)> {
    assert_eq!(fields.len(), ipc_fields.len());
    for (field, ipc_field) in fields.iter().zip(ipc_fields.iter()) {
        if let Some(field) = find_first_dict_field(id, field, ipc_field) {
            return Ok(field);
        }
    }
    Err(polars_err!(
        oos = OutOfSpecKind::InvalidId { requested_id: id }
    ))
}

/// Reads a dictionary from the reader,
/// updating `dictionaries` with the resulting dictionary
#[allow(clippy::too_many_arguments)]
pub fn read_dictionary<R: Read + Seek>(
    batch: arrow_format::ipc::DictionaryBatchRef,
    fields: &[Field],
    ipc_schema: &IpcSchema,
    dictionaries: &mut Dictionaries,
    reader: &mut R,
    block_offset: u64,
    file_size: u64,
    scratch: &mut Vec<u8>,
) -> PolarsResult<()> {
    if batch
        .is_delta()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferIsDelta(err)))?
    {
        polars_bail!(ComputeError: "delta dictionary batches not supported")
    }

    let id = batch
        .id()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferId(err)))?;
    let (first_field, first_ipc_field) = first_dict_field(id, fields, &ipc_schema.fields)?;

    let batch = batch
        .data()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferData(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingData))?;

    let value_type =
        if let DataType::Dictionary(_, value_type, _) = first_field.data_type.to_logical_type() {
            value_type.as_ref()
        } else {
            polars_bail!(oos = OutOfSpecKind::InvalidIdDataType { requested_id: id })
        };

    // Make a fake schema for the dictionary batch.
    let fields = vec![Field::new("", value_type.clone(), false)];
    let ipc_schema = IpcSchema {
        fields: vec![first_ipc_field.clone()],
        is_little_endian: ipc_schema.is_little_endian,
    };
    let chunk = read_record_batch(
        batch,
        &fields,
        &ipc_schema,
        None,
        None, // we must read the whole dictionary
        dictionaries,
        arrow_format::ipc::MetadataVersion::V5,
        reader,
        block_offset,
        file_size,
        scratch,
    )?;

    dictionaries.insert(id, chunk.into_arrays().pop().unwrap());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_iter() {
        let iter = 1..6;
        let iter = ProjectionIter::new(&[0, 2, 4], iter);
        let result: Vec<_> = iter.collect();
        use ProjectionResult::*;
        assert_eq!(
            result,
            vec![
                Selected(1),
                NotSelected(2),
                Selected(3),
                NotSelected(4),
                Selected(5)
            ]
        )
    }
}

pub fn prepare_projection(
    fields: &[Field],
    mut projection: Vec<usize>,
) -> (Vec<usize>, AHashMap<usize, usize>, Vec<Field>) {
    let fields = projection.iter().map(|x| fields[*x].clone()).collect();

    // todo: find way to do this more efficiently
    let mut indices = (0..projection.len()).collect::<Vec<_>>();
    indices.sort_unstable_by_key(|&i| &projection[i]);
    let map = indices.iter().copied().enumerate().fold(
        AHashMap::default(),
        |mut acc, (index, new_index)| {
            acc.insert(index, new_index);
            acc
        },
    );
    projection.sort_unstable();

    // check unique
    if !projection.is_empty() {
        let mut previous = projection[0];

        for &i in &projection[1..] {
            assert!(
                previous < i,
                "The projection on IPC must not contain duplicates"
            );
            previous = i;
        }
    }

    (projection, map, fields)
}

pub fn apply_projection(
    chunk: Chunk<Box<dyn Array>>,
    map: &AHashMap<usize, usize>,
) -> Chunk<Box<dyn Array>> {
    // re-order according to projection
    let arrays = chunk.into_arrays();
    let mut new_arrays = arrays.clone();

    map.iter()
        .for_each(|(old, new)| new_arrays[*new] = arrays[*old].clone());

    Chunk::new(new_arrays)
}
