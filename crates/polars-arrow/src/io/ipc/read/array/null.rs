use std::collections::VecDeque;

use polars_error::{polars_err, PolarsResult};

use super::super::Node;
use crate::array::NullArray;
use crate::datatypes::ArrowDataType;
use crate::io::ipc::read::array::{try_get_array_length, try_get_field_node};

pub fn read_null(
    field_nodes: &mut VecDeque<Node>,
    data_type: ArrowDataType,
    limit: Option<usize>,
) -> PolarsResult<NullArray> {
    let field_node = try_get_field_node(field_nodes, &data_type)?;

    let length = try_get_array_length(field_node, limit)?;

    NullArray::try_new(data_type, length)
}

pub fn skip_null(field_nodes: &mut VecDeque<Node>) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(
            oos = "IPC: unable to fetch the field for null. The file or stream is corrupted."
        )
    })?;
    Ok(())
}
