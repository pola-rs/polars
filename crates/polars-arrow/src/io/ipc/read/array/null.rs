use std::collections::VecDeque;

use polars_error::{polars_err, PolarsResult};

use super::super::{Node, OutOfSpecKind};
use crate::array::NullArray;
use crate::datatypes::ArrowDataType;

pub fn read_null(
    field_nodes: &mut VecDeque<Node>,
    data_type: ArrowDataType,
) -> PolarsResult<NullArray> {
    let field_node = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(oos =
            "IPC: unable to fetch the field for {data_type:?}. The file or stream is corrupted."
        )
    })?;

    let length: usize = field_node
        .length()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

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
