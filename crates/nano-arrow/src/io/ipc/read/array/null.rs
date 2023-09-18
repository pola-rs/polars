use std::collections::VecDeque;

use super::super::{Node, OutOfSpecKind};
use crate::array::NullArray;
use crate::datatypes::DataType;
use crate::error::{Error, Result};

pub fn read_null(field_nodes: &mut VecDeque<Node>, data_type: DataType) -> Result<NullArray> {
    let field_node = field_nodes.pop_front().ok_or_else(|| {
        Error::oos(format!(
            "IPC: unable to fetch the field for {data_type:?}. The file or stream is corrupted."
        ))
    })?;

    let length: usize = field_node
        .length()
        .try_into()
        .map_err(|_| Error::from(OutOfSpecKind::NegativeFooterLength))?;

    NullArray::try_new(data_type, length)
}

pub fn skip_null(field_nodes: &mut VecDeque<Node>) -> Result<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        Error::oos("IPC: unable to fetch the field for null. The file or stream is corrupted.")
    })?;
    Ok(())
}
