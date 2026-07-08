// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0.

use crate::parquet::error::{ParquetError, ParquetResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModuleType {
    Footer = 0,
    ColumnMetaData = 1,
    DataPage = 2,
    DictionaryPage = 3,
    DataPageHeader = 4,
    DictionaryPageHeader = 5,
    #[allow(dead_code)]
    ColumnIndex = 6,
    #[allow(dead_code)]
    OffsetIndex = 7,
}

pub(crate) fn create_footer_aad(file_aad: &[u8]) -> ParquetResult<Vec<u8>> {
    create_module_aad(file_aad, ModuleType::Footer, 0, 0, None)
}

pub(crate) fn create_module_aad(
    file_aad: &[u8],
    module_type: ModuleType,
    row_group_index: usize,
    column_ordinal: usize,
    page_ordinal: Option<usize>,
) -> ParquetResult<Vec<u8>> {
    let module_byte = module_type as u8;

    if module_type == ModuleType::Footer {
        let mut aad = Vec::with_capacity(file_aad.len() + 1);
        aad.extend_from_slice(file_aad);
        aad.push(module_byte);
        return Ok(aad);
    }

    if row_group_index > i16::MAX as usize {
        return Err(ParquetError::InvalidParameter(format!(
            "encrypted parquet files cannot have more than {} row groups",
            i16::MAX
        )));
    }
    if column_ordinal > i16::MAX as usize {
        return Err(ParquetError::InvalidParameter(format!(
            "encrypted parquet files cannot have more than {} columns",
            i16::MAX
        )));
    }

    if !matches!(
        module_type,
        ModuleType::DataPageHeader | ModuleType::DataPage
    ) {
        let mut aad = Vec::with_capacity(file_aad.len() + 5);
        aad.extend_from_slice(file_aad);
        aad.push(module_byte);
        aad.extend_from_slice(&(row_group_index as i16).to_le_bytes());
        aad.extend_from_slice(&(column_ordinal as i16).to_le_bytes());
        return Ok(aad);
    }

    let page_ordinal = page_ordinal
        .ok_or_else(|| ParquetError::InvalidParameter("page ordinal must be set".to_string()))?;
    if page_ordinal > i16::MAX as usize {
        return Err(ParquetError::InvalidParameter(format!(
            "encrypted parquet files cannot have more than {} pages per column chunk",
            i16::MAX
        )));
    }

    let mut aad = Vec::with_capacity(file_aad.len() + 7);
    aad.extend_from_slice(file_aad);
    aad.push(module_byte);
    aad.extend_from_slice(&(row_group_index as i16).to_le_bytes());
    aad.extend_from_slice(&(column_ordinal as i16).to_le_bytes());
    aad.extend_from_slice(&(page_ordinal as i16).to_le_bytes());
    Ok(aad)
}
