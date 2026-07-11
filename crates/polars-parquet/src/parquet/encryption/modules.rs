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
    #[cfg(feature = "bloom_filter")]
    BloomFilterHeader = 8,
    #[cfg(feature = "bloom_filter")]
    BloomFilterBitset = 9,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_page_aad_contains_page_ordinal() {
        let aad = create_module_aad(b"file", ModuleType::DataPageHeader, 1, 2, Some(3)).unwrap();
        assert_eq!(aad, b"file\x04\x01\x00\x02\x00\x03\x00");
    }

    #[test]
    fn non_page_aad_does_not_contain_page_ordinal() {
        let dictionary_aad =
            create_module_aad(b"file", ModuleType::DictionaryPageHeader, 1, 2, None).unwrap();
        assert_eq!(dictionary_aad, b"file\x05\x01\x00\x02\x00");
        #[cfg(feature = "bloom_filter")]
        {
            let bloom_aad =
                create_module_aad(b"file", ModuleType::BloomFilterBitset, 1, 2, None).unwrap();
            assert_eq!(bloom_aad, b"file\x09\x01\x00\x02\x00");
        }
    }
}
