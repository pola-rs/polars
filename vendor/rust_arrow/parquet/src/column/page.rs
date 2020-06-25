// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Contains Parquet Page definitions and page reader interface.

use crate::basic::{Encoding, PageType};
use crate::errors::Result;
use crate::file::{metadata::ColumnChunkMetaData, statistics::Statistics};
use crate::schema::types::{ColumnDescPtr, SchemaDescPtr};
use crate::util::memory::ByteBufferPtr;

/// Parquet Page definition.
///
/// List of supported pages.
/// These are 1-to-1 mapped from the equivalent Thrift definitions, except `buf` which
/// used to store uncompressed bytes of the page.
pub enum Page {
    DataPage {
        buf: ByteBufferPtr,
        num_values: u32,
        encoding: Encoding,
        def_level_encoding: Encoding,
        rep_level_encoding: Encoding,
        statistics: Option<Statistics>,
    },
    DataPageV2 {
        buf: ByteBufferPtr,
        num_values: u32,
        encoding: Encoding,
        num_nulls: u32,
        num_rows: u32,
        def_levels_byte_len: u32,
        rep_levels_byte_len: u32,
        is_compressed: bool,
        statistics: Option<Statistics>,
    },
    DictionaryPage {
        buf: ByteBufferPtr,
        num_values: u32,
        encoding: Encoding,
        is_sorted: bool,
    },
}

impl Page {
    /// Returns [`PageType`](crate::basic::PageType) for this page.
    pub fn page_type(&self) -> PageType {
        match self {
            &Page::DataPage { .. } => PageType::DATA_PAGE,
            &Page::DataPageV2 { .. } => PageType::DATA_PAGE_V2,
            &Page::DictionaryPage { .. } => PageType::DICTIONARY_PAGE,
        }
    }

    /// Returns internal byte buffer reference for this page.
    pub fn buffer(&self) -> &ByteBufferPtr {
        match self {
            &Page::DataPage { ref buf, .. } => &buf,
            &Page::DataPageV2 { ref buf, .. } => &buf,
            &Page::DictionaryPage { ref buf, .. } => &buf,
        }
    }

    /// Returns number of values in this page.
    pub fn num_values(&self) -> u32 {
        match self {
            &Page::DataPage { num_values, .. } => num_values,
            &Page::DataPageV2 { num_values, .. } => num_values,
            &Page::DictionaryPage { num_values, .. } => num_values,
        }
    }

    /// Returns this page [`Encoding`](crate::basic::Encoding).
    pub fn encoding(&self) -> Encoding {
        match self {
            &Page::DataPage { encoding, .. } => encoding,
            &Page::DataPageV2 { encoding, .. } => encoding,
            &Page::DictionaryPage { encoding, .. } => encoding,
        }
    }

    /// Returns optional [`Statistics`](crate::file::metadata::Statistics).
    pub fn statistics(&self) -> Option<&Statistics> {
        match self {
            &Page::DataPage { ref statistics, .. } => statistics.as_ref(),
            &Page::DataPageV2 { ref statistics, .. } => statistics.as_ref(),
            &Page::DictionaryPage { .. } => None,
        }
    }
}

/// Helper struct to represent pages with potentially compressed buffer (data page v1) or
/// compressed and concatenated buffer (def levels + rep levels + compressed values for
/// data page v2).
///
/// The difference with `Page` is that `Page` buffer is always uncompressed.
pub struct CompressedPage {
    compressed_page: Page,
    uncompressed_size: usize,
}

impl CompressedPage {
    /// Creates `CompressedPage` from a page with potentially compressed buffer and
    /// uncompressed size.
    pub fn new(compressed_page: Page, uncompressed_size: usize) -> Self {
        Self {
            compressed_page,
            uncompressed_size,
        }
    }

    /// Returns page type.
    pub fn page_type(&self) -> PageType {
        self.compressed_page.page_type()
    }

    /// Returns underlying page with potentially compressed buffer.
    pub fn compressed_page(&self) -> &Page {
        &self.compressed_page
    }

    /// Returns uncompressed size in bytes.
    pub fn uncompressed_size(&self) -> usize {
        self.uncompressed_size
    }

    /// Returns compressed size in bytes.
    ///
    /// Note that it is assumed that buffer is compressed, but it may not be. In this
    /// case compressed size will be equal to uncompressed size.
    pub fn compressed_size(&self) -> usize {
        self.compressed_page.buffer().len()
    }

    /// Number of values in page.
    pub fn num_values(&self) -> u32 {
        self.compressed_page.num_values()
    }

    /// Returns encoding for values in page.
    pub fn encoding(&self) -> Encoding {
        self.compressed_page.encoding()
    }

    /// Returns slice of compressed buffer in the page.
    pub fn data(&self) -> &[u8] {
        self.compressed_page.buffer().data()
    }
}

/// Contains page write metrics.
pub struct PageWriteSpec {
    pub page_type: PageType,
    pub uncompressed_size: usize,
    pub compressed_size: usize,
    pub num_values: u32,
    pub offset: u64,
    pub bytes_written: u64,
}

impl PageWriteSpec {
    /// Creates new spec with default page write metrics.
    pub fn new() -> Self {
        Self {
            page_type: PageType::DATA_PAGE,
            uncompressed_size: 0,
            compressed_size: 0,
            num_values: 0,
            offset: 0,
            bytes_written: 0,
        }
    }
}

/// API for reading pages from a column chunk.
/// This offers a iterator like API to get the next page.
pub trait PageReader {
    /// Gets the next page in the column chunk associated with this reader.
    /// Returns `None` if there are no pages left.
    fn get_next_page(&mut self) -> Result<Option<Page>>;
}

/// API for writing pages in a column chunk.
///
/// It is reasonable to assume that all pages will be written in the correct order, e.g.
/// dictionary page followed by data pages, or a set of data pages, etc.
pub trait PageWriter {
    /// Writes a page into the output stream/sink.
    /// Returns `PageWriteSpec` that contains information about written page metrics,
    /// including number of bytes, size, number of values, offset, etc.
    ///
    /// This method is called for every compressed page we write into underlying buffer,
    /// either data page or dictionary page.
    fn write_page(&mut self, page: CompressedPage) -> Result<PageWriteSpec>;

    /// Writes column chunk metadata into the output stream/sink.
    ///
    /// This method is called once before page writer is closed, normally when writes are
    /// finalised in column writer.
    fn write_metadata(&mut self, metadata: &ColumnChunkMetaData) -> Result<()>;

    /// Closes resources and flushes underlying sink.
    /// Page writer should not be used after this method is called.
    fn close(&mut self) -> Result<()>;
}

/// An iterator over pages of some specific column in a parquet file.
pub trait PageIterator: Iterator<Item = Result<Box<PageReader>>> {
    /// Get schema of parquet file.
    fn schema(&mut self) -> Result<SchemaDescPtr>;

    /// Get column schema of this page iterator.
    fn column_schema(&mut self) -> Result<ColumnDescPtr>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page() {
        let data_page = Page::DataPage {
            buf: ByteBufferPtr::new(vec![0, 1, 2]),
            num_values: 10,
            encoding: Encoding::PLAIN,
            def_level_encoding: Encoding::RLE,
            rep_level_encoding: Encoding::RLE,
            statistics: Some(Statistics::int32(Some(1), Some(2), None, 1, true)),
        };
        assert_eq!(data_page.page_type(), PageType::DATA_PAGE);
        assert_eq!(data_page.buffer().data(), vec![0, 1, 2].as_slice());
        assert_eq!(data_page.num_values(), 10);
        assert_eq!(data_page.encoding(), Encoding::PLAIN);
        assert_eq!(
            data_page.statistics(),
            Some(&Statistics::int32(Some(1), Some(2), None, 1, true))
        );

        let data_page_v2 = Page::DataPageV2 {
            buf: ByteBufferPtr::new(vec![0, 1, 2]),
            num_values: 10,
            encoding: Encoding::PLAIN,
            num_nulls: 5,
            num_rows: 20,
            def_levels_byte_len: 30,
            rep_levels_byte_len: 40,
            is_compressed: false,
            statistics: Some(Statistics::int32(Some(1), Some(2), None, 1, true)),
        };
        assert_eq!(data_page_v2.page_type(), PageType::DATA_PAGE_V2);
        assert_eq!(data_page_v2.buffer().data(), vec![0, 1, 2].as_slice());
        assert_eq!(data_page_v2.num_values(), 10);
        assert_eq!(data_page_v2.encoding(), Encoding::PLAIN);
        assert_eq!(
            data_page_v2.statistics(),
            Some(&Statistics::int32(Some(1), Some(2), None, 1, true))
        );

        let dict_page = Page::DictionaryPage {
            buf: ByteBufferPtr::new(vec![0, 1, 2]),
            num_values: 10,
            encoding: Encoding::PLAIN,
            is_sorted: false,
        };
        assert_eq!(dict_page.page_type(), PageType::DICTIONARY_PAGE);
        assert_eq!(dict_page.buffer().data(), vec![0, 1, 2].as_slice());
        assert_eq!(dict_page.num_values(), 10);
        assert_eq!(dict_page.encoding(), Encoding::PLAIN);
        assert_eq!(dict_page.statistics(), None);
    }

    #[test]
    fn test_compressed_page() {
        let data_page = Page::DataPage {
            buf: ByteBufferPtr::new(vec![0, 1, 2]),
            num_values: 10,
            encoding: Encoding::PLAIN,
            def_level_encoding: Encoding::RLE,
            rep_level_encoding: Encoding::RLE,
            statistics: Some(Statistics::int32(Some(1), Some(2), None, 1, true)),
        };

        let cpage = CompressedPage::new(data_page, 5);

        assert_eq!(cpage.page_type(), PageType::DATA_PAGE);
        assert_eq!(cpage.uncompressed_size(), 5);
        assert_eq!(cpage.compressed_size(), 3);
        assert_eq!(cpage.num_values(), 10);
        assert_eq!(cpage.encoding(), Encoding::PLAIN);
        assert_eq!(cpage.data(), &[0, 1, 2]);
    }
}
