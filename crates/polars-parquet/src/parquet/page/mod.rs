use std::sync::Arc;

pub use crate::thrift_format::{
    DataPageHeader as DataPageHeaderV1, DataPageHeaderV2, PageHeader as ParquetPageHeader,
};

use crate::indexes::Interval;
pub use crate::parquet_bridge::{DataPageHeaderExt, PageType};

use crate::compression::Compression;
use crate::encoding::{get_length, Encoding};
use crate::error::{Error, Result};
use crate::metadata::Descriptor;

use crate::statistics::{deserialize_statistics, Statistics};

/// A [`CompressedDataPage`] is compressed, encoded representation of a Parquet data page.
/// It holds actual data and thus cloning it is expensive.
#[derive(Debug)]
pub struct CompressedDataPage {
    pub(crate) header: DataPageHeader,
    pub(crate) buffer: Vec<u8>,
    pub(crate) compression: Compression,
    uncompressed_page_size: usize,
    pub(crate) descriptor: Descriptor,

    // The offset and length in rows
    pub(crate) selected_rows: Option<Vec<Interval>>,
}

impl CompressedDataPage {
    /// Returns a new [`CompressedDataPage`].
    pub fn new(
        header: DataPageHeader,
        buffer: Vec<u8>,
        compression: Compression,
        uncompressed_page_size: usize,
        descriptor: Descriptor,
        rows: Option<usize>,
    ) -> Self {
        Self::new_read(
            header,
            buffer,
            compression,
            uncompressed_page_size,
            descriptor,
            rows.map(|x| vec![Interval::new(0, x)]),
        )
    }

    /// Returns a new [`CompressedDataPage`].
    pub(crate) fn new_read(
        header: DataPageHeader,
        buffer: Vec<u8>,
        compression: Compression,
        uncompressed_page_size: usize,
        descriptor: Descriptor,
        selected_rows: Option<Vec<Interval>>,
    ) -> Self {
        Self {
            header,
            buffer,
            compression,
            uncompressed_page_size,
            descriptor,
            selected_rows,
        }
    }

    pub fn header(&self) -> &DataPageHeader {
        &self.header
    }

    pub fn uncompressed_size(&self) -> usize {
        self.uncompressed_page_size
    }

    pub fn compressed_size(&self) -> usize {
        self.buffer.len()
    }

    /// The compression of the data in this page.
    /// Note that what is compressed in a page depends on its version:
    /// in V1, the whole data (`[repetition levels][definition levels][values]`) is compressed; in V2 only the values are compressed.
    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// the rows to be selected by this page.
    /// When `None`, all rows are to be considered.
    pub fn selected_rows(&self) -> Option<&[Interval]> {
        self.selected_rows.as_deref()
    }

    pub fn num_values(&self) -> usize {
        self.header.num_values()
    }

    /// Decodes the raw statistics into a statistics
    pub fn statistics(&self) -> Option<Result<Arc<dyn Statistics>>> {
        match &self.header {
            DataPageHeader::V1(d) => d
                .statistics
                .as_ref()
                .map(|x| deserialize_statistics(x, self.descriptor.primitive_type.clone())),
            DataPageHeader::V2(d) => d
                .statistics
                .as_ref()
                .map(|x| deserialize_statistics(x, self.descriptor.primitive_type.clone())),
        }
    }

    #[inline]
    pub fn select_rows(&mut self, selected_rows: Vec<Interval>) {
        self.selected_rows = Some(selected_rows);
    }
}

#[derive(Debug, Clone)]
pub enum DataPageHeader {
    V1(DataPageHeaderV1),
    V2(DataPageHeaderV2),
}

impl DataPageHeader {
    pub fn num_values(&self) -> usize {
        match &self {
            DataPageHeader::V1(d) => d.num_values as usize,
            DataPageHeader::V2(d) => d.num_values as usize,
        }
    }
}

/// A [`DataPage`] is an uncompressed, encoded representation of a Parquet data page. It holds actual data
/// and thus cloning it is expensive.
#[derive(Debug, Clone)]
pub struct DataPage {
    pub(super) header: DataPageHeader,
    pub(super) buffer: Vec<u8>,
    pub descriptor: Descriptor,
    pub selected_rows: Option<Vec<Interval>>,
}

impl DataPage {
    pub fn new(
        header: DataPageHeader,
        buffer: Vec<u8>,
        descriptor: Descriptor,
        rows: Option<usize>,
    ) -> Self {
        Self::new_read(
            header,
            buffer,
            descriptor,
            rows.map(|x| vec![Interval::new(0, x)]),
        )
    }

    pub(crate) fn new_read(
        header: DataPageHeader,
        buffer: Vec<u8>,
        descriptor: Descriptor,
        selected_rows: Option<Vec<Interval>>,
    ) -> Self {
        Self {
            header,
            buffer,
            descriptor,
            selected_rows,
        }
    }

    pub fn header(&self) -> &DataPageHeader {
        &self.header
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// the rows to be selected by this page.
    /// When `None`, all rows are to be considered.
    pub fn selected_rows(&self) -> Option<&[Interval]> {
        self.selected_rows.as_deref()
    }

    /// Returns a mutable reference to the internal buffer.
    /// Useful to recover the buffer after the page has been decoded.
    pub fn buffer_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }

    pub fn num_values(&self) -> usize {
        self.header.num_values()
    }

    pub fn encoding(&self) -> Encoding {
        match &self.header {
            DataPageHeader::V1(d) => d.encoding(),
            DataPageHeader::V2(d) => d.encoding(),
        }
    }

    pub fn definition_level_encoding(&self) -> Encoding {
        match &self.header {
            DataPageHeader::V1(d) => d.definition_level_encoding(),
            DataPageHeader::V2(_) => Encoding::Rle,
        }
    }

    pub fn repetition_level_encoding(&self) -> Encoding {
        match &self.header {
            DataPageHeader::V1(d) => d.repetition_level_encoding(),
            DataPageHeader::V2(_) => Encoding::Rle,
        }
    }

    /// Decodes the raw statistics into a statistics
    pub fn statistics(&self) -> Option<Result<Arc<dyn Statistics>>> {
        match &self.header {
            DataPageHeader::V1(d) => d
                .statistics
                .as_ref()
                .map(|x| deserialize_statistics(x, self.descriptor.primitive_type.clone())),
            DataPageHeader::V2(d) => d
                .statistics
                .as_ref()
                .map(|x| deserialize_statistics(x, self.descriptor.primitive_type.clone())),
        }
    }
}

/// A [`Page`] is an uncompressed, encoded representation of a Parquet page. It may hold actual data
/// and thus cloning it may be expensive.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Page {
    /// A [`DataPage`]
    Data(DataPage),
    /// A [`DictPage`]
    Dict(DictPage),
}

impl Page {
    pub(crate) fn buffer(&mut self) -> &mut Vec<u8> {
        match self {
            Self::Data(page) => &mut page.buffer,
            Self::Dict(page) => &mut page.buffer,
        }
    }
}

/// A [`CompressedPage`] is a compressed, encoded representation of a Parquet page. It holds actual data
/// and thus cloning it is expensive.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum CompressedPage {
    Data(CompressedDataPage),
    Dict(CompressedDictPage),
}

impl CompressedPage {
    pub(crate) fn buffer(&mut self) -> &mut Vec<u8> {
        match self {
            CompressedPage::Data(page) => &mut page.buffer,
            CompressedPage::Dict(page) => &mut page.buffer,
        }
    }

    pub(crate) fn compression(&self) -> Compression {
        match self {
            CompressedPage::Data(page) => page.compression(),
            CompressedPage::Dict(page) => page.compression(),
        }
    }

    pub(crate) fn num_values(&self) -> usize {
        match self {
            CompressedPage::Data(page) => page.num_values(),
            CompressedPage::Dict(_) => 0,
        }
    }

    pub(crate) fn selected_rows(&self) -> Option<&[Interval]> {
        match self {
            CompressedPage::Data(page) => page.selected_rows(),
            CompressedPage::Dict(_) => None,
        }
    }

    pub(crate) fn uncompressed_size(&self) -> usize {
        match self {
            CompressedPage::Data(page) => page.uncompressed_page_size,
            CompressedPage::Dict(page) => page.uncompressed_page_size,
        }
    }
}

/// An uncompressed, encoded dictionary page.
#[derive(Debug)]
pub struct DictPage {
    pub buffer: Vec<u8>,
    pub num_values: usize,
    pub is_sorted: bool,
}

impl DictPage {
    pub fn new(buffer: Vec<u8>, num_values: usize, is_sorted: bool) -> Self {
        Self {
            buffer,
            num_values,
            is_sorted,
        }
    }
}

/// A compressed, encoded dictionary page.
#[derive(Debug)]
pub struct CompressedDictPage {
    pub(crate) buffer: Vec<u8>,
    compression: Compression,
    pub(crate) num_values: usize,
    pub(crate) uncompressed_page_size: usize,
    pub is_sorted: bool,
}

impl CompressedDictPage {
    pub fn new(
        buffer: Vec<u8>,
        compression: Compression,
        uncompressed_page_size: usize,
        num_values: usize,
        is_sorted: bool,
    ) -> Self {
        Self {
            buffer,
            compression,
            uncompressed_page_size,
            num_values,
            is_sorted,
        }
    }

    /// The compression of the data in this page.
    pub fn compression(&self) -> Compression {
        self.compression
    }
}

/// Splits the page buffer into 3 slices corresponding to (encoded rep levels, encoded def levels, encoded values) for v1 pages.
#[inline]
pub fn split_buffer_v1(
    buffer: &[u8],
    has_rep: bool,
    has_def: bool,
) -> Result<(&[u8], &[u8], &[u8])> {
    let (rep, buffer) = if has_rep {
        let level_buffer_length = get_length(buffer).ok_or_else(|| {
            Error::oos("The number of bytes declared in v1 rep levels is higher than the page size")
        })?;
        (
            buffer.get(4..4 + level_buffer_length).ok_or_else(|| {
                Error::oos(
                    "The number of bytes declared in v1 rep levels is higher than the page size",
                )
            })?,
            buffer.get(4 + level_buffer_length..).ok_or_else(|| {
                Error::oos(
                    "The number of bytes declared in v1 rep levels is higher than the page size",
                )
            })?,
        )
    } else {
        (&[] as &[u8], buffer)
    };

    let (def, buffer) = if has_def {
        let level_buffer_length = get_length(buffer).ok_or_else(|| {
            Error::oos("The number of bytes declared in v1 rep levels is higher than the page size")
        })?;
        (
            buffer.get(4..4 + level_buffer_length).ok_or_else(|| {
                Error::oos(
                    "The number of bytes declared in v1 def levels is higher than the page size",
                )
            })?,
            buffer.get(4 + level_buffer_length..).ok_or_else(|| {
                Error::oos(
                    "The number of bytes declared in v1 def levels is higher than the page size",
                )
            })?,
        )
    } else {
        (&[] as &[u8], buffer)
    };

    Ok((rep, def, buffer))
}

/// Splits the page buffer into 3 slices corresponding to (encoded rep levels, encoded def levels, encoded values) for v2 pages.
pub fn split_buffer_v2(
    buffer: &[u8],
    rep_level_buffer_length: usize,
    def_level_buffer_length: usize,
) -> Result<(&[u8], &[u8], &[u8])> {
    Ok((
        &buffer[..rep_level_buffer_length],
        &buffer[rep_level_buffer_length..rep_level_buffer_length + def_level_buffer_length],
        &buffer[rep_level_buffer_length + def_level_buffer_length..],
    ))
}

/// Splits the page buffer into 3 slices corresponding to (encoded rep levels, encoded def levels, encoded values).
pub fn split_buffer(page: &DataPage) -> Result<(&[u8], &[u8], &[u8])> {
    match page.header() {
        DataPageHeader::V1(_) => split_buffer_v1(
            page.buffer(),
            page.descriptor.max_rep_level > 0,
            page.descriptor.max_def_level > 0,
        ),
        DataPageHeader::V2(header) => {
            let def_level_buffer_length: usize = header.definition_levels_byte_length.try_into()?;
            let rep_level_buffer_length: usize = header.repetition_levels_byte_length.try_into()?;
            split_buffer_v2(
                page.buffer(),
                rep_level_buffer_length,
                def_level_buffer_length,
            )
        }
    }
}
