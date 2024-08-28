use super::CowBuffer;
use crate::parquet::compression::Compression;
use crate::parquet::encoding::{get_length, Encoding};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::Descriptor;
pub use crate::parquet::parquet_bridge::{DataPageHeaderExt, PageType};
use crate::parquet::statistics::Statistics;
pub use crate::parquet::thrift_format::{
    DataPageHeader as DataPageHeaderV1, DataPageHeaderV2, PageHeader as ParquetPageHeader,
};

pub enum PageResult {
    Single(Page),
    Two { dict: DictPage, data: DataPage },
}

/// A [`CompressedDataPage`] is compressed, encoded representation of a Parquet data page.
/// It holds actual data and thus cloning it is expensive.
#[derive(Debug)]
pub struct CompressedDataPage {
    pub(crate) header: DataPageHeader,
    pub(crate) buffer: CowBuffer,
    pub(crate) compression: Compression,
    uncompressed_page_size: usize,
    pub(crate) descriptor: Descriptor,
    pub num_rows: Option<usize>,
}

impl CompressedDataPage {
    /// Returns a new [`CompressedDataPage`].
    pub fn new(
        header: DataPageHeader,
        buffer: CowBuffer,
        compression: Compression,
        uncompressed_page_size: usize,
        descriptor: Descriptor,
        num_rows: usize,
    ) -> Self {
        Self {
            header,
            buffer,
            compression,
            uncompressed_page_size,
            descriptor,
            num_rows: Some(num_rows),
        }
    }

    /// Returns a new [`CompressedDataPage`].
    pub(crate) fn new_read(
        header: DataPageHeader,
        buffer: CowBuffer,
        compression: Compression,
        uncompressed_page_size: usize,
        descriptor: Descriptor,
    ) -> Self {
        Self {
            header,
            buffer,
            compression,
            uncompressed_page_size,
            descriptor,
            num_rows: None,
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

    pub fn num_values(&self) -> usize {
        self.header.num_values()
    }

    pub fn num_rows(&self) -> Option<usize> {
        self.num_rows
    }

    /// Decodes the raw statistics into a statistics
    pub fn statistics(&self) -> Option<ParquetResult<Statistics>> {
        match &self.header {
            DataPageHeader::V1(d) => d
                .statistics
                .as_ref()
                .map(|x| Statistics::deserialize(x, self.descriptor.primitive_type.clone())),
            DataPageHeader::V2(d) => d
                .statistics
                .as_ref()
                .map(|x| Statistics::deserialize(x, self.descriptor.primitive_type.clone())),
        }
    }

    pub fn slice_mut(&mut self) -> &mut CowBuffer {
        &mut self.buffer
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

    pub fn null_count(&self) -> Option<usize> {
        match &self {
            DataPageHeader::V1(_) => None,
            DataPageHeader::V2(d) => Some(d.num_nulls as usize),
        }
    }
}

/// A [`DataPage`] is an uncompressed, encoded representation of a Parquet data page. It holds actual data
/// and thus cloning it is expensive.
#[derive(Debug, Clone)]
pub struct DataPage {
    pub(super) header: DataPageHeader,
    pub(super) buffer: CowBuffer,
    pub descriptor: Descriptor,
    pub num_rows: Option<usize>,
}

impl DataPage {
    pub fn new(
        header: DataPageHeader,
        buffer: CowBuffer,
        descriptor: Descriptor,
        num_rows: usize,
    ) -> Self {
        Self {
            header,
            buffer,
            descriptor,
            num_rows: Some(num_rows),
        }
    }

    pub(crate) fn new_read(
        header: DataPageHeader,
        buffer: CowBuffer,
        descriptor: Descriptor,
    ) -> Self {
        Self {
            header,
            buffer,
            descriptor,
            num_rows: None,
        }
    }

    pub fn header(&self) -> &DataPageHeader {
        &self.header
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Returns a mutable reference to the internal buffer.
    /// Useful to recover the buffer after the page has been decoded.
    pub fn buffer_mut(&mut self) -> &mut Vec<u8> {
        self.buffer.to_mut()
    }

    pub fn num_values(&self) -> usize {
        self.header.num_values()
    }

    pub fn null_count(&self) -> Option<usize> {
        self.header.null_count()
    }

    pub fn num_rows(&self) -> Option<usize> {
        self.num_rows
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
    pub fn statistics(&self) -> Option<ParquetResult<Statistics>> {
        match &self.header {
            DataPageHeader::V1(d) => d
                .statistics
                .as_ref()
                .map(|x| Statistics::deserialize(x, self.descriptor.primitive_type.clone())),
            DataPageHeader::V2(d) => d
                .statistics
                .as_ref()
                .map(|x| Statistics::deserialize(x, self.descriptor.primitive_type.clone())),
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
    pub(crate) fn buffer_mut(&mut self) -> &mut Vec<u8> {
        match self {
            Self::Data(page) => page.buffer.to_mut(),
            Self::Dict(page) => page.buffer.to_mut(),
        }
    }

    pub(crate) fn unwrap_data(self) -> DataPage {
        match self {
            Self::Data(page) => page,
            _ => panic!(),
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
    pub(crate) fn buffer_mut(&mut self) -> &mut Vec<u8> {
        match self {
            CompressedPage::Data(page) => page.buffer.to_mut(),
            CompressedPage::Dict(page) => page.buffer.to_mut(),
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

    pub(crate) fn num_rows(&self) -> Option<usize> {
        match self {
            CompressedPage::Data(page) => page.num_rows(),
            CompressedPage::Dict(_) => Some(0),
        }
    }
}

/// An uncompressed, encoded dictionary page.
#[derive(Debug, Clone)]
pub struct DictPage {
    pub buffer: CowBuffer,
    pub num_values: usize,
    pub is_sorted: bool,
}

impl DictPage {
    pub fn new(buffer: CowBuffer, num_values: usize, is_sorted: bool) -> Self {
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
    pub(crate) buffer: CowBuffer,
    compression: Compression,
    pub(crate) num_values: usize,
    pub(crate) uncompressed_page_size: usize,
    pub is_sorted: bool,
}

impl CompressedDictPage {
    pub fn new(
        buffer: CowBuffer,
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

pub struct EncodedSplitBuffer<'a> {
    /// Encoded Repetition Levels
    pub rep: &'a [u8],
    /// Encoded Definition Levels
    pub def: &'a [u8],
    /// Encoded Values
    pub values: &'a [u8],
}

/// Splits the page buffer into 3 slices corresponding to (encoded rep levels, encoded def levels, encoded values) for v1 pages.
#[inline]
pub fn split_buffer_v1(
    buffer: &[u8],
    has_rep: bool,
    has_def: bool,
) -> ParquetResult<EncodedSplitBuffer> {
    let (rep, buffer) = if has_rep {
        let level_buffer_length = get_length(buffer).ok_or_else(|| {
            ParquetError::oos(
                "The number of bytes declared in v1 rep levels is higher than the page size",
            )
        })?;

        if buffer.len() < level_buffer_length + 4 {
            return Err(ParquetError::oos(
                "The number of bytes declared in v1 rep levels is higher than the page size",
            ));
        }

        buffer[4..].split_at(level_buffer_length)
    } else {
        (&[] as &[u8], buffer)
    };

    let (def, buffer) = if has_def {
        let level_buffer_length = get_length(buffer).ok_or_else(|| {
            ParquetError::oos(
                "The number of bytes declared in v1 def levels is higher than the page size",
            )
        })?;

        if buffer.len() < level_buffer_length + 4 {
            return Err(ParquetError::oos(
                "The number of bytes declared in v1 def levels is higher than the page size",
            ));
        }

        buffer[4..].split_at(level_buffer_length)
    } else {
        (&[] as &[u8], buffer)
    };

    Ok(EncodedSplitBuffer {
        rep,
        def,
        values: buffer,
    })
}

/// Splits the page buffer into 3 slices corresponding to (encoded rep levels, encoded def levels, encoded values) for v2 pages.
pub fn split_buffer_v2(
    buffer: &[u8],
    rep_level_buffer_length: usize,
    def_level_buffer_length: usize,
) -> ParquetResult<EncodedSplitBuffer> {
    let (rep, buffer) = buffer.split_at(rep_level_buffer_length);
    let (def, values) = buffer.split_at(def_level_buffer_length);

    Ok(EncodedSplitBuffer { rep, def, values })
}

/// Splits the page buffer into 3 slices corresponding to (encoded rep levels, encoded def levels, encoded values).
pub fn split_buffer(page: &DataPage) -> ParquetResult<EncodedSplitBuffer> {
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
        },
    }
}
