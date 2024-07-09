// Bridges structs from thrift-generated code to rust enums.

#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use super::thrift_format::{
    BoundaryOrder as ParquetBoundaryOrder, CompressionCodec, DataPageHeader, DataPageHeaderV2,
    DecimalType, Encoding as ParquetEncoding, FieldRepetitionType, IntType,
    LogicalType as ParquetLogicalType, PageType as ParquetPageType, TimeType,
    TimeUnit as ParquetTimeUnit, TimestampType,
};
use crate::parquet::error::ParquetError;

/// The repetition of a parquet field
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum Repetition {
    /// When the field has no null values
    Required,
    /// When the field may have null values
    Optional,
    /// When the field may be repeated (list field)
    Repeated,
}

impl TryFrom<FieldRepetitionType> for Repetition {
    type Error = ParquetError;

    fn try_from(repetition: FieldRepetitionType) -> Result<Self, Self::Error> {
        Ok(match repetition {
            FieldRepetitionType::REQUIRED => Repetition::Required,
            FieldRepetitionType::OPTIONAL => Repetition::Optional,
            FieldRepetitionType::REPEATED => Repetition::Repeated,
            _ => return Err(ParquetError::oos("Thrift out of range")),
        })
    }
}

impl From<Repetition> for FieldRepetitionType {
    fn from(repetition: Repetition) -> Self {
        match repetition {
            Repetition::Required => FieldRepetitionType::REQUIRED,
            Repetition::Optional => FieldRepetitionType::OPTIONAL,
            Repetition::Repeated => FieldRepetitionType::REPEATED,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum Compression {
    Uncompressed,
    Snappy,
    Gzip,
    Lzo,
    Brotli,
    Lz4,
    Zstd,
    Lz4Raw,
}

impl TryFrom<CompressionCodec> for Compression {
    type Error = ParquetError;

    fn try_from(codec: CompressionCodec) -> Result<Self, Self::Error> {
        Ok(match codec {
            CompressionCodec::UNCOMPRESSED => Compression::Uncompressed,
            CompressionCodec::SNAPPY => Compression::Snappy,
            CompressionCodec::GZIP => Compression::Gzip,
            CompressionCodec::LZO => Compression::Lzo,
            CompressionCodec::BROTLI => Compression::Brotli,
            CompressionCodec::LZ4 => Compression::Lz4,
            CompressionCodec::ZSTD => Compression::Zstd,
            CompressionCodec::LZ4_RAW => Compression::Lz4Raw,
            _ => return Err(ParquetError::oos("Thrift out of range")),
        })
    }
}

impl From<Compression> for CompressionCodec {
    fn from(codec: Compression) -> Self {
        match codec {
            Compression::Uncompressed => CompressionCodec::UNCOMPRESSED,
            Compression::Snappy => CompressionCodec::SNAPPY,
            Compression::Gzip => CompressionCodec::GZIP,
            Compression::Lzo => CompressionCodec::LZO,
            Compression::Brotli => CompressionCodec::BROTLI,
            Compression::Lz4 => CompressionCodec::LZ4,
            Compression::Zstd => CompressionCodec::ZSTD,
            Compression::Lz4Raw => CompressionCodec::LZ4_RAW,
        }
    }
}

/// Defines the compression settings for writing a parquet file.
///
/// If None is provided as a compression setting, then the default compression level is used.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub enum CompressionOptions {
    Uncompressed,
    Snappy,
    Gzip(Option<GzipLevel>),
    Lzo,
    Brotli(Option<BrotliLevel>),
    Lz4,
    Zstd(Option<ZstdLevel>),
    Lz4Raw,
}

impl From<CompressionOptions> for Compression {
    fn from(value: CompressionOptions) -> Self {
        match value {
            CompressionOptions::Uncompressed => Compression::Uncompressed,
            CompressionOptions::Snappy => Compression::Snappy,
            CompressionOptions::Gzip(_) => Compression::Gzip,
            CompressionOptions::Lzo => Compression::Lzo,
            CompressionOptions::Brotli(_) => Compression::Brotli,
            CompressionOptions::Lz4 => Compression::Lz4,
            CompressionOptions::Zstd(_) => Compression::Zstd,
            CompressionOptions::Lz4Raw => Compression::Lz4Raw,
        }
    }
}

impl From<CompressionOptions> for CompressionCodec {
    fn from(codec: CompressionOptions) -> Self {
        match codec {
            CompressionOptions::Uncompressed => CompressionCodec::UNCOMPRESSED,
            CompressionOptions::Snappy => CompressionCodec::SNAPPY,
            CompressionOptions::Gzip(_) => CompressionCodec::GZIP,
            CompressionOptions::Lzo => CompressionCodec::LZO,
            CompressionOptions::Brotli(_) => CompressionCodec::BROTLI,
            CompressionOptions::Lz4 => CompressionCodec::LZ4,
            CompressionOptions::Zstd(_) => CompressionCodec::ZSTD,
            CompressionOptions::Lz4Raw => CompressionCodec::LZ4_RAW,
        }
    }
}

/// Defines valid compression levels.
pub(crate) trait CompressionLevel<T: std::fmt::Display + std::cmp::PartialOrd> {
    const MINIMUM_LEVEL: T;
    const MAXIMUM_LEVEL: T;

    /// Tests if the provided compression level is valid.
    fn is_valid_level(level: T) -> Result<(), ParquetError> {
        let compression_range = Self::MINIMUM_LEVEL..=Self::MAXIMUM_LEVEL;
        if compression_range.contains(&level) {
            Ok(())
        } else {
            Err(ParquetError::InvalidParameter(format!(
                "valid compression range {}..={} exceeded.",
                compression_range.start(),
                compression_range.end()
            )))
        }
    }
}

/// Represents a valid brotli compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct BrotliLevel(u32);

impl Default for BrotliLevel {
    fn default() -> Self {
        Self(1)
    }
}

impl CompressionLevel<u32> for BrotliLevel {
    const MINIMUM_LEVEL: u32 = 0;
    const MAXIMUM_LEVEL: u32 = 11;
}

impl BrotliLevel {
    /// Attempts to create a brotli compression level.
    ///
    /// Compression levels must be valid.
    pub fn try_new(level: u32) -> Result<Self, ParquetError> {
        Self::is_valid_level(level).map(|_| Self(level))
    }

    /// Returns the compression level.
    pub fn compression_level(&self) -> u32 {
        self.0
    }
}

/// Represents a valid gzip compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct GzipLevel(u8);

impl Default for GzipLevel {
    fn default() -> Self {
        // The default as of miniz_oxide 0.5.1 is 6 for compression level
        // (miniz_oxide::deflate::CompressionLevel::DefaultLevel)
        Self(6)
    }
}

impl CompressionLevel<u8> for GzipLevel {
    const MINIMUM_LEVEL: u8 = 0;
    const MAXIMUM_LEVEL: u8 = 10;
}

impl GzipLevel {
    /// Attempts to create a gzip compression level.
    ///
    /// Compression levels must be valid (i.e. be acceptable for [`flate2::Compression`]).
    pub fn try_new(level: u8) -> Result<Self, ParquetError> {
        Self::is_valid_level(level).map(|_| Self(level))
    }

    /// Returns the compression level.
    pub fn compression_level(&self) -> u8 {
        self.0
    }
}

#[cfg(feature = "gzip")]
impl From<GzipLevel> for flate2::Compression {
    fn from(level: GzipLevel) -> Self {
        Self::new(level.compression_level() as u32)
    }
}

/// Represents a valid zstd compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct ZstdLevel(i32);

impl CompressionLevel<i32> for ZstdLevel {
    // zstd binds to C, and hence zstd::compression_level_range() is not const as this calls the
    // underlying C library.
    const MINIMUM_LEVEL: i32 = 1;
    const MAXIMUM_LEVEL: i32 = 22;
}

impl ZstdLevel {
    /// Attempts to create a zstd compression level from a given compression level.
    ///
    /// Compression levels must be valid (i.e. be acceptable for [`zstd::compression_level_range`]).
    pub fn try_new(level: i32) -> Result<Self, ParquetError> {
        Self::is_valid_level(level).map(|_| Self(level))
    }

    /// Returns the compression level.
    pub fn compression_level(&self) -> i32 {
        self.0
    }
}

#[cfg(feature = "zstd")]
impl Default for ZstdLevel {
    fn default() -> Self {
        Self(zstd::DEFAULT_COMPRESSION_LEVEL)
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub enum PageType {
    DataPage,
    DataPageV2,
    DictionaryPage,
}

impl TryFrom<ParquetPageType> for PageType {
    type Error = ParquetError;

    fn try_from(type_: ParquetPageType) -> Result<Self, Self::Error> {
        Ok(match type_ {
            ParquetPageType::DATA_PAGE => PageType::DataPage,
            ParquetPageType::DATA_PAGE_V2 => PageType::DataPageV2,
            ParquetPageType::DICTIONARY_PAGE => PageType::DictionaryPage,
            _ => return Err(ParquetError::oos("Thrift out of range")),
        })
    }
}

impl From<PageType> for ParquetPageType {
    fn from(type_: PageType) -> Self {
        match type_ {
            PageType::DataPage => ParquetPageType::DATA_PAGE,
            PageType::DataPageV2 => ParquetPageType::DATA_PAGE_V2,
            PageType::DictionaryPage => ParquetPageType::DICTIONARY_PAGE,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub enum Encoding {
    /// Default encoding.
    /// BOOLEAN - 1 bit per value. 0 is false; 1 is true.
    /// INT32 - 4 bytes per value.  Stored as little-endian.
    /// INT64 - 8 bytes per value.  Stored as little-endian.
    /// FLOAT - 4 bytes per value.  IEEE. Stored as little-endian.
    /// DOUBLE - 8 bytes per value.  IEEE. Stored as little-endian.
    /// BYTE_ARRAY - 4 byte length stored as little endian, followed by bytes.
    /// FIXED_LEN_BYTE_ARRAY - Just the bytes.
    Plain,
    /// Deprecated: Dictionary encoding. The values in the dictionary are encoded in the
    /// plain type.
    /// in a data page use RLE_DICTIONARY instead.
    /// in a Dictionary page use PLAIN instead
    PlainDictionary,
    /// Group packed run length encoding. Usable for definition/repetition levels
    /// encoding and Booleans (on one bit: 0 is false; 1 is true.)
    Rle,
    /// Bit packed encoding.  This can only be used if the data has a known max
    /// width.  Usable for definition/repetition levels encoding.
    BitPacked,
    /// Delta encoding for integers. This can be used for int columns and works best
    /// on sorted data
    DeltaBinaryPacked,
    /// Encoding for byte arrays to separate the length values and the data. The lengths
    /// are encoded using DELTA_BINARY_PACKED
    DeltaLengthByteArray,
    /// Incremental-encoded byte array. Prefix lengths are encoded using DELTA_BINARY_PACKED.
    /// Suffixes are stored as delta length byte arrays.
    DeltaByteArray,
    /// Dictionary encoding: the ids are encoded using the RLE encoding
    RleDictionary,
    /// Encoding for floating-point data.
    /// K byte-streams are created where K is the size in bytes of the data type.
    /// The individual bytes of an FP value are scattered to the corresponding stream and
    /// the streams are concatenated.
    /// This itself does not reduce the size of the data but can lead to better compression
    /// afterwards.
    ByteStreamSplit,
}

impl TryFrom<ParquetEncoding> for Encoding {
    type Error = ParquetError;

    fn try_from(encoding: ParquetEncoding) -> Result<Self, Self::Error> {
        Ok(match encoding {
            ParquetEncoding::PLAIN => Encoding::Plain,
            ParquetEncoding::PLAIN_DICTIONARY => Encoding::PlainDictionary,
            ParquetEncoding::RLE => Encoding::Rle,
            ParquetEncoding::BIT_PACKED => Encoding::BitPacked,
            ParquetEncoding::DELTA_BINARY_PACKED => Encoding::DeltaBinaryPacked,
            ParquetEncoding::DELTA_LENGTH_BYTE_ARRAY => Encoding::DeltaLengthByteArray,
            ParquetEncoding::DELTA_BYTE_ARRAY => Encoding::DeltaByteArray,
            ParquetEncoding::RLE_DICTIONARY => Encoding::RleDictionary,
            ParquetEncoding::BYTE_STREAM_SPLIT => Encoding::ByteStreamSplit,
            _ => return Err(ParquetError::oos("Thrift out of range")),
        })
    }
}

impl From<Encoding> for ParquetEncoding {
    fn from(encoding: Encoding) -> Self {
        match encoding {
            Encoding::Plain => ParquetEncoding::PLAIN,
            Encoding::PlainDictionary => ParquetEncoding::PLAIN_DICTIONARY,
            Encoding::Rle => ParquetEncoding::RLE,
            Encoding::BitPacked => ParquetEncoding::BIT_PACKED,
            Encoding::DeltaBinaryPacked => ParquetEncoding::DELTA_BINARY_PACKED,
            Encoding::DeltaLengthByteArray => ParquetEncoding::DELTA_LENGTH_BYTE_ARRAY,
            Encoding::DeltaByteArray => ParquetEncoding::DELTA_BYTE_ARRAY,
            Encoding::RleDictionary => ParquetEncoding::RLE_DICTIONARY,
            Encoding::ByteStreamSplit => ParquetEncoding::BYTE_STREAM_SPLIT,
        }
    }
}

/// Enum to annotate whether lists of min/max elements inside ColumnIndex
/// are ordered and if so, in which direction.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub enum BoundaryOrder {
    Unordered,
    Ascending,
    Descending,
}

impl Default for BoundaryOrder {
    fn default() -> Self {
        Self::Unordered
    }
}

impl TryFrom<ParquetBoundaryOrder> for BoundaryOrder {
    type Error = ParquetError;

    fn try_from(encoding: ParquetBoundaryOrder) -> Result<Self, Self::Error> {
        Ok(match encoding {
            ParquetBoundaryOrder::UNORDERED => BoundaryOrder::Unordered,
            ParquetBoundaryOrder::ASCENDING => BoundaryOrder::Ascending,
            ParquetBoundaryOrder::DESCENDING => BoundaryOrder::Descending,
            _ => return Err(ParquetError::oos("BoundaryOrder Thrift value out of range")),
        })
    }
}

impl From<BoundaryOrder> for ParquetBoundaryOrder {
    fn from(encoding: BoundaryOrder) -> Self {
        match encoding {
            BoundaryOrder::Unordered => ParquetBoundaryOrder::UNORDERED,
            BoundaryOrder::Ascending => ParquetBoundaryOrder::ASCENDING,
            BoundaryOrder::Descending => ParquetBoundaryOrder::DESCENDING,
        }
    }
}

pub trait DataPageHeaderExt {
    fn encoding(&self) -> Encoding;
    fn repetition_level_encoding(&self) -> Encoding;
    fn definition_level_encoding(&self) -> Encoding;
}

impl DataPageHeaderExt for DataPageHeader {
    fn encoding(&self) -> Encoding {
        self.encoding.try_into().unwrap()
    }

    fn repetition_level_encoding(&self) -> Encoding {
        self.repetition_level_encoding.try_into().unwrap()
    }

    fn definition_level_encoding(&self) -> Encoding {
        self.definition_level_encoding.try_into().unwrap()
    }
}

impl DataPageHeaderExt for DataPageHeaderV2 {
    fn encoding(&self) -> Encoding {
        self.encoding.try_into().unwrap()
    }

    fn repetition_level_encoding(&self) -> Encoding {
        Encoding::Rle
    }

    fn definition_level_encoding(&self) -> Encoding {
        Encoding::Rle
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum TimeUnit {
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl From<ParquetTimeUnit> for TimeUnit {
    fn from(encoding: ParquetTimeUnit) -> Self {
        match encoding {
            ParquetTimeUnit::MILLIS(_) => TimeUnit::Milliseconds,
            ParquetTimeUnit::MICROS(_) => TimeUnit::Microseconds,
            ParquetTimeUnit::NANOS(_) => TimeUnit::Nanoseconds,
        }
    }
}

impl From<TimeUnit> for ParquetTimeUnit {
    fn from(unit: TimeUnit) -> Self {
        match unit {
            TimeUnit::Milliseconds => ParquetTimeUnit::MILLIS(Default::default()),
            TimeUnit::Microseconds => ParquetTimeUnit::MICROS(Default::default()),
            TimeUnit::Nanoseconds => ParquetTimeUnit::NANOS(Default::default()),
        }
    }
}

/// Enum of all valid logical integer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum IntegerType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum PrimitiveLogicalType {
    String,
    Enum,
    Decimal(usize, usize),
    Date,
    Time {
        unit: TimeUnit,
        is_adjusted_to_utc: bool,
    },
    Timestamp {
        unit: TimeUnit,
        is_adjusted_to_utc: bool,
    },
    Integer(IntegerType),
    Unknown,
    Json,
    Bson,
    Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum GroupLogicalType {
    Map,
    List,
}

impl From<GroupLogicalType> for ParquetLogicalType {
    fn from(type_: GroupLogicalType) -> Self {
        match type_ {
            GroupLogicalType::Map => ParquetLogicalType::MAP(Default::default()),
            GroupLogicalType::List => ParquetLogicalType::LIST(Default::default()),
        }
    }
}

impl From<(i32, bool)> for IntegerType {
    fn from((bit_width, is_signed): (i32, bool)) -> Self {
        match (bit_width, is_signed) {
            (8, true) => IntegerType::Int8,
            (16, true) => IntegerType::Int16,
            (32, true) => IntegerType::Int32,
            (64, true) => IntegerType::Int64,
            (8, false) => IntegerType::UInt8,
            (16, false) => IntegerType::UInt16,
            (32, false) => IntegerType::UInt32,
            (64, false) => IntegerType::UInt64,
            // The above are the only possible annotations for parquet's int32. Anything else
            // is a deviation to the parquet specification and we ignore
            _ => IntegerType::Int32,
        }
    }
}

impl From<IntegerType> for (usize, bool) {
    fn from(type_: IntegerType) -> (usize, bool) {
        match type_ {
            IntegerType::Int8 => (8, true),
            IntegerType::Int16 => (16, true),
            IntegerType::Int32 => (32, true),
            IntegerType::Int64 => (64, true),
            IntegerType::UInt8 => (8, false),
            IntegerType::UInt16 => (16, false),
            IntegerType::UInt32 => (32, false),
            IntegerType::UInt64 => (64, false),
        }
    }
}

impl TryFrom<ParquetLogicalType> for PrimitiveLogicalType {
    type Error = ParquetError;

    fn try_from(type_: ParquetLogicalType) -> Result<Self, Self::Error> {
        Ok(match type_ {
            ParquetLogicalType::STRING(_) => PrimitiveLogicalType::String,
            ParquetLogicalType::ENUM(_) => PrimitiveLogicalType::Enum,
            ParquetLogicalType::DECIMAL(decimal) => PrimitiveLogicalType::Decimal(
                decimal.precision.try_into()?,
                decimal.scale.try_into()?,
            ),
            ParquetLogicalType::DATE(_) => PrimitiveLogicalType::Date,
            ParquetLogicalType::TIME(time) => PrimitiveLogicalType::Time {
                unit: time.unit.into(),
                is_adjusted_to_utc: time.is_adjusted_to_u_t_c,
            },
            ParquetLogicalType::TIMESTAMP(time) => PrimitiveLogicalType::Timestamp {
                unit: time.unit.into(),
                is_adjusted_to_utc: time.is_adjusted_to_u_t_c,
            },
            ParquetLogicalType::INTEGER(int) => {
                PrimitiveLogicalType::Integer((int.bit_width as i32, int.is_signed).into())
            },
            ParquetLogicalType::UNKNOWN(_) => PrimitiveLogicalType::Unknown,
            ParquetLogicalType::JSON(_) => PrimitiveLogicalType::Json,
            ParquetLogicalType::BSON(_) => PrimitiveLogicalType::Bson,
            ParquetLogicalType::UUID(_) => PrimitiveLogicalType::Uuid,
            _ => return Err(ParquetError::oos("LogicalType value out of range")),
        })
    }
}

impl TryFrom<ParquetLogicalType> for GroupLogicalType {
    type Error = ParquetError;

    fn try_from(type_: ParquetLogicalType) -> Result<Self, Self::Error> {
        Ok(match type_ {
            ParquetLogicalType::LIST(_) => GroupLogicalType::List,
            ParquetLogicalType::MAP(_) => GroupLogicalType::Map,
            _ => return Err(ParquetError::oos("LogicalType value out of range")),
        })
    }
}

impl From<PrimitiveLogicalType> for ParquetLogicalType {
    fn from(type_: PrimitiveLogicalType) -> Self {
        match type_ {
            PrimitiveLogicalType::String => ParquetLogicalType::STRING(Default::default()),
            PrimitiveLogicalType::Enum => ParquetLogicalType::ENUM(Default::default()),
            PrimitiveLogicalType::Decimal(precision, scale) => {
                ParquetLogicalType::DECIMAL(DecimalType {
                    precision: precision as i32,
                    scale: scale as i32,
                })
            },
            PrimitiveLogicalType::Date => ParquetLogicalType::DATE(Default::default()),
            PrimitiveLogicalType::Time {
                unit,
                is_adjusted_to_utc,
            } => ParquetLogicalType::TIME(TimeType {
                unit: unit.into(),
                is_adjusted_to_u_t_c: is_adjusted_to_utc,
            }),
            PrimitiveLogicalType::Timestamp {
                unit,
                is_adjusted_to_utc,
            } => ParquetLogicalType::TIMESTAMP(TimestampType {
                unit: unit.into(),
                is_adjusted_to_u_t_c: is_adjusted_to_utc,
            }),
            PrimitiveLogicalType::Integer(integer) => {
                let (bit_width, is_signed) = integer.into();
                ParquetLogicalType::INTEGER(IntType {
                    bit_width: bit_width as i8,
                    is_signed,
                })
            },
            PrimitiveLogicalType::Unknown => ParquetLogicalType::UNKNOWN(Default::default()),
            PrimitiveLogicalType::Json => ParquetLogicalType::JSON(Default::default()),
            PrimitiveLogicalType::Bson => ParquetLogicalType::BSON(Default::default()),
            PrimitiveLogicalType::Uuid => ParquetLogicalType::UUID(Default::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_primitive() -> Result<(), ParquetError> {
        use PrimitiveLogicalType::*;
        let a = vec![
            String,
            Enum,
            Decimal(3, 1),
            Date,
            Time {
                unit: TimeUnit::Milliseconds,
                is_adjusted_to_utc: true,
            },
            Timestamp {
                unit: TimeUnit::Milliseconds,
                is_adjusted_to_utc: true,
            },
            Integer(IntegerType::Int16),
            Unknown,
            Json,
            Bson,
            Uuid,
        ];
        for a in a {
            let c: ParquetLogicalType = a.into();
            let e: PrimitiveLogicalType = c.try_into()?;
            assert_eq!(e, a);
        }
        Ok(())
    }

    #[test]
    fn round_trip_encoding() -> Result<(), ParquetError> {
        use Encoding::*;
        let a = vec![
            Plain,
            PlainDictionary,
            Rle,
            BitPacked,
            DeltaBinaryPacked,
            DeltaLengthByteArray,
            DeltaByteArray,
            RleDictionary,
            ByteStreamSplit,
        ];
        for a in a {
            let c: ParquetEncoding = a.into();
            let e: Encoding = c.try_into()?;
            assert_eq!(e, a);
        }
        Ok(())
    }

    #[test]
    fn round_compression() -> Result<(), ParquetError> {
        use Compression::*;
        let a = vec![Uncompressed, Snappy, Gzip, Lzo, Brotli, Lz4, Zstd, Lz4Raw];
        for a in a {
            let c: CompressionCodec = a.into();
            let e: Compression = c.try_into()?;
            assert_eq!(e, a);
        }
        Ok(())
    }
}
