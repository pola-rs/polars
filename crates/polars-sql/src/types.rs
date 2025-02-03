//! This module supports mapping SQL datatypes to Polars datatypes.
//!
//! It also provides utility functions for working with SQL datatypes.
use polars_core::datatypes::{DataType, TimeUnit};
use polars_error::{polars_bail, PolarsResult};
use polars_plan::dsl::{lit, Expr};
use regex::{Regex, RegexBuilder};
use sqlparser::ast::{
    ArrayElemTypeDef, DataType as SQLDataType, ExactNumberInfo, Ident, ObjectName, TimezoneInfo,
};

static DATETIME_LITERAL_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
static DATE_LITERAL_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
static TIME_LITERAL_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();

pub fn bitstring_to_bytes_literal(b: &String) -> PolarsResult<Expr> {
    let n_bits = b.len();
    if !b.chars().all(|c| c == '0' || c == '1') || n_bits > 64 {
        polars_bail!(
            SQLSyntax:
            "bit string literal should contain only 0s and 1s and have length <= 64; found '{}' with length {}", b, n_bits
        )
    }
    let s = b.as_str();
    Ok(lit(match n_bits {
        0 => b"".to_vec(),
        1..=8 => u8::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
        9..=16 => u16::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
        17..=32 => u32::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
        _ => u64::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
    }))
}

pub fn is_iso_datetime(value: &str) -> bool {
    let dtm_regex = DATETIME_LITERAL_RE.get_or_init(|| {
        RegexBuilder::new(
            r"^\d{4}-[01]\d-[0-3]\d[ T](?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.\d{1,9})?$",
        )
        .build()
        .unwrap()
    });
    dtm_regex.is_match(value)
}

pub fn is_iso_date(value: &str) -> bool {
    let dt_regex = DATE_LITERAL_RE.get_or_init(|| {
        RegexBuilder::new(r"^\d{4}-[01]\d-[0-3]\d$")
            .build()
            .unwrap()
    });
    dt_regex.is_match(value)
}

pub fn is_iso_time(value: &str) -> bool {
    let tm_regex = TIME_LITERAL_RE.get_or_init(|| {
        RegexBuilder::new(r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.\d{1,9})?$")
            .build()
            .unwrap()
    });
    tm_regex.is_match(value)
}

fn timeunit_from_precision(prec: &Option<u64>) -> PolarsResult<TimeUnit> {
    Ok(match prec {
        None => TimeUnit::Microseconds,
        Some(n) if (1u64..=3u64).contains(n) => TimeUnit::Milliseconds,
        Some(n) if (4u64..=6u64).contains(n) => TimeUnit::Microseconds,
        Some(n) if (7u64..=9u64).contains(n) => TimeUnit::Nanoseconds,
        Some(n) => {
            polars_bail!(SQLSyntax: "invalid temporal type precision (expected 1-9, found {})", n)
        },
    })
}

pub(crate) fn map_sql_dtype_to_polars(dtype: &SQLDataType) -> PolarsResult<DataType> {
    Ok(match dtype {
        // ---------------------------------
        // array/list
        // ---------------------------------
        SQLDataType::Array(ArrayElemTypeDef::AngleBracket(inner_type))
        | SQLDataType::Array(ArrayElemTypeDef::SquareBracket(inner_type, _)) => {
            DataType::List(Box::new(map_sql_dtype_to_polars(inner_type)?))
        },

        // ---------------------------------
        // binary
        // ---------------------------------
        SQLDataType::Bytea
        | SQLDataType::Bytes(_)
        | SQLDataType::Binary(_)
        | SQLDataType::Blob(_)
        | SQLDataType::Varbinary(_) => DataType::Binary,

        // ---------------------------------
        // boolean
        // ---------------------------------
        SQLDataType::Boolean | SQLDataType::Bool => DataType::Boolean,

        // ---------------------------------
        // signed integer
        // ---------------------------------
        SQLDataType::Int(_) | SQLDataType::Integer(_) => DataType::Int32,
        SQLDataType::Int2(_) | SQLDataType::SmallInt(_) => DataType::Int16,
        SQLDataType::Int4(_) | SQLDataType::MediumInt(_) => DataType::Int32,
        SQLDataType::Int8(_) | SQLDataType::BigInt(_) => DataType::Int64,
        SQLDataType::TinyInt(_) => DataType::Int8,

        // ---------------------------------
        // unsigned integer: the following do not map to PostgreSQL types/syntax, but
        // are enabled for wider compatibility (eg: "CAST(col AS BIGINT UNSIGNED)").
        // ---------------------------------
        SQLDataType::UnsignedTinyInt(_) => DataType::UInt8, // see also: "custom" types below
        SQLDataType::UnsignedInt(_) | SQLDataType::UnsignedInteger(_) => DataType::UInt32,
        SQLDataType::UnsignedInt2(_) | SQLDataType::UnsignedSmallInt(_) => DataType::UInt16,
        SQLDataType::UnsignedInt4(_) | SQLDataType::UnsignedMediumInt(_) => DataType::UInt32,
        SQLDataType::UnsignedInt8(_) | SQLDataType::UnsignedBigInt(_) | SQLDataType::UInt8 => {
            DataType::UInt64
        },

        // ---------------------------------
        // float
        // ---------------------------------
        SQLDataType::Double | SQLDataType::DoublePrecision | SQLDataType::Float8 => {
            DataType::Float64
        },
        SQLDataType::Float(n_bytes) => match n_bytes {
            Some(n) if (1u64..=24u64).contains(n) => DataType::Float32,
            Some(n) if (25u64..=53u64).contains(n) => DataType::Float64,
            Some(n) => {
                polars_bail!(SQLSyntax: "unsupported `float` size (expected a value between 1 and 53, found {})", n)
            },
            None => DataType::Float64,
        },
        SQLDataType::Float4 | SQLDataType::Real => DataType::Float32,

        // ---------------------------------
        // decimal
        // ---------------------------------
        #[cfg(feature = "dtype-decimal")]
        SQLDataType::Dec(info) | SQLDataType::Decimal(info) | SQLDataType::Numeric(info) => {
            match *info {
                ExactNumberInfo::PrecisionAndScale(p, s) => {
                    DataType::Decimal(Some(p as usize), Some(s as usize))
                },
                ExactNumberInfo::Precision(p) => DataType::Decimal(Some(p as usize), Some(0)),
                ExactNumberInfo::None => DataType::Decimal(Some(38), Some(9)),
            }
        },

        // ---------------------------------
        // temporal
        // ---------------------------------
        SQLDataType::Date => DataType::Date,
        SQLDataType::Interval => DataType::Duration(TimeUnit::Microseconds),
        SQLDataType::Time(_, tz) => match tz {
            TimezoneInfo::None => DataType::Time,
            _ => {
                polars_bail!(SQLInterface: "`time` with timezone is not supported; found tz={}", tz)
            },
        },
        SQLDataType::Datetime(prec) => DataType::Datetime(timeunit_from_precision(prec)?, None),
        SQLDataType::Timestamp(prec, tz) => match tz {
            TimezoneInfo::None => DataType::Datetime(timeunit_from_precision(prec)?, None),
            _ => {
                polars_bail!(SQLInterface: "`timestamp` with timezone is not (yet) supported")
            },
        },

        // ---------------------------------
        // string
        // ---------------------------------
        SQLDataType::Char(_)
        | SQLDataType::CharVarying(_)
        | SQLDataType::Character(_)
        | SQLDataType::CharacterVarying(_)
        | SQLDataType::Clob(_)
        | SQLDataType::String(_)
        | SQLDataType::Text
        | SQLDataType::Uuid
        | SQLDataType::Varchar(_) => DataType::String,

        // ---------------------------------
        // custom
        // ---------------------------------
        SQLDataType::Custom(ObjectName(idents), _) => match idents.as_slice() {
            [Ident { value, .. }] => match value.to_lowercase().as_str() {
                // these integer types are not supported by the PostgreSQL core distribution,
                // but they ARE available via `pguint` (https://github.com/petere/pguint), an
                // extension maintained by one of the PostgreSQL core developers.
                "uint1" => DataType::UInt8,
                "uint2" => DataType::UInt16,
                "uint4" | "uint" => DataType::UInt32,
                "uint8" => DataType::UInt64,
                // `pguint` also provides a 1 byte (8bit) integer type alias
                "int1" => DataType::Int8,
                _ => {
                    polars_bail!(SQLInterface: "datatype {:?} is not currently supported", value)
                },
            },
            _ => {
                polars_bail!(SQLInterface: "datatype {:?} is not currently supported", idents)
            },
        },
        _ => {
            polars_bail!(SQLInterface: "datatype {:?} is not currently supported", dtype)
        },
    })
}
