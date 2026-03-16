//! This module supports mapping SQL datatypes to Polars datatypes.
//!
//! It also provides utility functions for working with SQL datatypes.
use polars_core::datatypes::{DataType, TimeUnit};
use polars_error::{PolarsResult, polars_bail};
use polars_plan::dsl::Expr;
use polars_plan::dsl::functions::lit;
use sqlparser::ast::{
    ArrayElemTypeDef, DataType as SQLDataType, ExactNumberInfo, Ident, ObjectName, ObjectNamePart,
    TimezoneInfo,
};

polars_utils::regex_cache::cached_regex! {
    static DATETIME_LITERAL_RE = r"^\d{4}-[01]\d-[0-3]\d[ T](?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](\.\d{1,9})?)?$";
    static DATE_LITERAL_RE = r"^\d{4}-[01]\d-[0-3]\d$";
    static TIME_LITERAL_RE = r"^(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](\.\d{1,9})?)?$";
}

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
    DATETIME_LITERAL_RE.is_match(value)
}

pub fn is_iso_date(value: &str) -> bool {
    DATE_LITERAL_RE.is_match(value)
}

pub fn is_iso_time(value: &str) -> bool {
    TIME_LITERAL_RE.is_match(value)
}

pub(crate) fn timeunit_from_precision(prec: &Option<u64>) -> PolarsResult<TimeUnit> {
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
        SQLDataType::TinyInt(_) => DataType::Int8,
        SQLDataType::Int16 | SQLDataType::Int2(_) | SQLDataType::SmallInt(_) => DataType::Int16,
        SQLDataType::Int32
        | SQLDataType::Int4(_)
        | SQLDataType::MediumInt(_)
        | SQLDataType::Integer(_)
        | SQLDataType::Int(_) => DataType::Int32,
        SQLDataType::Int64 | SQLDataType::Int8(_) | SQLDataType::BigInt(_) => DataType::Int64,
        SQLDataType::Int128 | SQLDataType::HugeInt => DataType::Int128,

        // ---------------------------------
        // unsigned integer
        // ---------------------------------
        SQLDataType::UTinyInt | SQLDataType::TinyIntUnsigned(_) => DataType::UInt8,
        SQLDataType::Int2Unsigned(_)
        | SQLDataType::SmallIntUnsigned(_)
        | SQLDataType::USmallInt
        | SQLDataType::UInt16 => DataType::UInt16,
        SQLDataType::Int4Unsigned(_) | SQLDataType::MediumIntUnsigned(_) | SQLDataType::UInt32 => {
            DataType::UInt32
        },
        SQLDataType::Int8Unsigned(_)
        | SQLDataType::BigIntUnsigned(_)
        | SQLDataType::UBigInt
        | SQLDataType::UInt64
        | SQLDataType::UInt8 => DataType::UInt64,
        SQLDataType::IntUnsigned(_) | SQLDataType::UnsignedInteger => DataType::UInt32,
        SQLDataType::UHugeInt => DataType::UInt128,

        // ---------------------------------
        // float
        // ---------------------------------
        SQLDataType::Float4 | SQLDataType::Real => DataType::Float32,
        SQLDataType::Double(_) | SQLDataType::DoublePrecision | SQLDataType::Float8 => {
            DataType::Float64
        },
        SQLDataType::Float(n_bytes) => match n_bytes {
            ExactNumberInfo::Precision(n) if (1u64..=24u64).contains(n) => DataType::Float32,
            ExactNumberInfo::Precision(n) if (25u64..=53u64).contains(n) => DataType::Float64,
            ExactNumberInfo::Precision(n) => {
                polars_bail!(SQLSyntax: "unsupported `float` size (expected a value between 1 and 53, found {})", n)
            },
            ExactNumberInfo::None => DataType::Float64,
            ExactNumberInfo::PrecisionAndScale(_, _) => {
                polars_bail!(SQLSyntax: "FLOAT does not support scale parameter")
            },
        },

        // ---------------------------------
        // decimal
        // ---------------------------------
        #[cfg(feature = "dtype-decimal")]
        SQLDataType::Dec(info)
        | SQLDataType::Decimal(info)
        | SQLDataType::BigDecimal(info)
        | SQLDataType::Numeric(info) => match *info {
            ExactNumberInfo::PrecisionAndScale(p, s) => DataType::Decimal(p as usize, s as usize),
            ExactNumberInfo::Precision(p) => DataType::Decimal(p as usize, 0),
            ExactNumberInfo::None => DataType::Decimal(38, 9),
        },

        // ---------------------------------
        // temporal
        // ---------------------------------
        SQLDataType::Date => DataType::Date,
        SQLDataType::Interval { fields, precision } => {
            if !fields.is_none() {
                // eg: "YEARS TO MONTH"
                polars_bail!(SQLInterface: "`interval` definition with fields={:?} is not supported", fields)
            }
            let time_unit = match precision {
                Some(p) if (1u64..=3u64).contains(p) => TimeUnit::Milliseconds,
                Some(p) if (4u64..=6u64).contains(p) => TimeUnit::Microseconds,
                Some(p) if (7u64..=9u64).contains(p) => TimeUnit::Nanoseconds,
                Some(p) => {
                    polars_bail!(SQLSyntax: "invalid `interval` precision (expected 1-9, found {})", p)
                },
                None => TimeUnit::Microseconds,
            };
            DataType::Duration(time_unit)
        },
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
            [ObjectNamePart::Identifier(Ident { value, .. })] => {
                match value.to_lowercase().as_str() {
                    // these integer types are not supported by the PostgreSQL core distribution,
                    // but they ARE available via `pguint` (https://github.com/petere/pguint), an
                    // extension maintained by one of the PostgreSQL core developers, and/or DuckDB.
                    "int1" => DataType::Int8,
                    "uint1" | "utinyint" => DataType::UInt8,
                    "uint2" | "usmallint" => DataType::UInt16,
                    "uint4" | "uinteger" | "uint" => DataType::UInt32,
                    "uint8" | "ubigint" => DataType::UInt64,
                    _ => {
                        polars_bail!(SQLInterface: "datatype {:?} is not currently supported", value)
                    },
                }
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
