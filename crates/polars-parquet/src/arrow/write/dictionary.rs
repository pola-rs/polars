use arrow::array::{Array, DictionaryArray, DictionaryKey};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::DataType;
use parquet2::encoding::hybrid_rle::encode_u32;
use parquet2::encoding::Encoding;
use parquet2::page::{DictPage, Page};
use parquet2::schema::types::PrimitiveType;
use parquet2::statistics::{serialize_statistics, ParquetStatistics};
use parquet2::write::DynIter;
use polars_error::{polars_bail, PolarsResult};

use super::binary::{
    build_statistics as binary_build_statistics, encode_plain as binary_encode_plain,
};
use super::fixed_len_bytes::{
    build_statistics as fixed_binary_build_statistics, encode_plain as fixed_binary_encode_plain,
};
use super::primitive::{
    build_statistics as primitive_build_statistics, encode_plain as primitive_encode_plain,
};
use super::utf8::{build_statistics as utf8_build_statistics, encode_plain as utf8_encode_plain};
use super::{nested, Nested, WriteOptions};
use crate::arrow::read::schema::is_nullable;
use crate::arrow::write::{slice_nested_leaf, utils};

fn serialize_def_levels_simple(
    validity: Option<&Bitmap>,
    length: usize,
    is_optional: bool,
    options: WriteOptions,
    buffer: &mut Vec<u8>,
) -> PolarsResult<()> {
    utils::write_def_levels(buffer, is_optional, validity, length, options.version)
}

fn serialize_keys_values<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    validity: Option<&Bitmap>,
    buffer: &mut Vec<u8>,
) -> PolarsResult<()> {
    let keys = array.keys_values_iter().map(|x| x as u32);
    if let Some(validity) = validity {
        // discard indices whose values are null.
        let keys = keys
            .zip(validity.iter())
            .filter(|&(_key, is_valid)| is_valid)
            .map(|(key, _is_valid)| key);
        let num_bits = utils::get_bit_width(keys.clone().max().unwrap_or(0) as u64);

        let keys = utils::ExactSizedIter::new(keys, array.len() - validity.unset_bits());

        // num_bits as a single byte
        buffer.push(num_bits as u8);

        // followed by the encoded indices.
        Ok(encode_u32(buffer, keys, num_bits)?)
    } else {
        let num_bits = utils::get_bit_width(keys.clone().max().unwrap_or(0) as u64);

        // num_bits as a single byte
        buffer.push(num_bits as u8);

        // followed by the encoded indices.
        Ok(encode_u32(buffer, keys, num_bits)?)
    }
}

fn serialize_levels(
    validity: Option<&Bitmap>,
    length: usize,
    type_: &PrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    buffer: &mut Vec<u8>,
) -> PolarsResult<(usize, usize)> {
    if nested.len() == 1 {
        let is_optional = is_nullable(&type_.field_info);
        serialize_def_levels_simple(validity, length, is_optional, options, buffer)?;
        let definition_levels_byte_length = buffer.len();
        Ok((0, definition_levels_byte_length))
    } else {
        nested::write_rep_and_def(options.version, nested, buffer)
    }
}

fn normalized_validity<K: DictionaryKey>(array: &DictionaryArray<K>) -> Option<Bitmap> {
    match (array.keys().validity(), array.values().validity()) {
        (None, None) => None,
        (None, rhs) => rhs.cloned(),
        (lhs, None) => lhs.cloned(),
        (Some(_), Some(rhs)) => {
            let projected_validity = array
                .keys_iter()
                .map(|x| x.map(|x| rhs.get_bit(x)).unwrap_or(false));
            MutableBitmap::from_trusted_len_iter(projected_validity).into()
        },
    }
}

fn serialize_keys<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    type_: PrimitiveType,
    nested: &[Nested],
    statistics: Option<ParquetStatistics>,
    options: WriteOptions,
) -> PolarsResult<Page> {
    let mut buffer = vec![];

    // parquet only accepts a single validity - we "&" the validities into a single one
    // and ignore keys whole _value_ is null.
    let validity = normalized_validity(array);
    let (start, len) = slice_nested_leaf(nested);

    let mut nested = nested.to_vec();
    let array = array.clone().sliced(start, len);
    if let Some(Nested::Primitive(_, _, c)) = nested.last_mut() {
        *c = len;
    } else {
        unreachable!("")
    }

    let (repetition_levels_byte_length, definition_levels_byte_length) = serialize_levels(
        validity.as_ref(),
        array.len(),
        &type_,
        &nested,
        options,
        &mut buffer,
    )?;

    serialize_keys_values(&array, validity.as_ref(), &mut buffer)?;

    let (num_values, num_rows) = if nested.len() == 1 {
        (array.len(), array.len())
    } else {
        (nested::num_values(&nested), nested[0].len())
    };

    utils::build_plain_page(
        buffer,
        num_values,
        num_rows,
        array.null_count(),
        repetition_levels_byte_length,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        Encoding::RleDictionary,
    )
    .map(Page::Data)
}

macro_rules! dyn_prim {
    ($from:ty, $to:ty, $array:expr, $options:expr, $type_:expr) => {{
        let values = $array.values().as_any().downcast_ref().unwrap();

        let buffer = primitive_encode_plain::<$from, $to>(values, false, vec![]);

        let stats: Option<ParquetStatistics> = if $options.write_statistics {
            let mut stats = primitive_build_statistics::<$from, $to>(values, $type_.clone());
            stats.null_count = Some($array.null_count() as i64);
            let stats = serialize_statistics(&stats);
            Some(stats)
        } else {
            None
        };
        (DictPage::new(buffer, values.len(), false), stats)
    }};
}

pub fn array_to_pages<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    type_: PrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<DynIter<'static, PolarsResult<Page>>> {
    match encoding {
        Encoding::PlainDictionary | Encoding::RleDictionary => {
            // write DictPage
            let (dict_page, statistics): (_, Option<ParquetStatistics>) =
                match array.values().data_type().to_logical_type() {
                    DataType::Int8 => dyn_prim!(i8, i32, array, options, type_),
                    DataType::Int16 => dyn_prim!(i16, i32, array, options, type_),
                    DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
                        dyn_prim!(i32, i32, array, options, type_)
                    },
                    DataType::Int64
                    | DataType::Date64
                    | DataType::Time64(_)
                    | DataType::Timestamp(_, _)
                    | DataType::Duration(_) => dyn_prim!(i64, i64, array, options, type_),
                    DataType::UInt8 => dyn_prim!(u8, i32, array, options, type_),
                    DataType::UInt16 => dyn_prim!(u16, i32, array, options, type_),
                    DataType::UInt32 => dyn_prim!(u32, i32, array, options, type_),
                    DataType::UInt64 => dyn_prim!(u64, i64, array, options, type_),
                    DataType::Float32 => dyn_prim!(f32, f32, array, options, type_),
                    DataType::Float64 => dyn_prim!(f64, f64, array, options, type_),
                    DataType::Utf8 => {
                        let array = array.values().as_any().downcast_ref().unwrap();

                        let mut buffer = vec![];
                        utf8_encode_plain::<i32>(array, false, &mut buffer);
                        let stats = if options.write_statistics {
                            Some(utf8_build_statistics(array, type_.clone()))
                        } else {
                            None
                        };
                        (DictPage::new(buffer, array.len(), false), stats)
                    },
                    DataType::LargeUtf8 => {
                        let array = array.values().as_any().downcast_ref().unwrap();

                        let mut buffer = vec![];
                        utf8_encode_plain::<i64>(array, false, &mut buffer);
                        let stats = if options.write_statistics {
                            Some(utf8_build_statistics(array, type_.clone()))
                        } else {
                            None
                        };
                        (DictPage::new(buffer, array.len(), false), stats)
                    },
                    DataType::Binary => {
                        let array = array.values().as_any().downcast_ref().unwrap();

                        let mut buffer = vec![];
                        binary_encode_plain::<i32>(array, false, &mut buffer);
                        let stats = if options.write_statistics {
                            Some(binary_build_statistics(array, type_.clone()))
                        } else {
                            None
                        };
                        (DictPage::new(buffer, array.len(), false), stats)
                    },
                    DataType::LargeBinary => {
                        let values = array.values().as_any().downcast_ref().unwrap();

                        let mut buffer = vec![];
                        binary_encode_plain::<i64>(values, false, &mut buffer);
                        let stats = if options.write_statistics {
                            let mut stats = binary_build_statistics(values, type_.clone());
                            stats.null_count = Some(array.null_count() as i64);
                            Some(stats)
                        } else {
                            None
                        };
                        (DictPage::new(buffer, values.len(), false), stats)
                    },
                    DataType::FixedSizeBinary(_) => {
                        let mut buffer = vec![];
                        let array = array.values().as_any().downcast_ref().unwrap();
                        fixed_binary_encode_plain(array, false, &mut buffer);
                        let stats = if options.write_statistics {
                            let mut stats = fixed_binary_build_statistics(array, type_.clone());
                            stats.null_count = Some(array.null_count() as i64);
                            Some(serialize_statistics(&stats))
                        } else {
                            None
                        };
                        (DictPage::new(buffer, array.len(), false), stats)
                    },
                    other => {
                        polars_bail!(nyi =
                            "Writing dictionary arrays to parquet only support data type {other:?}"
                        )
                    },
                };
            let dict_page = Page::Dict(dict_page);

            // write DataPage pointing to DictPage
            let data_page = serialize_keys(array, type_, nested, statistics, options)?;

            let iter = std::iter::once(Ok(dict_page)).chain(std::iter::once(Ok(data_page)));
            Ok(DynIter::new(Box::new(iter)))
        },
        _ => polars_bail!(nyi = "Dictionary arrays only support dictionary encoding"),
    }
}
