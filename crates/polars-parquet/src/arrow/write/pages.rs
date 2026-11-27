use std::fmt::Debug;

use polars_arrow::array::{Array, FixedSizeListArray, ListArray, MapArray, StructArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::datatypes::{ArrowDataType, PhysicalType};
use polars_arrow::offset::{Offset, OffsetsBuffer};
use polars_error::{PolarsResult, polars_bail};

use super::{Encoding, WriteOptions, array_to_pages};
use crate::arrow::read::schema::is_nullable;
use crate::parquet::page::Page;
use crate::parquet::schema::types::{ParquetType, PrimitiveType as ParquetPrimitiveType};
use crate::write::DynIter;

#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveNested {
    pub is_optional: bool,
    pub validity: Option<Bitmap>,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ListNested<O: Offset> {
    pub is_optional: bool,
    pub offsets: OffsetsBuffer<O>,
    pub validity: Option<Bitmap>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedSizeListNested {
    pub validity: Option<Bitmap>,
    pub is_optional: bool,
    pub width: usize,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructNested {
    pub is_optional: bool,
    pub validity: Option<Bitmap>,
    pub length: usize,
}

impl<O: Offset> ListNested<O> {
    pub fn new(offsets: OffsetsBuffer<O>, validity: Option<Bitmap>, is_optional: bool) -> Self {
        Self {
            is_optional,
            offsets,
            validity,
        }
    }
}

/// Descriptor of nested information of a field
#[derive(Debug, Clone, PartialEq)]
pub enum Nested {
    /// a primitive (leaf or parquet column)
    Primitive(PrimitiveNested),
    List(ListNested<i32>),
    LargeList(ListNested<i64>),
    FixedSizeList(FixedSizeListNested),
    Struct(StructNested),
}

impl Nested {
    /// Returns the length (number of rows) of the element
    pub fn len(&self) -> usize {
        match self {
            Nested::Primitive(nested) => nested.length,
            Nested::List(nested) => nested.offsets.len_proxy(),
            Nested::LargeList(nested) => nested.offsets.len_proxy(),
            Nested::FixedSizeList(nested) => nested.length,
            Nested::Struct(nested) => nested.length,
        }
    }

    pub fn primitive(validity: Option<Bitmap>, is_optional: bool, length: usize) -> Self {
        Self::Primitive(PrimitiveNested {
            validity,
            is_optional,
            length,
        })
    }

    pub fn list(validity: Option<Bitmap>, is_optional: bool, offsets: OffsetsBuffer<i32>) -> Self {
        Self::List(ListNested {
            validity,
            is_optional,
            offsets,
        })
    }

    pub fn large_list(
        validity: Option<Bitmap>,
        is_optional: bool,
        offsets: OffsetsBuffer<i64>,
    ) -> Self {
        Self::LargeList(ListNested {
            validity,
            is_optional,
            offsets,
        })
    }

    pub fn fixed_size_list(
        validity: Option<Bitmap>,
        is_optional: bool,
        width: usize,
        length: usize,
    ) -> Self {
        Self::FixedSizeList(FixedSizeListNested {
            validity,
            is_optional,
            width,
            length,
        })
    }

    pub fn structure(validity: Option<Bitmap>, is_optional: bool, length: usize) -> Self {
        Self::Struct(StructNested {
            validity,
            is_optional,
            length,
        })
    }
}

/// Constructs the necessary `Vec<Vec<Nested>>` to write the rep and def levels of `array` to parquet
pub fn to_nested(array: &dyn Array, type_: &ParquetType) -> PolarsResult<Vec<Vec<Nested>>> {
    let mut nested = vec![];

    to_nested_recursive(array, type_, &mut nested, vec![])?;
    Ok(nested)
}

fn to_nested_recursive(
    array: &dyn Array,
    type_: &ParquetType,
    nested: &mut Vec<Vec<Nested>>,
    mut parents: Vec<Nested>,
) -> PolarsResult<()> {
    let is_optional = is_nullable(type_.get_field_info());

    if !is_optional && array.null_count() > 0 {
        polars_bail!(InvalidOperation: "writing a missing value to required field '{}'", type_.name());
    }

    use PhysicalType::*;
    match array.dtype().to_physical_type() {
        Struct => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let fields = if let ParquetType::GroupType { fields, .. } = type_ {
                fields
            } else {
                // @NOTE: Support empty struct by mapping to Boolean array.
                if let ArrowDataType::Struct(fs) = array.dtype()
                    && fs.is_empty()
                {
                    parents.push(Nested::Primitive(PrimitiveNested {
                        validity: array.validity().cloned(),
                        is_optional,
                        length: array.len(),
                    }));
                    nested.push(parents);
                    return Ok(());
                }

                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a struct array",
                )
            };

            parents.push(Nested::Struct(StructNested {
                is_optional,
                validity: array.validity().cloned(),
                length: array.len(),
            }));

            for (type_, array) in fields.iter().zip(array.values()) {
                to_nested_recursive(array.as_ref(), type_, nested, parents.clone())?;
            }
        },
        FixedSizeList => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array",
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array",
                )
            };

            parents.push(Nested::FixedSizeList(FixedSizeListNested {
                validity: array.validity().cloned(),
                length: array.len(),
                width: array.size(),
                is_optional,
            }));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        List => {
            let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array",
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array",
                )
            };

            parents.push(Nested::List(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        LargeList => {
            let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array",
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array",
                )
            };

            parents.push(Nested::LargeList(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        Map => {
            let array = array.as_any().downcast_ref::<MapArray>().unwrap();

            let entry_types = if let ParquetType::GroupType { fields, .. } = type_
                && let [ParquetType::GroupType { fields, .. }] = fields.as_slice()
            {
                fields
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group holding a single group for a map array",
                )
            };

            let entries = array.field();
            let entries = entries.as_any().downcast_ref::<StructArray>().unwrap();

            parents.push(Nested::List(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));

            parents.push(Nested::Struct(StructNested {
                is_optional: false,
                validity: entries.validity().cloned(),
                length: entries.len(),
            }));

            for (type_, array) in entry_types.iter().zip(entries.values()) {
                to_nested_recursive(array.as_ref(), type_, nested, parents.clone())?;
            }
        },
        _ => {
            parents.push(Nested::Primitive(PrimitiveNested {
                validity: array.validity().cloned(),
                is_optional,
                length: array.len(),
            }));
            nested.push(parents)
        },
    }
    Ok(())
}

fn expand_list_validity<'a, O: Offset>(
    array: &'a ListArray<O>,
    validity: BitmapState,
    array_stack: &mut Vec<(&'a dyn Array, BitmapState)>,
) {
    expand_nested_validity(
        array.len(),
        array.offsets(),
        array.values().as_ref(),
        validity,
        array_stack,
    )
}

/// Push `values` with the outer validity expanded through `offsets`, so that it applies to
/// each element rather than to each row.
fn expand_nested_validity<'a, O: Offset>(
    len: usize,
    offsets_buffer: &OffsetsBuffer<O>,
    values: &'a dyn Array,
    validity: BitmapState,
    array_stack: &mut Vec<(&'a dyn Array, BitmapState)>,
) {
    let BitmapState::SomeSet(list_validity) = validity else {
        array_stack.push((
            values,
            match validity {
                BitmapState::AllSet => BitmapState::AllSet,
                BitmapState::SomeSet(_) => unreachable!(),
                BitmapState::AllUnset(_) => BitmapState::AllUnset(values.len()),
            },
        ));
        return;
    };

    let offsets = offsets_buffer.buffer();
    let mut validity = MutableBitmap::with_capacity(values.len());
    let mut list_validity_iter = list_validity.iter();

    // @NOTE: We need to take into account here that the list might only point to a slice of the
    // values, therefore we need to extend the validity mask with dummy values to match the length
    // of the values array.

    let mut idx = 0;
    validity.extend_constant(offsets[0].to_usize(), false);
    while list_validity_iter.num_remaining() > 0 {
        let num_ones = list_validity_iter.take_leading_ones();
        let num_elements = offsets[idx + num_ones] - offsets[idx];
        validity.extend_constant(num_elements.to_usize(), true);

        idx += num_ones;

        let num_zeros = list_validity_iter.take_leading_zeros();
        let num_elements = offsets[idx + num_zeros] - offsets[idx];
        validity.extend_constant(num_elements.to_usize(), false);

        idx += num_zeros;
    }
    validity.extend_constant(values.len() - validity.len(), false);

    debug_assert_eq!(idx, len);
    let validity = validity.freeze();

    debug_assert_eq!(validity.len(), values.len());
    array_stack.push((values, BitmapState::SomeSet(validity)));
}

#[derive(Clone)]
enum BitmapState {
    AllSet,
    SomeSet(Bitmap),
    AllUnset(usize),
}

impl From<Option<&Bitmap>> for BitmapState {
    fn from(bm: Option<&Bitmap>) -> Self {
        let Some(bm) = bm else {
            return Self::AllSet;
        };

        let null_count = bm.unset_bits();

        if null_count == 0 {
            Self::AllSet
        } else if null_count == bm.len() {
            Self::AllUnset(bm.len())
        } else {
            Self::SomeSet(bm.clone())
        }
    }
}

impl From<BitmapState> for Option<Bitmap> {
    fn from(bms: BitmapState) -> Self {
        match bms {
            BitmapState::AllSet => None,
            BitmapState::SomeSet(bm) => Some(bm),
            BitmapState::AllUnset(len) => Some(Bitmap::new_zeroed(len)),
        }
    }
}

impl std::ops::BitAnd for &BitmapState {
    type Output = BitmapState;

    fn bitand(self, rhs: Self) -> Self::Output {
        use BitmapState as B;
        match (self, rhs) {
            (B::AllSet, B::AllSet) => B::AllSet,
            (B::AllSet, B::SomeSet(v)) | (B::SomeSet(v), B::AllSet) => B::SomeSet(v.clone()),
            (B::SomeSet(lhs), B::SomeSet(rhs)) => {
                let result = lhs & rhs;
                let null_count = result.unset_bits();

                if null_count == 0 {
                    B::AllSet
                } else if null_count == result.len() {
                    B::AllUnset(result.len())
                } else {
                    B::SomeSet(result)
                }
            },
            (B::AllUnset(len), _) | (_, B::AllUnset(len)) => B::AllUnset(*len),
        }
    }
}

/// Convert [`Array`] to a `Vec<Box<dyn Array>>` leaves in DFS order.
///
/// Each leaf array has the validity propagated from the nesting levels above.
pub fn to_leaves(array: &dyn Array, leaves: &mut Vec<Box<dyn Array>>) {
    use PhysicalType as P;

    leaves.clear();
    let mut array_stack: Vec<(&dyn Array, BitmapState)> = Vec::new();

    array_stack.push((array, BitmapState::AllSet));

    while let Some((array, inherited_validity)) = array_stack.pop() {
        let child_validity = BitmapState::from(array.validity());
        let validity = (&child_validity) & (&inherited_validity);

        match array.dtype().to_physical_type() {
            P::Struct if !matches!(array.dtype(), ArrowDataType::Struct(fs) if fs.is_empty()) => {
                let array = array.as_any().downcast_ref::<StructArray>().unwrap();

                leaves.reserve(array.len().saturating_sub(1));
                array
                    .values()
                    .iter()
                    .rev()
                    .for_each(|field| array_stack.push((field.as_ref(), validity.clone())));
            },
            P::List => {
                let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
                expand_list_validity(array, validity, &mut array_stack);
            },
            P::LargeList => {
                let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
                expand_list_validity(array, validity, &mut array_stack);
            },
            P::FixedSizeList => {
                let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

                let BitmapState::SomeSet(fsl_validity) = validity else {
                    array_stack.push((
                        array.values().as_ref(),
                        match validity {
                            BitmapState::AllSet => BitmapState::AllSet,
                            BitmapState::SomeSet(_) => unreachable!(),
                            BitmapState::AllUnset(_) => BitmapState::AllUnset(array.values().len()),
                        },
                    ));
                    continue;
                };

                let num_values = array.values().len();
                let size = array.size();

                let mut validity = MutableBitmap::with_capacity(num_values);
                let mut fsl_validity_iter = fsl_validity.iter();

                let mut idx = 0;
                while fsl_validity_iter.num_remaining() > 0 {
                    let num_ones = fsl_validity_iter.take_leading_ones();
                    let num_elements = num_ones * size;
                    validity.extend_constant(num_elements, true);

                    idx += num_ones;

                    let num_zeros = fsl_validity_iter.take_leading_zeros();
                    let num_elements = num_zeros * size;
                    validity.extend_constant(num_elements, false);

                    idx += num_zeros;
                }

                debug_assert_eq!(idx, array.len());

                let validity = BitmapState::SomeSet(validity.freeze());

                array_stack.push((array.values().as_ref(), validity));
            },
            P::Map => {
                let array = array.as_any().downcast_ref::<MapArray>().unwrap();

                expand_nested_validity(
                    array.len(),
                    array.offsets(),
                    array.field().as_ref(),
                    validity,
                    &mut array_stack,
                );
            },
            P::Null
            | P::Boolean
            | P::Primitive(_)
            | P::Binary
            | P::FixedSizeBinary
            | P::LargeBinary
            | P::Utf8
            | P::LargeUtf8
            | P::Dictionary(_)
            | P::BinaryView
            | P::Utf8View
            | P::Struct => {
                leaves.push(array.with_validity(validity.into()));
            },

            other => todo!("Writing {:?} to parquet not yet implemented", other),
        }
    }
}

/// Convert `ParquetType` to `Vec<ParquetPrimitiveType>` leaves in DFS order.
pub fn to_parquet_leaves(type_: ParquetType) -> Vec<ParquetPrimitiveType> {
    let mut leaves = vec![];
    to_parquet_leaves_recursive(type_, &mut leaves);
    leaves
}

fn to_parquet_leaves_recursive(type_: ParquetType, leaves: &mut Vec<ParquetPrimitiveType>) {
    match type_ {
        ParquetType::PrimitiveType(primitive) => leaves.push(primitive),
        ParquetType::GroupType { fields, .. } => {
            fields
                .into_iter()
                .for_each(|type_| to_parquet_leaves_recursive(type_, leaves));
        },
    }
}

/// Returns a vector of iterators of [`Page`], one per leaf column in the array
pub fn array_to_columns<A: AsRef<dyn Array> + Send + Sync>(
    array: A,
    type_: ParquetType,
    options: WriteOptions,
    encoding: &[Encoding],
) -> PolarsResult<Vec<DynIter<'static, PolarsResult<Page>>>> {
    let array = array.as_ref();

    let nested = to_nested(array, &type_)?;

    let types = to_parquet_leaves(type_);

    let mut values = Vec::new();
    to_leaves(array, &mut values);

    assert_eq!(encoding.len(), types.len());

    let x = values
        .iter()
        .zip(nested)
        .zip(types)
        .zip(encoding.iter())
        .map(|(((values, nested), type_), encoding)| {
            array_to_pages(values.as_ref(), type_, &nested, options, *encoding)
        })
        .collect::<PolarsResult<Vec<DynIter<'static, PolarsResult<Page>>>>>()?;
    Ok(x)
}

#[cfg(test)]
mod tests {
    use polars_arrow::array::*;
    use polars_arrow::datatypes::*;

    use super::super::{
        CompressionOptions, FieldInfo, ParquetPhysicalType, StatisticsOptions, Version,
        to_parquet_type,
    };
    use super::*;
    use crate::parquet::schema::Repetition;
    use crate::parquet::schema::types::{
        GroupLogicalType, PrimitiveConvertedType, PrimitiveLogicalType,
    };

    #[test]
    fn string_views_and_offsets_produce_the_same_plain_pages() {
        let values = [
            Some("abc"),
            None,
            Some(""),
            Some("long enough to be external"),
            Some("tail"),
        ];
        let pages = |array: Box<dyn Array>| {
            let field = Field::new("s".into(), array.dtype().clone(), true);
            let options = WriteOptions {
                statistics: StatisticsOptions::default(),
                version: Version::V1,
                compression: CompressionOptions::Uncompressed,
                data_page_size: Some(32),
            };
            let mut columns = array_to_columns(
                array,
                to_parquet_type(&field).unwrap(),
                options,
                &[Encoding::Plain],
            )
            .unwrap();
            columns
                .pop()
                .unwrap()
                .map(|page| {
                    let crate::parquet::page::Page::Data(page) = page.unwrap() else {
                        panic!("expected a plain data page");
                    };
                    (page.num_values(), page.buffer().to_vec())
                })
                .collect::<Vec<_>>()
        };
        let offsets = pages(Utf8Array::<i64>::from_iter(values).boxed());
        assert!(offsets.len() > 1);
        assert_eq!(pages(Utf8ViewArray::from_slice(values).boxed()), offsets);
    }

    #[test]
    fn dictionary_statistics_only_include_referenced_values() {
        use crate::parquet::page::Page;
        use crate::parquet::statistics::Statistics;

        let plain = Int32Array::from([Some(1), None, Some(3), None]);
        let dict = DictionaryArray::try_from_keys(
            UInt32Array::from([Some(1), None, Some(3), None]),
            Int32Array::from([Some(-99), Some(1), None, Some(3), Some(99)]).boxed(),
            false,
        )
        .unwrap();
        for array in [plain.boxed(), dict.boxed()] {
            let type_ =
                to_parquet_type(&Field::new("a".into(), array.dtype().clone(), true)).unwrap();
            let options = WriteOptions {
                statistics: StatisticsOptions::full(),
                version: Version::V1,
                compression: CompressionOptions::Uncompressed,
                data_page_size: None,
            };
            let mut columns =
                array_to_columns(array, type_, options, &[Encoding::RleDictionary]).unwrap();
            assert!(matches!(columns[0].next(), Some(Ok(Page::Dict(_)))));
            let Some(Ok(Page::Data(page))) = columns[0].next() else {
                panic!("expected a dictionary-encoded data page");
            };
            let Statistics::Int32(stats) = page.statistics().unwrap().unwrap() else {
                panic!("expected Int32 statistics");
            };
            assert_eq!(
                (stats.min_value, stats.max_value, stats.null_count),
                (Some(1), Some(3), Some(2))
            );
        }
    }

    #[test]
    fn nested_dictionary_uses_one_dictionary_and_splits_skewed_rows() {
        for (offsets, page_size, expected_rows) in [
            (vec![0, 1, 2, 130, 131, 132], 32, vec![2, 1, 2]),
            (vec![0, 2, 3, 6, 6, 7], 20, vec![1, 1, 1, 2]),
        ] {
            let values =
                Utf8ViewArray::from_slice_values(vec!["repeat"; *offsets.last().unwrap() as usize]);
            let dtype = ListArray::<i64>::default_datatype(ArrowDataType::Utf8View);
            let array =
                ListArray::<i64>::new(dtype, offsets.try_into().unwrap(), values.boxed(), None);
            for version in [Version::V1, Version::V2] {
                let field = Field::new("s".into(), array.dtype().clone(), true);
                let options = WriteOptions {
                    statistics: StatisticsOptions::default(),
                    version,
                    compression: CompressionOptions::Uncompressed,
                    data_page_size: Some(page_size),
                };
                let mut columns = array_to_columns(
                    array.clone().boxed(),
                    to_parquet_type(&field).unwrap(),
                    options,
                    &[Encoding::RleDictionary],
                )
                .unwrap();
                let pages = columns
                    .pop()
                    .unwrap()
                    .collect::<PolarsResult<Vec<_>>>()
                    .unwrap();
                assert_eq!(
                    pages.iter().filter(|p| matches!(p, Page::Dict(_))).count(),
                    1
                );
                let rows = pages
                    .iter()
                    .filter_map(|page| match page {
                        Page::Data(page) => {
                            assert_eq!(page.encoding(), Encoding::RleDictionary);
                            Some(page.num_rows().unwrap())
                        },
                        Page::Dict(_) => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(rows, expected_rows);
            }
        }
    }

    #[test]
    fn nested_dictionary_falls_back_for_unique_values() {
        let values = (0..1024).map(|i| format!("unique-{i}")).collect::<Vec<_>>();
        let values = Utf8ViewArray::from_slice_values(values);
        let array = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(ArrowDataType::Utf8View),
            vec![0, 512, 1024].try_into().unwrap(),
            values.boxed(),
            None,
        );
        let field = Field::new("s".into(), array.dtype().clone(), true);
        let options = WriteOptions {
            statistics: StatisticsOptions::default(),
            version: Version::V1,
            compression: CompressionOptions::Uncompressed,
            data_page_size: Some(1024),
        };
        let mut columns = array_to_columns(
            array.boxed(),
            to_parquet_type(&field).unwrap(),
            options,
            &[Encoding::RleDictionary],
        )
        .unwrap();
        for page in columns.pop().unwrap() {
            let Page::Data(page) = page.unwrap() else {
                panic!("expected plain page")
            };
            assert_eq!(page.encoding(), Encoding::Plain);
        }
    }

    fn round_trip_nested_dictionary(array: Box<dyn Array>, version: Version, page_size: usize) {
        use std::io::Cursor;
        use std::sync::Arc;

        use polars_arrow::record_batch::RecordBatchT;
        use polars_buffer::Buffer;

        use crate::parquet::read::{BasicDecompressor, PageReader};
        use crate::read::{column_iter_to_arrays, read_metadata};
        use crate::write::{FileWriter, RowGroupIterator, get_dtype_encoding};

        let field = Field::new("nested".into(), array.dtype().clone(), true);
        let schema = ArrowSchema::from_iter([field.clone()]);
        let options = WriteOptions {
            statistics: StatisticsOptions::default(),
            version,
            compression: CompressionOptions::Uncompressed,
            data_page_size: Some(page_size),
        };
        let batch = RecordBatchT::new(array.len(), Arc::new(schema.clone()), vec![array.clone()]);
        let groups = RowGroupIterator::try_new(
            std::iter::once(Ok(batch)),
            &schema,
            options,
            Buffer::from_iter([get_dtype_encoding(array.dtype())]),
        )
        .unwrap();
        let mut writer = FileWriter::try_new(Vec::new(), schema, options, None).unwrap();
        for group in groups {
            writer.write(array.len() as u64, group.unwrap()).unwrap();
        }
        writer.end(None).unwrap();
        let bytes = writer.into_inner();
        let metadata = read_metadata(&mut Cursor::new(&bytes)).unwrap();
        let columns = metadata.row_groups[0]
            .columns_under_root_iter("nested")
            .unwrap();
        let (columns, types): (Vec<_>, Vec<_>) = columns
            .map(|column| {
                let range = column.byte_range().unwrap();
                let chunk = bytes[range.start as usize..range.end as usize].to_vec();
                let size = chunk.len();
                let pages = PageReader::new(
                    Cursor::new(Buffer::from_vec(chunk)),
                    column,
                    Vec::new(),
                    size * 2 + 1024,
                )
                .unwrap();
                (
                    BasicDecompressor::new(pages, Vec::new()),
                    &column.descriptor().descriptor.primitive_type,
                )
            })
            .unzip();
        let (actual, _) = column_iter_to_arrays(columns, types, field, None).unwrap();
        let actual = polars_arrow::compute::concatenate::concatenate(
            &actual.iter().map(|a| a.as_ref()).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(actual.as_ref(), array.as_ref());
    }

    #[test]
    fn nested_dictionary_round_trips_nulls_slices_and_multiple_pages() {
        let mut values = vec![
            Some("x"),
            Some(""),
            Some("hidden"),
            None,
            Some("é"),
            Some("x"),
        ];
        values.extend(std::iter::repeat_n(Some("long repeated value"), 100));
        let values = Utf8ViewArray::from_slice(values);
        let offsets = vec![0i64, 2, 2, 3, 5, 6, 106];
        let validity = Some(Bitmap::from([true, true, false, true, true, true]));
        let strings = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(ArrowDataType::Utf8View),
            offsets.clone().try_into().unwrap(),
            values.clone().boxed(),
            validity.clone(),
        )
        .boxed();
        let binary = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(ArrowDataType::BinaryView),
            offsets.clone().try_into().unwrap(),
            values.to_binview().boxed(),
            validity.clone(),
        )
        .boxed();
        let lists = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(strings.dtype().clone()),
            vec![0i64, 2, 2, 4, 6].try_into().unwrap(),
            strings.clone(),
            None,
        )
        .boxed();
        let fields = vec![
            Field::new("s".into(), strings.dtype().clone(), true),
            Field::new("b".into(), binary.dtype().clone(), true),
        ];
        let structs = StructArray::new(
            ArrowDataType::Struct(fields),
            6,
            vec![strings.clone(), binary.clone()],
            Some(Bitmap::from([true, true, true, false, true, true])),
        )
        .boxed();
        let keys = Utf8ViewArray::from_slice_values(
            (0..106)
                .map(|i| format!("key-{}", i % 100))
                .collect::<Vec<_>>(),
        );
        let entries = StructArray::new(
            ArrowDataType::Struct(vec![
                Field::new("key".into(), ArrowDataType::Utf8View, false),
                Field::new("value".into(), ArrowDataType::Utf8View, true),
            ]),
            106,
            vec![keys.boxed(), values.boxed()],
            None,
        );
        let map = MapArray::new(
            ArrowDataType::Map(
                Box::new(Field::new("entries".into(), entries.dtype().clone(), false)),
                false,
            ),
            offsets
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            entries.boxed(),
            validity,
        )
        .boxed();
        let empty = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(ArrowDataType::Utf8View),
            vec![0i64; 4].try_into().unwrap(),
            Utf8ViewArray::new_empty(ArrowDataType::Utf8View).boxed(),
            Some(Bitmap::from([true, false, true])),
        )
        .boxed();
        let nulls = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(ArrowDataType::Utf8View),
            vec![0i64, 2, 3, 4].try_into().unwrap(),
            Utf8ViewArray::new_null(ArrowDataType::Utf8View, 4).boxed(),
            None,
        )
        .boxed();
        for array in [strings, binary, lists, structs, map, empty, nulls] {
            let sliced = array.sliced(1, array.len() - 1);
            for array in [array, sliced] {
                for version in [Version::V1, Version::V2] {
                    for page_size in [16, 128] {
                        round_trip_nested_dictionary(array.clone(), version, page_size);
                    }
                }
            }
        }
    }

    #[test]
    fn test_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b".into(), ArrowDataType::Boolean, false),
            Field::new("c".into(), ArrowDataType::Int32, false),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            4,
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".into(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".into(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".into(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };
        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                vec![
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_struct_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b".into(), ArrowDataType::Boolean, false),
            Field::new("c".into(), ArrowDataType::Int32, false),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            4,
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let fields = vec![
            Field::new("b".into(), array.dtype().clone(), true),
            Field::new("c".into(), array.dtype().clone(), true),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            4,
            vec![Box::new(array.clone()), Box::new(array)],
            None,
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".into(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".into(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".into(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".into(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![type_.clone(), type_],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                // a.b.b
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                // a.b.c
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                // a.c.b
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                // a.c.c
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_list_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b".into(), ArrowDataType::Boolean, false),
            Field::new("c".into(), ArrowDataType::Int32, false),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            4,
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let array = ListArray::new(
            ArrowDataType::List(Box::new(Field::new(
                "l".into(),
                array.dtype().clone(),
                true,
            ))),
            vec![0i32, 2, 4].try_into().unwrap(),
            Box::new(array),
            None,
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".into(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".into(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".into(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "l".into(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![ParquetType::GroupType {
                field_info: FieldInfo {
                    name: "list".into(),
                    repetition: Repetition::Repeated,
                    id: None,
                },
                logical_type: None,
                converted_type: None,
                fields: vec![type_],
            }],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 4].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 4].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_map() {
        let kv_type = ArrowDataType::Struct(vec![
            Field::new("k".into(), ArrowDataType::Utf8, false),
            Field::new("v".into(), ArrowDataType::Int32, false),
        ]);
        let kv_field = Field::new("kv".into(), kv_type.clone(), false);
        let map_type = ArrowDataType::Map(Box::new(kv_field), false);

        let key_array = Utf8Array::<i32>::from_slice(["k1", "k2", "k3", "k4", "k5", "k6"]).boxed();
        let val_array = Int32Array::from_slice([42, 28, 19, 31, 21, 17]).boxed();
        let kv_array = StructArray::try_new(kv_type, 6, vec![key_array, val_array], None)
            .unwrap()
            .boxed();
        let offsets = OffsetsBuffer::try_from(vec![0, 2, 3, 4, 6]).unwrap();

        let array = MapArray::try_new(map_type, offsets, kv_array, None).unwrap();

        // Per the specification the repeated `key_value` group holds the key and the value
        // directly; it *is* the entries struct, so there is no group in between.
        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "m".into(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: Some(GroupLogicalType::Map),
            converted_type: None,
            fields: vec![ParquetType::GroupType {
                field_info: FieldInfo {
                    name: "key_value".into(),
                    repetition: Repetition::Repeated,
                    id: None,
                },
                logical_type: None,
                converted_type: None,
                fields: vec![
                    ParquetType::PrimitiveType(ParquetPrimitiveType {
                        field_info: FieldInfo {
                            name: "k".into(),
                            repetition: Repetition::Required,
                            id: None,
                        },
                        logical_type: Some(PrimitiveLogicalType::String),
                        converted_type: Some(PrimitiveConvertedType::Utf8),
                        physical_type: ParquetPhysicalType::ByteArray,
                    }),
                    ParquetType::PrimitiveType(ParquetPrimitiveType {
                        field_info: FieldInfo {
                            name: "v".into(),
                            repetition: Repetition::Required,
                            id: None,
                        },
                        logical_type: None,
                        converted_type: None,
                        physical_type: ParquetPhysicalType::Int32,
                    }),
                ],
            }],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 3, 4, 6].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(None, false, 6),
                    Nested::primitive(None, false, 6),
                ],
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 3, 4, 6].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(None, false, 6),
                    Nested::primitive(None, false, 6),
                ],
            ]
        );
    }

    #[test]
    fn test_list_num_nulls() {
        // [[1, null], null, [], [2]], where the null list hides the value 9.
        let values = Int32Array::from([Some(1), None, Some(9), Some(2)]).boxed();
        let array = ListArray::new(
            ArrowDataType::List(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Int32,
                true,
            ))),
            vec![0i32, 2, 3, 3, 4].try_into().unwrap(),
            values,
            Some(Bitmap::from([true, false, true, true])),
        );
        let type_ = to_parquet_type(&Field::new("a".into(), array.dtype().clone(), true)).unwrap();

        for version in [Version::V1, Version::V2] {
            let options = WriteOptions {
                statistics: StatisticsOptions::full(),
                version,
                compression: CompressionOptions::Uncompressed,
                data_page_size: None,
            };
            let mut columns = array_to_columns(
                array.clone().boxed(),
                type_.clone(),
                options,
                &[Encoding::Plain],
            )
            .unwrap();
            let Some(Ok(Page::Data(page))) = columns[0].next() else {
                panic!("expected a data page");
            };

            // The values 1 and 2 are the only non-null values of the 5 values.
            assert_eq!(page.num_values(), 5);
            assert_eq!(page.num_nulls, Some(3));
            if version == Version::V2 {
                assert_eq!(page.null_count(), Some(3));
            }
        }
    }

    fn write_list_pages(
        array: &dyn Array,
        data_page_size: usize,
        mut check_page: impl FnMut(&crate::parquet::page::DataPage, usize),
    ) {
        let type_ = to_parquet_type(&Field::new("a".into(), array.dtype().clone(), true)).unwrap();
        for version in [Version::V1, Version::V2] {
            let options = WriteOptions {
                statistics: StatisticsOptions::empty(),
                version,
                compression: CompressionOptions::Uncompressed,
                data_page_size: Some(data_page_size),
            };
            let mut columns =
                array_to_columns(array.to_boxed(), type_.clone(), options, &[Encoding::Plain])
                    .unwrap();
            let mut rows = 0;
            for page in columns.remove(0) {
                let Page::Data(page) = page.unwrap() else {
                    panic!("expected a data page");
                };
                assert!(page.buffer().len() <= data_page_size);
                check_page(&page, rows);
                rows += page.num_rows().unwrap();
            }
            assert_eq!(rows, array.len());
        }
    }

    #[test]
    fn test_skewed_u8_lists() {
        // The first and last rows are sliced off, leaving four adjacent rows of 144
        // UInt8 elements each, followed by empty rows. Each encoded row fits, but two
        // do not fit because Parquet writes each value as a four-byte INT32.
        let offsets = [0, 1, 145, 289, 433, 577]
            .into_iter()
            .chain(std::iter::repeat_n(577, 2_000))
            .chain([578])
            .collect::<Vec<i64>>();
        let array = ListArray::new(
            ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                ArrowDataType::UInt8,
                false,
            ))),
            offsets.try_into().unwrap(),
            UInt8Array::from_slice([7; 578]).boxed(),
            None,
        )
        .sliced(1, 2_004);

        write_list_pages(&array, 1024, |page, start| {
            let rows = page.num_rows().unwrap();
            let nonempty = (start + rows).min(4) - start.min(4);
            assert!(nonempty <= 1);
            assert_eq!(page.num_values(), nonempty * 144 + rows - nonempty);
        });
    }

    #[test]
    fn test_empty_and_null_lists() {
        const ROWS: usize = 1024;
        let empty_leaves = [
            BinaryViewArray::from_slice_values::<&[u8], _>([]).boxed(),
            BinaryArray::<i64>::from_slice::<&[u8], _>([]).boxed(),
        ];
        for leaf in empty_leaves {
            let array = ListArray::new(
                ArrowDataType::LargeList(Box::new(Field::new(
                    "item".into(),
                    leaf.dtype().clone(),
                    false,
                ))),
                vec![0i64; ROWS + 1].try_into().unwrap(),
                leaf,
                Some((0..ROWS).map(|i| i % 2 == 0).collect()),
            );
            write_list_pages(&array, 64, |page, _| {
                assert_eq!(page.num_values(), page.num_rows().unwrap());
                assert_eq!(page.num_nulls, Some(page.num_values()));
            });
        }
    }

    #[test]
    fn test_nested_empty_and_null_lists() {
        const ROWS: usize = 64;
        const CHILDREN: usize = 16;
        let leaves = BinaryViewArray::from_slice_values::<&[u8], _>([]).boxed();
        let children = ListArray::new(
            ArrowDataType::List(Box::new(Field::new(
                "item".into(),
                leaves.dtype().clone(),
                false,
            ))),
            vec![0i32; ROWS * CHILDREN + 1].try_into().unwrap(),
            leaves,
            Some((0..ROWS * CHILDREN).map(|i| i % 2 == 0).collect()),
        );
        let array = ListArray::new(
            ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                children.dtype().clone(),
                true,
            ))),
            (0..=ROWS as i64)
                .map(|i| i * CHILDREN as i64)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            children.boxed(),
            Some((0..ROWS).map(|i| i % 2 == 0).collect()),
        );
        write_list_pages(&array, 64, |page, start| {
            let rows = page.num_rows().unwrap();
            let nonnull = (start..start + rows).filter(|i| i % 2 == 0).count();
            assert_eq!(page.num_values(), nonnull * CHILDREN + rows - nonnull);
            assert!(page.num_values() * 2 <= 64);
        });
    }

    /// Pages are cut by their size in bytes, not by the average value size, so a few large
    /// values among many small ones do not end up in a single oversized page.
    #[test]
    fn test_skewed_value_sizes() {
        const PAGE_SIZE: usize = 1024;
        // Room for the levels on top of the values and their 4-byte length prefixes.
        const PAGE_OVERHEAD: usize = 64;

        let small = vec![0u8; 1];
        let large = vec![0u8; 300];
        let values: Vec<&[u8]> = std::iter::repeat_n(small.as_slice(), 1000)
            .chain(std::iter::repeat_n(large.as_slice(), 4))
            .chain(std::iter::repeat_n(small.as_slice(), 1000))
            .collect();
        let num_rows = values.len();

        let view = BinaryViewArray::from_slice_values(&values).boxed();
        let large_binary = BinaryArray::<i64>::from_slice(&values).boxed();
        let strings: Vec<String> = values
            .iter()
            .map(|v| String::from_utf8(v.to_vec()).unwrap())
            .collect();
        let large_utf8 = Utf8Array::<i64>::from_slice(&strings).boxed();
        let list = ListArray::<i64>::new(
            ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                ArrowDataType::BinaryView,
                false,
            ))),
            (0..=num_rows as i64)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            view.clone(),
            None,
        )
        .boxed();

        let options = WriteOptions {
            statistics: StatisticsOptions::empty(),
            version: Version::V1,
            compression: CompressionOptions::Uncompressed,
            data_page_size: Some(PAGE_SIZE),
        };

        for array in [view, large_binary, large_utf8, list] {
            let type_ =
                to_parquet_type(&Field::new("a".into(), array.dtype().clone(), false)).unwrap();
            let mut columns =
                array_to_columns(array.clone(), type_, options, &[Encoding::Plain]).unwrap();

            let mut rows_written = 0;
            for page in columns.remove(0) {
                let Page::Data(page) = page.unwrap() else {
                    panic!("expected a data page");
                };
                rows_written += page.num_values();
                if page.num_values() > 1 {
                    assert!(
                        page.buffer().len() <= PAGE_SIZE + PAGE_OVERHEAD,
                        "{:?}: page of {} values is {} bytes",
                        array.dtype(),
                        page.num_values(),
                        page.buffer().len(),
                    );
                }
            }
            assert_eq!(rows_written, num_rows);
        }
    }
}
