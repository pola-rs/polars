mod nested;

use arrow::array::{Array, NullArray};
use arrow::datatypes::ArrowDataType;

pub(crate) use nested::NullDecoder;

use super::{BasicDecompressor, CompressedPagesIter};
use crate::parquet::page::Page;

/// Converts [`PagesIter`] to an [`ArrayIter`]
pub fn iter_to_arrays<I>(
    mut iter: BasicDecompressor<I>,
    data_type: ArrowDataType,
    num_rows: usize,
) -> Box<dyn Array>
where
    I: CompressedPagesIter,
{
    use streaming_decompression::FallibleStreamingIterator;

    let mut len = 0usize;

    while let Ok(Some(page)) = iter.next() {
        match page {
            Page::Dict(_) => continue,
            Page::Data(page) => {
                let rows = page.num_values();
                len = (len + rows).min(num_rows);
                if len == num_rows {
                    break;
                }
            },
        }
    }

    Box::new(NullArray::new(data_type, len))
}

#[cfg(test)]
mod tests {
    // use arrow::array::NullArray;
    // use arrow::datatypes::ArrowDataType;
    // use polars_error::*;
    //
    // use super::iter_to_arrays;
    // use crate::parquet::encoding::Encoding;
    // use crate::parquet::error::ParquetError;
    // use crate::parquet::metadata::Descriptor;
    // use crate::parquet::page::{CompressedPage, DataPage, DataPageHeader, DataPageHeaderV1, Page};
    // use crate::parquet::schema::types::{PhysicalType, PrimitiveType};
    // #[allow(unused_imports)]
    // use crate::parquet::{fallible_streaming_iterator, CowBuffer};

    // @TODO
    // #[test]
    // fn limit() {
    //     let new_page = |values: i32| {
    //         Page::Data(DataPage::new(
    //             DataPageHeader::V1(DataPageHeaderV1 {
    //                 num_values: values,
    //                 encoding: Encoding::Plain.into(),
    //                 definition_level_encoding: Encoding::Plain.into(),
    //                 repetition_level_encoding: Encoding::Plain.into(),
    //                 statistics: None,
    //             }),
    //             CowBuffer::Owned(vec![]),
    //             Descriptor {
    //                 primitive_type: PrimitiveType::from_physical(
    //                     "a".to_string(),
    //                     PhysicalType::Int32,
    //                 ),
    //                 max_def_level: 0,
    //                 max_rep_level: 0,
    //             },
    //             None,
    //         ))
    //     };
    //
    //     let p1 = new_page(100);
    //     let p2 = new_page(100);
    //     let pages = vec![Result::<_, ParquetError>::Ok(p1), Ok(p2)];
    //     let pages = pages.into_iter().map(|p| Ok(CompressedPage::from(p?))).collect();
    //     let arrays = iter_to_arrays(pages.into_iter(), ArrowDataType::Null, Some(10), 101);
    //
    //     let arrays = arrays.collect::<PolarsResult<Vec<_>>>().unwrap();
    //     let expected = std::iter::repeat(NullArray::new(ArrowDataType::Null, 10).boxed())
    //         .take(10)
    //         .chain(std::iter::once(
    //             NullArray::new(ArrowDataType::Null, 1).boxed(),
    //         ));
    //     assert_eq!(arrays, expected.collect::<Vec<_>>())
    // }
}
