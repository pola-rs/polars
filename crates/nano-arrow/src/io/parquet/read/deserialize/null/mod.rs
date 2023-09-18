mod nested;

use parquet2::page::Page;

use crate::{array::NullArray, datatypes::DataType};

use super::super::{ArrayIter, Pages};
pub(super) use nested::NestedIter;

/// Converts [`Pages`] to an [`ArrayIter`]
pub fn iter_to_arrays<'a, I>(
    mut iter: I,
    data_type: DataType,
    chunk_size: Option<usize>,
    num_rows: usize,
) -> ArrayIter<'a>
where
    I: 'a + Pages,
{
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
            }
        }
    }

    if len == 0 {
        return Box::new(std::iter::empty());
    }

    let chunk_size = chunk_size.unwrap_or(len);

    let complete_chunks = len / chunk_size;

    let remainder = len - (complete_chunks * chunk_size);
    let i_data_type = data_type.clone();
    let complete = (0..complete_chunks)
        .map(move |_| Ok(NullArray::new(i_data_type.clone(), chunk_size).boxed()));
    if len % chunk_size == 0 {
        Box::new(complete)
    } else {
        let array = NullArray::new(data_type, remainder);
        Box::new(complete.chain(std::iter::once(Ok(array.boxed()))))
    }
}

#[cfg(test)]
mod tests {
    use parquet2::{
        encoding::Encoding,
        error::Error as ParquetError,
        metadata::Descriptor,
        page::{DataPage, DataPageHeader, DataPageHeaderV1, Page},
        schema::types::{PhysicalType, PrimitiveType},
    };

    use crate::{array::NullArray, datatypes::DataType, error::Error};

    use super::iter_to_arrays;

    #[test]
    fn limit() {
        let new_page = |values: i32| {
            Page::Data(DataPage::new(
                DataPageHeader::V1(DataPageHeaderV1 {
                    num_values: values,
                    encoding: Encoding::Plain.into(),
                    definition_level_encoding: Encoding::Plain.into(),
                    repetition_level_encoding: Encoding::Plain.into(),
                    statistics: None,
                }),
                vec![],
                Descriptor {
                    primitive_type: PrimitiveType::from_physical(
                        "a".to_string(),
                        PhysicalType::Int32,
                    ),
                    max_def_level: 0,
                    max_rep_level: 0,
                },
                None,
            ))
        };

        let p1 = new_page(100);
        let p2 = new_page(100);
        let pages = vec![Result::<_, ParquetError>::Ok(&p1), Ok(&p2)];
        let pages = fallible_streaming_iterator::convert(pages.into_iter());
        let arrays = iter_to_arrays(pages, DataType::Null, Some(10), 101);

        let arrays = arrays.collect::<Result<Vec<_>, Error>>().unwrap();
        let expected = std::iter::repeat(NullArray::new(DataType::Null, 10).boxed())
            .take(10)
            .chain(std::iter::once(NullArray::new(DataType::Null, 1).boxed()));
        assert_eq!(arrays, expected.collect::<Vec<_>>())
    }
}
