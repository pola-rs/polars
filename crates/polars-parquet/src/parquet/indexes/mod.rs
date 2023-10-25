mod index;
mod intervals;

pub use crate::parquet_bridge::BoundaryOrder;
pub use crate::thrift_format::PageLocation;

pub use self::index::{BooleanIndex, ByteIndex, FixedLenByteIndex, Index, NativeIndex, PageIndex};
pub use intervals::{compute_rows, select_pages, FilteredPage, Interval};

#[cfg(test)]
mod tests {
    use super::*;

    use crate::schema::types::{PhysicalType, PrimitiveType};

    #[test]
    fn test_basic() {
        let locations = &[PageLocation {
            offset: 100,
            compressed_page_size: 10,
            first_row_index: 0,
        }];
        let num_rows = 10;

        let row_intervals = compute_rows(&[true; 1], locations, num_rows).unwrap();
        assert_eq!(row_intervals, vec![Interval::new(0, 10)])
    }

    #[test]
    fn test_multiple() {
        // two pages
        let index = ByteIndex {
            primitive_type: PrimitiveType::from_physical("c1".to_string(), PhysicalType::ByteArray),
            indexes: vec![
                PageIndex {
                    min: Some(vec![0]),
                    max: Some(vec![8, 9]),
                    null_count: Some(0),
                },
                PageIndex {
                    min: Some(vec![20]),
                    max: Some(vec![98, 99]),
                    null_count: Some(0),
                },
            ],
            boundary_order: Default::default(),
        };
        let locations = &[
            PageLocation {
                offset: 100,
                compressed_page_size: 10,
                first_row_index: 0,
            },
            PageLocation {
                offset: 110,
                compressed_page_size: 20,
                first_row_index: 5,
            },
        ];
        let num_rows = 10;

        // filter of the form `x > "a"`
        let selector = |page: &PageIndex<Vec<u8>>| {
            page.max
                .as_ref()
                .map(|x| x.as_slice()[0] > 97)
                .unwrap_or(false) // no max is present => all nulls => not selected
        };
        let selected = index.indexes.iter().map(selector).collect::<Vec<_>>();

        let rows = compute_rows(&selected, locations, num_rows).unwrap();
        assert_eq!(rows, vec![Interval::new(5, 5)]);

        let pages = select_pages(&rows, locations, num_rows).unwrap();

        assert_eq!(
            pages,
            vec![
                FilteredPage {
                    start: 100,
                    length: 10,
                    selected_rows: vec![],
                    num_rows: 5
                },
                FilteredPage {
                    start: 110,
                    length: 20,
                    selected_rows: vec![Interval::new(0, 5)],
                    num_rows: 5
                }
            ]
        );
    }

    #[test]
    fn test_other_column() {
        let locations = &[
            PageLocation {
                offset: 100,
                compressed_page_size: 20,
                first_row_index: 0,
            },
            PageLocation {
                offset: 120,
                compressed_page_size: 20,
                first_row_index: 10,
            },
        ];
        let num_rows = 100;

        let intervals = &[Interval::new(5, 5)];

        let pages = select_pages(intervals, locations, num_rows).unwrap();

        assert_eq!(
            pages,
            vec![
                FilteredPage {
                    start: 100,
                    length: 20,
                    selected_rows: vec![Interval::new(5, 5)],
                    num_rows: 10,
                },
                FilteredPage {
                    start: 120,
                    length: 20,
                    selected_rows: vec![],
                    num_rows: 90
                },
            ]
        );
    }

    #[test]
    fn test_other_interval_in_middle() {
        let locations = &[
            PageLocation {
                offset: 100,
                compressed_page_size: 20,
                first_row_index: 0,
            },
            PageLocation {
                offset: 120,
                compressed_page_size: 20,
                first_row_index: 10,
            },
            PageLocation {
                offset: 140,
                compressed_page_size: 20,
                first_row_index: 100,
            },
        ];
        let num_rows = 200;

        // interval partially intersects 2 pages (0 and 1)
        let intervals = &[Interval::new(5, 6)];

        let pages = select_pages(intervals, locations, num_rows).unwrap();

        assert_eq!(
            pages,
            vec![
                FilteredPage {
                    start: 100,
                    length: 20,
                    selected_rows: vec![Interval::new(5, 5)],
                    num_rows: 10,
                },
                FilteredPage {
                    start: 120,
                    length: 20,
                    selected_rows: vec![Interval::new(0, 1)],
                    num_rows: 90,
                },
                FilteredPage {
                    start: 140,
                    length: 20,
                    selected_rows: vec![],
                    num_rows: 100
                },
            ]
        );
    }

    #[test]
    fn test_other_column2() {
        let locations = &[
            PageLocation {
                offset: 100,
                compressed_page_size: 20,
                first_row_index: 0,
            },
            PageLocation {
                offset: 120,
                compressed_page_size: 20,
                first_row_index: 10,
            },
            PageLocation {
                offset: 140,
                compressed_page_size: 20,
                first_row_index: 100,
            },
        ];
        let num_rows = 200;

        // interval partially intersects 1 page (0)
        let intervals = &[Interval::new(0, 1)];

        let pages = select_pages(intervals, locations, num_rows).unwrap();

        assert_eq!(
            pages,
            vec![
                FilteredPage {
                    start: 100,
                    length: 20,
                    selected_rows: vec![Interval::new(0, 1)],
                    num_rows: 10,
                },
                FilteredPage {
                    start: 120,
                    length: 20,
                    selected_rows: vec![],
                    num_rows: 90
                },
                FilteredPage {
                    start: 140,
                    length: 20,
                    selected_rows: vec![],
                    num_rows: 100
                },
            ]
        );
    }
}
