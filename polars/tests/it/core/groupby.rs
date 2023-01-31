use polars_core::series::IsSorted;

use super::*;

#[test]
fn test_sorted_groupby() -> PolarsResult<()> {
    // nulls last
    let mut s = Series::new("a", &[Some(1), Some(1), Some(1), Some(6), Some(6), None]);
    s.set_sorted_flag(IsSorted::Ascending);
    for mt in [true, false] {
        let out = s.group_tuples(mt, false)?;
        assert_eq!(out.unwrap_slice(), &[[0, 3], [3, 2], [5, 1]]);
    }

    // nulls first
    let mut s = Series::new(
        "a",
        &[None, None, Some(1), Some(1), Some(1), Some(6), Some(6)],
    );
    s.set_sorted_flag(IsSorted::Ascending);
    for mt in [true, false] {
        let out = s.group_tuples(mt, false)?;
        assert_eq!(out.unwrap_slice(), &[[0, 2], [2, 3], [5, 2]]);
    }

    // nulls last
    let mut s = Series::new("a", &[Some(1), Some(1), Some(1), Some(6), Some(6), None]);
    s.set_sorted_flag(IsSorted::Ascending);
    for mt in [true, false] {
        let out = s.group_tuples(mt, false)?;
        assert_eq!(out.unwrap_slice(), &[[0, 3], [3, 2], [5, 1]]);
    }

    // nulls first reverse sorted
    let mut s = Series::new(
        "a",
        &[
            None,
            None,
            Some(3),
            Some(3),
            Some(1),
            Some(1),
            Some(1),
            Some(-1),
        ],
    );
    s.set_sorted_flag(IsSorted::Descending);
    for mt in [false, true] {
        let out = s.group_tuples(mt, false)?;
        assert_eq!(out.unwrap_slice(), &[[0, 2], [2, 2], [4, 3], [7, 1]]);
    }

    // nulls last reverse sorted
    let mut s = Series::new(
        "a",
        &[
            Some(15),
            Some(15),
            Some(15),
            Some(15),
            Some(14),
            Some(13),
            Some(11),
            Some(11),
            Some(3),
            Some(3),
            Some(1),
            Some(1),
            Some(1),
            Some(-1),
            None,
            None,
            None,
        ],
    );
    s.set_sorted_flag(IsSorted::Descending);
    for mt in [false, true] {
        let out = s.group_tuples(mt, false)?;
        assert_eq!(
            out.unwrap_slice(),
            &[
                [0, 4],
                [4, 1],
                [5, 1],
                [6, 2],
                [8, 2],
                [10, 3],
                [13, 1],
                [14, 3]
            ]
        );
    }

    Ok(())
}
