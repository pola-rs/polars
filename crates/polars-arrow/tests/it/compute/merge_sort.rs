use std::iter::once;

use polars_arrow::array::*;
use polars_arrow::compute::merge_sort::*;
use polars_arrow::compute::sort::sort;
use polars_arrow::error::Result;

#[test]
fn merge_u32() -> Result<()> {
    let a0: &dyn Array = &Int32Array::from_slice([0, 1, 2, 3]);
    let a1: &dyn Array = &Int32Array::from_slice([2, 3, 4, 5]);

    let options = SortOptions::default();
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;

    // (0, 1, 2) corresponds to slice [1, 2] of a0
    // (1, 2, 2) corresponds to slice [4, 5] of a1
    // slices are already sorted => identity
    let result =
        merge_sort_slices(once(&(0, 1, 2)), once(&(1, 2, 2)), &comparator).collect::<Vec<_>>();

    assert_eq!(result, vec![(0, 1, 2), (1, 2, 2)]);

    // (0, 2, 2) corresponds to slice [2, 3] of a0
    // (1, 0, 3) corresponds to slice [2, 3, 4] of a1
    let result =
        merge_sort_slices(once(&(0, 2, 2)), once(&(1, 0, 3)), &comparator).collect::<Vec<_>>();

    //   2 (a0) , [2, 3] (a1) ,   3 (a0) ,   4 (a1)
    // (0, 2, 1), (1, 0, 2)   , (0, 3, 1), (1, 2, 1)
    assert_eq!(result, vec![(0, 2, 1), (1, 0, 2), (0, 3, 1), (1, 2, 1)]);
    Ok(())
}

#[test]
fn merge_null_first() -> Result<()> {
    let a0: &dyn Array = &Int32Array::from(&[None, Some(0)]);
    let a1: &dyn Array = &Int32Array::from(&[Some(2), Some(3)]);
    let options = SortOptions {
        descending: false,
        nulls_first: true,
    };
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;
    let result =
        merge_sort_slices(once(&(0, 0, 2)), once(&(1, 0, 2)), &comparator).collect::<Vec<_>>();
    assert_eq!(result, vec![(0, 0, 2), (1, 0, 2)]);

    let a0: &dyn Array = &Int32Array::from(&[Some(0), None]);
    let a1: &dyn Array = &Int32Array::from(&[Some(2), Some(3)]);
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;
    let result =
        merge_sort_slices(once(&(0, 0, 2)), once(&(1, 0, 2)), &comparator).collect::<Vec<_>>();
    assert_eq!(result, vec![(0, 0, 1), (1, 0, 2), (0, 1, 1)]);

    let a0: &dyn Array = &Int32Array::from(&[Some(0), None]);
    let a1: &dyn Array = &Int32Array::from(&[Some(3), Some(2)]);
    let options = SortOptions {
        descending: true,
        nulls_first: false,
    };
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;
    let result =
        merge_sort_slices(once(&(0, 0, 2)), once(&(1, 0, 2)), &comparator).collect::<Vec<_>>();
    assert_eq!(result, vec![(1, 0, 2), (0, 0, 2)]);

    let a0: &dyn Array = &Int32Array::from(&[None, Some(0)]);
    let a1: &dyn Array = &Int32Array::from(&[Some(3), Some(2)]);
    let options = SortOptions {
        descending: true,
        nulls_first: true,
    };
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;
    let result =
        merge_sort_slices(once(&(0, 0, 2)), once(&(1, 0, 2)), &comparator).collect::<Vec<_>>();
    assert_eq!(result, vec![(0, 0, 1), (1, 0, 2), (0, 1, 1)]);

    Ok(())
}

#[test]
fn merge_with_limit() -> Result<()> {
    let a0: &dyn Array = &Int32Array::from_slice([0, 2, 4, 6, 8]);
    let a1: &dyn Array = &Int32Array::from_slice([1, 3, 5, 7, 9]);

    let options = SortOptions::default();
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;

    let slices = merge_sort_slices(once(&(0, 0, 5)), once(&(1, 0, 5)), &comparator);
    // thus, they can be used to take from the arrays
    let array = take_arrays(&arrays, slices, Some(5));

    let expected = Int32Array::from_slice([0, 1, 2, 3, 4]);
    // values are right
    assert_eq!(expected, array.as_ref());
    Ok(())
}

#[test]
fn merge_slices_to_vec() -> Result<()> {
    let a0: &dyn Array = &Int32Array::from_slice([0, 2, 4, 6, 8]);
    let a1: &dyn Array = &Int32Array::from_slice([1, 3, 5, 7, 9]);

    let options = SortOptions::default();
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;

    let slices = merge_sort_slices(once(&(0, 0, 5)), once(&(1, 0, 5)), &comparator);
    let vec = slices.to_vec(Some(5));
    assert_eq!(vec, [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]);
    Ok(())
}

#[test]
fn merge_4_i32() -> Result<()> {
    let a0: &dyn Array = &Int32Array::from_slice([0, 1]);
    let a1: &dyn Array = &Int32Array::from_slice([2, 6]);
    let a2: &dyn Array = &Int32Array::from_slice([3, 5]);
    let a3: &dyn Array = &Int32Array::from_slice([4, 7]);

    let options = SortOptions::default();
    let arrays = vec![a0, a1, a2, a3];
    let pairs = vec![(arrays.as_ref(), &options)];
    let slices = slices(&pairs)?;

    // slices are right.
    assert_eq!(
        slices,
        vec![
            (0, 0, 2),
            (1, 0, 1),
            (2, 0, 1),
            (3, 0, 1), // 4
            (2, 1, 1), // 5
            (1, 1, 1), // 6
            (3, 1, 1), // 7
        ]
    );

    // thus, they can be used to take from the arrays
    let array = take_arrays(&arrays, slices, None);

    let expected = Int32Array::from_slice([0, 1, 2, 3, 4, 5, 6, 7]);

    // values are right
    assert_eq!(expected, array.as_ref());
    Ok(())
}

#[test]
fn merge_binary() -> Result<()> {
    let a0: &dyn Array = &BinaryArray::<i32>::from_slice([b"a", b"c", b"d", b"e"]);
    let a1: &dyn Array = &BinaryArray::<i32>::from_slice([b"b", b"y", b"z", b"z"]);

    let options = SortOptions::default();
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;

    // (0, 0, 4) corresponds to slice ["a", "c", "d", "e"] of a0
    // (1, 0, 4) corresponds to slice ["b", "y", "z", "z"] of a1

    let result =
        merge_sort_slices(once(&(0, 0, 4)), once(&(1, 0, 4)), &comparator).collect::<Vec<_>>();

    // "a" (a0) , "b" (a1) ,  ["c", "d", "e"] (a0), ["y", "z", "z"] (a1)
    // (0, 0, 1), (1, 0, 1),      (0, 1, 3)       ,      (1, 1, 3)
    assert_eq!(result, vec![(0, 0, 1), (1, 0, 1), (0, 1, 3), (1, 1, 3)]);

    // (0, 1, 2) corresponds to slice ["c", "d"] of a0
    // (1, 0, 3) corresponds to slice ["b", "y", "z"] of a1
    let result =
        merge_sort_slices(once(&(0, 1, 2)), once(&(1, 0, 3)), &comparator).collect::<Vec<_>>();

    // "b" (a1) , ["c", "d"] (a0) , ["y", "z"]
    // (1, 0, 1), (0, 1, 2)       , (1, 1, 2)
    assert_eq!(result, vec![(1, 0, 1), (0, 1, 2), (1, 1, 2)]);
    Ok(())
}

#[test]
fn merge_string() -> Result<()> {
    let a0: &dyn Array = &Utf8Array::<i32>::from_slice(["a", "c", "d", "e"]);
    let a1: &dyn Array = &Utf8Array::<i32>::from_slice(["b", "y", "z", "z"]);

    let options = SortOptions::default();
    let arrays = vec![a0, a1];
    let pairs = vec![(arrays.as_ref(), &options)];
    let comparator = build_comparator(&pairs)?;

    // (0, 0, 4) corresponds to slice ["a", "c", "d", "e"] of a0
    // (1, 0, 4) corresponds to slice ["b", "y", "z", "z"] of a1

    let result =
        merge_sort_slices(once(&(0, 0, 4)), once(&(1, 0, 4)), &comparator).collect::<Vec<_>>();

    // "a" (a0) , "b" (a1) ,  ["c", "d", "e"] (a0), ["y", "z", "z"] (a1)
    // (0, 0, 1), (1, 0, 1),      (0, 1, 3)       ,      (1, 1, 3)
    assert_eq!(result, vec![(0, 0, 1), (1, 0, 1), (0, 1, 3), (1, 1, 3)]);

    // (0, 1, 2) corresponds to slice ["c", "d"] of a0
    // (1, 0, 3) corresponds to slice ["b", "y", "z"] of a1
    let result =
        merge_sort_slices(once(&(0, 1, 2)), once(&(1, 0, 3)), &comparator).collect::<Vec<_>>();

    // "b" (a1) , ["c", "d"] (a0) , ["y", "z"]
    // (1, 0, 1), (0, 1, 2)       , (1, 1, 2)
    assert_eq!(result, vec![(1, 0, 1), (0, 1, 2), (1, 1, 2)]);
    Ok(())
}

#[test]
fn merge_sort_many() -> Result<()> {
    // column 1
    let a00: &dyn Array = &Int32Array::from_slice([0, 1, 2, 3]);
    let a01: &dyn Array = &Int32Array::from_slice([2, 3, 4]);
    // column 2
    let a10: &dyn Array = &Utf8Array::<i32>::from_slice(["a", "c", "d", "e"]);
    let a11: &dyn Array = &Utf8Array::<i32>::from_slice(["b", "y", "z"]);
    // column 3
    // arrays to be sorted via the columns above
    let array0: &dyn Array = &Int32Array::from_slice([0, 1, 2, 3]);
    let array1: &dyn Array = &Int32Array::from_slice([4, 5, 6]);

    let expected = Int32Array::from_slice([
        0, // 0 (a00) < 2 (a01)
        1, // 1 (a00) < 2 (a01)
        4, // 2 (a00) == 2 (a01), "d" (a10) > "b" (a11)
        2, // 2 (a00) < 3 (a01)
        3, // 3 (a00) == 3 (a01), "e" (a10) < "y" (a11)
        5, // arrays0 has finished
        6, // arrays0 has finished
    ]);

    // merge-sort according to column 1 and then column 2
    let options = SortOptions::default();
    let arrays0 = vec![a00, a01];
    let arrays1 = vec![a10, a11];
    let pairs = vec![(arrays0.as_ref(), &options), (arrays1.as_ref(), &options)];
    let slices = slices(&pairs)?;

    let array = take_arrays(&[array0, array1], slices, None);

    assert_eq!(expected, array.as_ref());
    Ok(())
}

#[test]
fn test_sort() -> Result<()> {
    let data0 = vec![4, 1, 2, 10, 3, 3];
    let data1 = vec![5, 1, 0, 6, 7];

    let mut expected_data = [data0.clone(), data1.clone()].concat();
    expected_data.sort_unstable();
    let expected = Int32Array::from_slice(&expected_data);

    let a0: &dyn Array = &Int32Array::from_slice(&data0);
    let a1: &dyn Array = &Int32Array::from_slice(&data1);

    let options = SortOptions::default();

    // sort individually, potentially in parallel.
    let a0 = sort(a0, &options, None)?;
    let a1 = sort(a1, &options, None)?;

    // merge then. If multiple arrays, this can be applied in parallel.
    let result = merge_sort(a0.as_ref(), a1.as_ref(), &options, None)?;

    assert_eq!(expected, result.as_ref());
    Ok(())
}
