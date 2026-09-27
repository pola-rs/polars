use std::sync::Arc;

use polars_arrow::array::PrimitiveArray;
use polars_core::prelude::*;
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;

use super::hot::HotKeyRows;
use super::keys::KeyRowKeys;
use super::layout::{KeyRowLayout, bytes_eq};
use super::map::KeyRowIndexMap;
use crate::groups::new_hash_grouper;
use crate::hash_keys::HashKeys;
use crate::hot_groups::new_hash_hot_grouper;
use crate::idx_table::new_idx_table;

fn keys(df: &DataFrame, null_is_valid: bool, random_state: &PlRandomState) -> KeyRowKeys {
    let layout = KeyRowLayout::new(df.columns().iter().map(|c| c.dtype()));
    KeyRowKeys::from_columns(
        df.columns(),
        Arc::new(layout.unwrap()),
        random_state,
        null_is_valid,
    )
}

#[test]
fn layout_places_wide_columns_first() {
    let layout = KeyRowLayout::new(&[DataType::Int32, DataType::String, DataType::Int64]).unwrap();
    let offsets: Vec<usize> = layout.cols.iter().map(|c| c.offset).collect();
    assert_eq!(offsets, [24, 0, 16]);
    assert_eq!(layout.null_offset, 28);
    assert_eq!(layout.stride_words, 4);
    assert_eq!(layout.view_words, [0]);
    assert_eq!(layout.plain_words, [0, 2, 3]);

    let two_i64 = KeyRowLayout::new(&[DataType::Int64, DataType::Int64]).unwrap();
    assert_eq!(two_i64.stride_words, 3);
    assert!(KeyRowLayout::new(&[DataType::Int64, DataType::Null]).is_none());
    assert!(KeyRowLayout::new(&[DataType::String, DataType::Binary, DataType::String]).is_none());
}

#[test]
fn equal_content_has_equal_rows_and_hashes() {
    let long = "a string that does not fit in a view";
    let a = df!(
        "s" => [Some("x"), Some(long), None, Some(long)],
        "f" => [Some(0.0), Some(f64::NAN), Some(1.0), None],
    )
    .unwrap();
    // Same content, but the long strings live in other buffers at other offsets.
    let b = df!(
        "s" => [Some(long), Some("x"), Some("padding to move the next string"), Some(long)],
        "f" => [None, Some(-0.0), Some(2.0), Some(-f64::NAN)],
    )
    .unwrap();
    let random_state = PlRandomState::default();
    let (ka, kb) = (keys(&a, true, &random_state), keys(&b, true, &random_state));
    let mut map = KeyRowIndexMap::<()>::new(ka.layout.clone());
    unsafe {
        let groups: Vec<IdxSize> = (0..4)
            .map(|i| map.get_or_insert_with(&ka, i, || ()).0)
            .collect();
        assert_eq!(groups, [0, 1, 2, 3]);
        let found: Vec<Option<IdxSize>> = (0..4).map(|i| map.get_index_of(&kb, i)).collect();
        assert_eq!(found, [Some(3), Some(0), None, Some(1)]);
        assert_eq!(ka.hashes.value(3), kb.hashes.value(0));
        assert_eq!(ka.hashes.value(1), kb.hashes.value(3));
    }
    let out = map.keys_frame(a.schema());
    assert!(out.equals_missing(&a));

    // The same keys as rows, as a hot grouper hands them over.
    let mut hot = HotKeyRows::new(ka.layout.clone());
    unsafe {
        for i in 0..4 {
            hot.push(&ka, i);
        }
        assert!((0..4).all(|i| hot.eq_key(i as IdxSize, &ka, i)));
        assert!(!hot.eq_key(0, &ka, 1));
    }
    let rows = hot.keys();
    let mut map = KeyRowIndexMap::<()>::new(rows.layout.clone());
    unsafe {
        for i in 0..4 {
            map.get_or_insert_with(&rows, i, || ());
        }
        assert_eq!(map.get_or_insert_with(&rows, 3, || ()), (3, false));
        let found: Vec<Option<IdxSize>> = (0..4).map(|i| map.get_index_of(&kb, i)).collect();
        assert_eq!(found, [Some(3), Some(0), None, Some(1)]);
    }
    assert!(map.keys_frame(a.schema()).equals_missing(&a));
}

#[test]
fn colliding_hashes_are_told_apart() {
    let df = df!(
        "a" => [Some(1i64), Some(2), None, Some(1), Some(2), None],
        "s" => ["x", "y", "x", "x", "y", "x"],
    )
    .unwrap();
    let mut ks = keys(&df, true, &PlRandomState::default());
    ks.hashes = PrimitiveArray::from_vec(vec![42; df.height()]);
    let idxs: Vec<IdxSize> = (0..6).collect();
    let mut map = KeyRowIndexMap::<()>::new(ks.layout.clone());
    let (mut groups, mut found) = (Vec::new(), Vec::new());
    unsafe {
        map.get_or_insert_batch(&ks, &idxs, |_| (), &mut groups);
        map.get_indices_of(&ks, &[4, 5, 3], &mut found);
    }
    assert_eq!(groups, [0, 1, 2, 0, 1, 2]);
    assert_eq!(found, [1, 2, 0]);

    let mut hot = HotKeyRows::new(ks.layout.clone());
    unsafe {
        for i in 0..3 {
            hot.push(&ks, i);
        }
    }
    let rows = hot.keys();
    let mut map = KeyRowIndexMap::<()>::new(rows.layout.clone());
    let (mut groups, mut found) = (Vec::new(), Vec::new());
    unsafe {
        map.get_or_insert_batch(&rows, &[2, 1, 0, 1], |_| (), &mut groups);
        map.get_indices_of(&ks, &idxs, &mut found);
    }
    assert_eq!(groups, [0, 1, 2, 1]);
    assert_eq!(found, [2, 1, 0, 2, 1, 0]);
}

#[test]
fn colliding_new_keys_get_indices_in_order() {
    let df = df!("a" => [1i64, 2, 3, 2], "b" => [1i64, 2, 3, 2]).unwrap();
    let mut ks = keys(&df, true, &PlRandomState::default());
    ks.hashes = PrimitiveArray::from_vec(vec![42, 42, 7, 42]);
    let mut map = KeyRowIndexMap::<IdxSize>::new(ks.layout.clone());
    let mut groups = Vec::new();
    unsafe {
        map.get_or_insert_batch(&ks, &[0], |r| r as IdxSize, &mut groups);
        map.get_or_insert_batch(&ks, &[1, 2, 3], |r| r as IdxSize, &mut groups);
    }
    assert_eq!(groups, [0, 1, 2, 1]);
    let values: Vec<IdxSize> = (0..3).map(|g| *map.get_value(g).unwrap()).collect();
    assert_eq!(values, [0, 0, 1]);
    assert!(map.keys_frame(df.schema()).equals(&df.slice(0, 3)));
}

#[test]
#[should_panic]
fn columns_of_unequal_length_are_rejected() {
    let columns = [
        Column::new("a".into(), [1i64, 2]),
        Column::new("b".into(), [1i64]),
    ];
    let layout = KeyRowLayout::new(columns.iter().map(|c| c.dtype())).unwrap();
    KeyRowKeys::from_columns(&columns, Arc::new(layout), &PlRandomState::default(), true);
}

#[test]
fn empty_map_has_no_keys() {
    let schema = Schema::from_iter([
        Field::new("a".into(), DataType::Int64),
        Field::new("s".into(), DataType::String),
    ]);
    let layout = KeyRowLayout::new(schema.iter_values()).unwrap();
    let map = KeyRowIndexMap::<()>::new(Arc::new(layout));
    let out = map.keys_frame(&schema);
    assert_eq!(out.height(), 0);
    assert_eq!(**out.schema(), schema);
}

#[test]
fn bytes_eq_compares_every_byte() {
    let a: Vec<u8> = (0..41).collect();
    for n in 0..=40 {
        let x = &a[..n];
        assert!(bytes_eq(x, &a[..n]));
        assert!(n == 0 || !bytes_eq(x, &a[1..=n]));
        for i in 0..n {
            let mut y = x.to_vec();
            y[i] ^= 1;
            assert!(!bytes_eq(x, &y));
        }
    }
}

#[test]
fn nulls_are_keys_only_when_valid() {
    let df = df!("a" => [Some(1i64), None], "b" => [Some(true), Some(false)]).unwrap();
    assert!(
        keys(&df, true, &PlRandomState::default())
            .validity
            .is_none()
    );
    let invalid = keys(&df, false, &PlRandomState::default())
        .validity
        .unwrap();
    assert_eq!(invalid.iter().collect::<Vec<_>>(), [true, false]);
}

fn two_i64_schema() -> Arc<Schema> {
    Arc::new(Schema::from_iter([
        Field::new("a".into(), DataType::Int64),
        Field::new("b".into(), DataType::Int64),
    ]))
}

fn hash_keys(df: DataFrame) -> HashKeys {
    HashKeys::from_df(&df, PlRandomState::default(), true, false)
}

fn i32_i64_keys() -> HashKeys {
    hash_keys(df!("a" => [1i32, 2], "b" => [1i64, 2]).unwrap())
}

#[test]
#[should_panic(expected = "key schema")]
fn hot_grouper_rejects_keys_of_another_schema() {
    let mut grouper = new_hash_hot_grouper(two_i64_schema(), 16);
    let (mut hot, mut groups, mut cold) = (Vec::new(), Vec::new(), Vec::new());
    grouper.insert_keys(&i32_i64_keys(), &mut hot, &mut groups, &mut cold, false);
}

#[test]
#[should_panic(expected = "key schema")]
fn grouper_rejects_keys_of_another_schema() {
    let mut grouper = new_hash_grouper(two_i64_schema());
    unsafe { grouper.insert_keys_subset(&i32_i64_keys(), &[0, 1], None) };
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "key schema")]
fn partitioned_groupers_reject_keys_of_another_schema() {
    let groupers = vec![new_hash_grouper(two_i64_schema())];
    let partitioner = HashPartitioner::new(1, 0);
    let mut matches = Vec::new();
    unsafe {
        groupers[0].probe_partitioned_groupers(
            &groupers,
            &i32_i64_keys(),
            &partitioner,
            false,
            &mut matches,
        )
    };
}

#[test]
#[should_panic(expected = "key schema")]
fn grouper_rejects_output_schema_of_another_layout() {
    let mut grouper = new_hash_grouper(two_i64_schema());
    let keys = hash_keys(df!("a" => [1i64], "b" => [1i64]).unwrap());
    unsafe { grouper.insert_keys_subset(&keys, &[0], None) };
    let schema = Schema::from_iter([
        Field::new("a".into(), DataType::Int32),
        Field::new("b".into(), DataType::Int64),
    ]);
    grouper.get_keys_in_group_order(&schema);
}

#[test]
#[should_panic(expected = "key schema")]
fn idx_table_rejects_keys_of_another_schema() {
    let mut table = new_idx_table(two_i64_schema());
    table.insert_keys(&i32_i64_keys(), false);
}

#[test]
#[should_panic(expected = "key schema")]
fn idx_table_rejects_probe_keys_of_another_schema() {
    let mut table = new_idx_table(two_i64_schema());
    table.insert_keys(
        &hash_keys(df!("a" => [1i64], "b" => [1i64]).unwrap()),
        false,
    );
    let (mut table_match, mut probe_match) = (Vec::new(), Vec::new());
    table.probe(
        &i32_i64_keys(),
        &mut table_match,
        &mut probe_match,
        false,
        false,
        IdxSize::MAX,
    );
}
