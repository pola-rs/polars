use super::*;

#[test]
#[cfg(feature = "unique_counts")]
fn test_list_broadcast() {
    // simply test if this runs
    df![
        "g" => [1, 1, 1],
        "a" => [1, 2, 3],
    ]
    .unwrap()
    .lazy()
    .groupby([col("g")])
    .agg([col("a").unique_counts() * count()])
    .collect()
    .unwrap();
}
