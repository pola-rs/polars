use super::*;

#[test]
#[cfg(all(feature = "concat_str", feature = "strings"))]
fn test_format_str() {
    let a = df![
        "a" => [1, 2],
        "b" => ["a", "b"]
    ]
    .unwrap();

    let out = a
        .lazy()
        .select([format_str("({}, {}]", [col("a"), col("b")])
            .unwrap()
            .alias("formatted")])
        .collect()
        .unwrap();

    let expected = df![
        "formatted" => ["(1, a]", "(2, b]"]
    ]
    .unwrap();

    assert!(out.equals_missing(&expected));
}
