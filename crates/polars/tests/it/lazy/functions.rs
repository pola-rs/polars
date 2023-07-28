// used only if feature="format_str"
#[allow(unused_imports)]
use super::*;

#[test]
#[cfg(feature = "format_str")]
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

    assert!(out.frame_equal_missing(&expected));
}
