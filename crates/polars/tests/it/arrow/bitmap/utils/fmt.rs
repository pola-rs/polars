use arrow::bitmap::utils::fmt;

struct A<'a>(&'a [u8], usize, usize);

impl<'a> std::fmt::Debug for A<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt(self.0, self.1, self.2, f)
    }
}

#[test]
fn test_debug() -> std::fmt::Result {
    let test = |bytes, offset, len, bytes_str| {
        assert_eq!(
            format!("{:?}", A(bytes, offset, len)),
            format!("Bitmap {{ len: {len}, offset: {offset}, bytes: {bytes_str} }}")
        );
    };
    test(&[1], 0, 0, "[]");
    test(&[0b11000001], 0, 8, "[0b11000001]");
    test(&[0b11000001, 1], 0, 9, "[0b11000001, 0b_______1]");
    test(&[1], 0, 2, "[0b______01]");
    test(&[1], 1, 2, "[0b_____00_]");
    test(&[1], 2, 2, "[0b____00__]");
    test(&[1], 3, 2, "[0b___00___]");
    test(&[1], 4, 2, "[0b__00____]");
    test(&[1], 5, 2, "[0b_00_____]");
    test(&[1], 6, 2, "[0b00______]");
    test(&[0b11000001, 1], 1, 9, "[0b1100000_, 0b______01]");
    test(&[0b11000001, 1, 1, 1], 1, 9, "[0b1100000_, 0b______01]");
    test(
        &[0b11000001, 1, 1],
        2,
        16,
        "[0b110000__, 0b00000001, 0b______01]",
    );
    Ok(())
}
