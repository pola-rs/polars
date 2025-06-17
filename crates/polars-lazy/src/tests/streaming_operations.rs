use super::*;

#[test]
fn test_shift_streaming() {
    let lf = scan_foods_parquet(true);

    for neg in [false, true] {
        for i in 3..8 {
            let offset = if neg { -i } else { i };

            let out = lf
                .clone()
                .with_row_index("foo", None)
                .select([col("foo").shift(lit(offset))])
                .with_new_streaming(true)
                .collect()
                .unwrap();
            assert_eq!(out.shape(), (27, 1));
            assert_eq!(out.column("foo").unwrap().null_count(), i as usize);
        }
    }
}
