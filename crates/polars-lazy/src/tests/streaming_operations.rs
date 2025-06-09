use super::*;

#[test]
fn test_shift_streaming() {
    let lf = scan_foods_parquet(true);

    for i in 0..27 {
        let out = lf
            .clone()
            .with_row_index("foo", None)
            .select([
                col("foo").shift(lit(i as u64)),
                //first().sort(Default::default()).shift(lit(1)),
                //first().sort(Default::default()).alias("origi"),
            ])
            //.shift(3)
            .with_new_streaming(true)
            .collect()
            .unwrap();
        assert_eq!(out.shape(), (27, 1));
        assert_eq!(out.column("foo").unwrap().null_count(), i);
    }
}
