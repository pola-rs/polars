#[ignore = "Fuzz test. Takes too long"]
#[test]
fn fuzz_test_delta_encoding() -> Result<(), Box<dyn std::error::Error>> {
    use rand::Rng;

    use super::DeltaGatherer;
    use crate::parquet::error::ParquetResult;

    struct SimpleGatherer;

    impl DeltaGatherer for SimpleGatherer {
        type Target = Vec<i64>;

        fn target_len(&self, target: &Self::Target) -> usize {
            target.len()
        }

        fn target_reserve(&self, target: &mut Self::Target, n: usize) {
            target.reserve(n);
        }

        fn gather_one(&mut self, target: &mut Self::Target, v: i64) -> ParquetResult<()> {
            target.push(v);
            Ok(())
        }
    }

    const MIN_VALUES: usize = 1;
    const MAX_VALUES: usize = 515;

    const MIN: i64 = i64::MIN;
    const MAX: i64 = i64::MAX;

    const NUM_ITERATIONS: usize = 1_000_000;

    let mut values = Vec::with_capacity(MAX_VALUES);
    let mut rng = rand::thread_rng();

    let mut encoded = Vec::with_capacity(MAX_VALUES);
    let mut decoded = Vec::with_capacity(MAX_VALUES);
    let mut gatherer = SimpleGatherer;

    for i in 0..NUM_ITERATIONS {
        values.clear();

        let num_values = rng.gen_range(MIN_VALUES..=MAX_VALUES);
        values.extend(std::iter::from_fn(|| Some(rng.gen_range(MIN..=MAX))).take(num_values));

        encoded.clear();
        decoded.clear();

        super::encode(
            values.iter().copied(),
            &mut encoded,
            1 << rng.gen_range(0..=2),
        );
        let (mut decoder, rem) = super::Decoder::try_new(&encoded)?;

        assert!(rem.is_empty());

        let mut num_remaining = num_values;
        while num_remaining > 0 {
            let n = rng.gen_range(1usize..=num_remaining);
            decoder.gather_n_into(&mut decoded, n, &mut gatherer)?;
            num_remaining -= n;
        }

        assert_eq!(values, decoded);

        if i % 1000 == 999 {
            eprintln!("[INFO]: {} iterations done.", i + 1);
        }
    }

    Ok(())
}
