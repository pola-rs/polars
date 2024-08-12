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

    const MIN_VALUES: usize = 125;
    const MAX_VALUES: usize = 135;

    const MIN: i64 = -512;
    const MAX: i64 = 512;

    const NUM_ITERATIONS: usize = 1_000_000;

    let mut values = Vec::with_capacity(MAX_VALUES);
    let mut rng = rand::thread_rng();

    let mut encoded = Vec::with_capacity(MAX_VALUES);
    let mut decoded = Vec::with_capacity(MAX_VALUES);
    let mut gatherer = SimpleGatherer;

    for _ in 0..NUM_ITERATIONS {
        values.clear();

        let num_values = rng.gen_range(MIN_VALUES..=MAX_VALUES);
        values.extend(std::iter::from_fn(|| Some(rng.gen_range(MIN..=MAX))).take(num_values));

        encoded.clear();
        decoded.clear();

        super::encode(values.iter().copied(), &mut encoded);
        let (mut decoder, rem) = super::Decoder::try_new(&encoded)?;

        assert!(rem.is_empty());

        let mut num_remaining = num_values;
        while num_remaining > 0 {
            let n = rng.gen_range(0usize..=num_remaining);
            decoder.gather_n_into(&mut decoded, n, &mut gatherer)?;
            num_remaining -= n;
        }

        assert_eq!(values, decoded);
    }

    Ok(())
}

#[test]
fn breakage() -> Result<(), Box<dyn std::error::Error>> {
    let values: &[i64] = &[-476];

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

    let mut gatherer = SimpleGatherer;
    let mut encoded = Vec::new();
    let mut decoded = Vec::new();
    let gathers = vec![0, 1, 0];

    super::encode(values.iter().copied(), &mut encoded);
    let (mut decoder, rem) = super::Decoder::try_new(&encoded)?;

    assert!(rem.is_empty());

    for g in gathers {
        decoder.gather_n_into(&mut decoded, g, &mut gatherer)?;
    }

    assert_eq!(values, decoded);

    Ok(())
}
