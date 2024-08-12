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
    const MAX_VALUES: usize = 1000;

    const MIN: i64 = -512;
    const MAX: i64 = 512;

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
        
        super::encode(values.iter().copied(), &mut encoded, 1 << rng.gen_range(0..=2));
        let (mut decoder, rem) = super::Decoder::try_new(&encoded)?;

        assert!(rem.is_empty());

        let mut num_remaining = num_values;
        while num_remaining > 0 {
            let n = rng.gen_range(0usize..=num_remaining);
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

#[test]
fn breakage() -> Result<(), Box<dyn std::error::Error>> {
    let values: &[i64] = &[
        447, 224, 201, -404, -187, -350, -301, -409, 425, 506, -287, -470, 480, -30, 373, -495,
        288, -337, -196, 188, 488, -53, 336, -163, 392, -255, 41, 465, 47, 91, -437, -259, -69,
        251, 237, -197, -508, -356, 119, 242, 63, -339, 450, -27, -176, 472, -449, 298, -303, -463,
        101, 267, 165, -195, 467, 301, 268, -199, -247, -285, 404, -227, -30, 242, 91, 219, 450,
        -402, -300, -473, -199, 491, -512, -425, 211, -88, -302, 316, 126, 207, 215, -322, -92,
        462, 280, 374, 21, -490, 159, 434, 372, 205, -211, 59, -213, -222, 193, -297, -449, -5,
        -241, -320, 218, 50, 177, -80, -468, -366, 286, -214, -353, -453, -390, 429, -484, -351,
        -195, 261, 508, -483, -396, 56, 209, 1, -335, 398, -317, 379, -217, -347,
    ];

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
    let gathers = vec![100, 28, 1, 1];

    super::encode(values.iter().copied(), &mut encoded, 2);
    let (mut decoder, rem) = super::Decoder::try_new(&encoded)?;

    assert!(rem.is_empty());

    for g in gathers {
        decoder.gather_n_into(&mut decoded, g, &mut gatherer)?;
    }

    assert_eq!(values, decoded);

    Ok(())
}
