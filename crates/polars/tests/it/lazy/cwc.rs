use polars::prelude::*;

#[test]
#[ignore = "fuzz test: Takes to long"]
fn fuzz_cluster_with_columns() {
    const PRIMES: &[i32] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    use rand::Rng;

    macro_rules! to_str {
        ($col:expr) => {
            std::str::from_utf8(std::slice::from_ref(&$col)).unwrap()
        };
    }

    fn rnd_prime(rng: &'_ mut rand::rngs::ThreadRng) -> i32 {
        PRIMES[rng.gen_range(0..PRIMES.len())]
    }

    fn sample(rng: &'_ mut rand::rngs::ThreadRng, slice: &[u8]) -> u8 {
        assert!(!slice.is_empty());
        slice[rng.gen_range(0..slice.len())]
    }

    fn gen_expr(rng: &mut rand::rngs::ThreadRng, used_cols: &[u8]) -> Expr {
        let mut depth = 0;

        use rand::Rng;

        fn leaf(rng: &mut rand::rngs::ThreadRng, used_cols: &[u8]) -> Expr {
            if rng.gen() {
                lit(rnd_prime(rng))
            } else {
                col(to_str!(sample(rng, used_cols)))
            }
        }

        let mut e = leaf(rng, used_cols);

        loop {
            if depth >= 10 || rng.gen() {
                return e;
            } else {
                e = e * col(to_str!(sample(rng, used_cols)));
            }

            depth += 1;
        }
    }

    use std::ops::RangeInclusive;

    const NUM_ORIGINAL_COLS: RangeInclusive<usize> = 1..=6;
    const NUM_WITH_COLUMNS: RangeInclusive<usize> = 1..=64;
    const NUM_EXPRS: RangeInclusive<usize> = 1..=8;

    let mut rng = rand::thread_rng();
    let rng = &mut rng;

    let mut unused_cols: Vec<u8> = Vec::with_capacity(26);
    let mut used_cols: Vec<u8> = Vec::with_capacity(26);

    let mut series: Vec<Series> = Vec::with_capacity(*NUM_ORIGINAL_COLS.end());

    let mut used: Vec<u8> = Vec::with_capacity(26);

    let num_fuzzes = 100_000;
    for _ in 0..num_fuzzes {
        unused_cols.clear();
        used_cols.clear();
        unused_cols.extend(b'a'..=b'z');

        let num_with_columns = rng.gen_range(NUM_WITH_COLUMNS.clone());
        let num_columns = rng.gen_range(NUM_ORIGINAL_COLS.clone());

        for _ in 0..num_columns {
            let column = rng.gen_range(0..unused_cols.len());
            let column = unused_cols.swap_remove(column);

            series.push(Series::new(to_str!(column), vec![rnd_prime(rng)]));
            used_cols.push(column);
        }

        let mut lf = DataFrame::new(std::mem::take(&mut series)).unwrap().lazy();

        for _ in 0..num_with_columns {
            let num_exprs = rng.gen_range(0..8);
            let mut exprs = Vec::with_capacity(*NUM_EXPRS.end());
            used.clear();

            for _ in 0..num_exprs {
                let col = loop {
                    let col = if unused_cols.is_empty() || rng.gen() {
                        sample(rng, &used_cols)
                    } else {
                        sample(rng, &unused_cols)
                    };

                    if !used.contains(&col) {
                        break col;
                    }
                };

                used.push(col);

                exprs.push(gen_expr(rng, &used_cols).alias(to_str!(col)));
            }

            lf = lf.with_columns(exprs);

            for u in &used {
                if let Some(idx) = unused_cols.iter().position(|x| x == u) {
                    unused_cols.remove(idx);
                    used_cols.push(*u);
                }
            }
        }

        lf = lf.without_optimizations();
        let cwc = lf.clone().with_cluster_with_columns(true);

        assert_eq!(lf.collect().unwrap(), cwc.collect().unwrap());
    }
}
