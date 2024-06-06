use polars::prelude::*;

#[ignore]
#[test]
fn fuzz_exprs() {
    const PRIMES: &[i32] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    use rand::Rng;

    let lf = DataFrame::new(vec![
        Series::new("A", vec![1, 2, 3, 4, 5]),
        Series::new("B", vec![Some(5), Some(4), None, Some(2), Some(1)]),
        Series::new("C", vec!["str", "", "a quite long string", "my", "string"]),
    ])
    .unwrap()
    .lazy();
    let empty = DataFrame::new(vec![
        Series::new("A", Vec::<bool>::new()),
        Series::new("B", Vec::<u32>::new()),
        Series::new("C", Vec::<&str>::new()),
    ])
    .unwrap()
    .lazy();

    fn rnd_prime(rng: &'_ mut rand::rngs::ThreadRng) -> i32 {
        PRIMES[rng.gen_range(0..PRIMES.len())]
    }

    fn gen_expr(rng: &mut rand::rngs::ThreadRng) -> Expr {
        let mut depth = 0;

        use rand::Rng;

        fn leaf(rng: &mut rand::rngs::ThreadRng) -> Expr {
            match rng.gen::<u32>() % 4 {
                0 => col("A"),
                1 => col("B"),
                2 => col("C"),
                _ => lit(rnd_prime(rng)),
            }
        }

        let mut e = leaf(rng);

        loop {
            if depth >= 10 || rng.gen::<u32>() % 4 == 0 {
                return e;
            } else {
                let rhs = leaf(rng);

                e = match rng.gen::<u32>() % 19 {
                    0 => e.eq(rhs),
                    1 => e.eq_missing(rhs),
                    2 => e.neq(rhs),
                    3 => e.neq_missing(rhs),
                    4 => e.lt(rhs),
                    5 => e.lt_eq(rhs),
                    6 => e.gt(rhs),
                    7 => e.gt_eq(rhs),
                    8 => e + rhs,
                    9 => e - rhs,
                    10 => e * rhs,
                    11 => e / rhs,
                    12 => Expr::BinaryExpr {
                        left: Arc::new(e),
                        right: Arc::new(rhs),
                        op: Operator::TrueDivide,
                    },
                    13 => e.floor_div(rhs),
                    14 => e % rhs,
                    15 => e.and(rhs),
                    16 => e.or(rhs),
                    17 => e.xor(rhs),
                    18 => e.logical_and(rhs),
                    19 => e.logical_or(rhs),
                    _ => unreachable!(),
                };
            }

            depth += 1;
        }
    }

    let mut rng = rand::thread_rng();
    let rng = &mut rng;

    let num_fuzzes = 100_000;
    for _ in 0..num_fuzzes {
        let exprs = vec![
            gen_expr(rng).alias("X"),
            gen_expr(rng).alias("Y"),
            gen_expr(rng).alias("Z"),
            gen_expr(rng).alias("W"),
            gen_expr(rng).alias("I"),
            gen_expr(rng).alias("J"),
        ];

        let wc = match rng.gen::<u32>() % 2 {
            0 => lf.clone(),
            _ => empty.clone(),
        };
        let wc = wc.with_columns(exprs);

        let unoptimized = wc.clone().without_optimizations();
        let optimized = wc;

        match (optimized.collect(), unoptimized.collect()) {
            (Ok(o), Ok(u)) => assert_eq!(o, u),
            (Err(_), Err(_)) => {},
            (_, _) => panic!("One failed!"),
        }
    }
}
