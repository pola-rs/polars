//! API to read and use bloom filters
mod hash;
mod read;
mod split_block;

pub use hash::{hash_byte, hash_native};
pub use read::read;
pub use split_block::{insert, is_in_set};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let mut bitset = vec![0; 32];

        // insert
        for a in 0..10i64 {
            let hash = hash_native(a);
            insert(&mut bitset, hash);
        }

        // bloom filter produced by parquet-mr/spark for a column of i64 (0..=10)
        /*
        import pyspark.sql  // 3.2.1
        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        spark.conf.set("parquet.bloom.filter.enabled", True)
        spark.conf.set("parquet.bloom.filter.expected.ndv", 10)
        spark.conf.set("parquet.bloom.filter.max.bytes", 32)

        data = [(i % 10,) for i in range(100)]
        df = spark.createDataFrame(data, ["id"]).repartition(1)

        df.write.parquet("bla.parquet", mode = "overwrite")
        */
        let expected: &[u8] = &[
            24, 130, 24, 8, 134, 8, 68, 6, 2, 101, 128, 10, 64, 2, 38, 78, 114, 1, 64, 38, 1, 192,
            194, 152, 64, 70, 0, 36, 56, 121, 64, 0,
        ];
        assert_eq!(bitset, expected);

        // check
        for a in 0..11i64 {
            let hash = hash_native(a);

            let valid = is_in_set(&bitset, hash);

            assert_eq!(a < 10, valid);
        }
    }

    #[test]
    fn binary() {
        let mut bitset = vec![0; 32];

        // insert
        for a in 0..10i64 {
            let value = format!("a{}", a);
            let hash = hash_byte(value);
            insert(&mut bitset, hash);
        }

        // bloom filter produced by parquet-mr/spark for a column of i64 f"a{i}" for i in 0..10
        let expected: &[u8] = &[
            200, 1, 80, 20, 64, 68, 8, 109, 6, 37, 4, 67, 144, 80, 96, 32, 8, 132, 43, 33, 0, 5,
            99, 65, 2, 0, 224, 44, 64, 78, 96, 4,
        ];
        assert_eq!(bitset, expected);
    }
}
