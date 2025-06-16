#[cfg(test)]
mod test_unsigned_arithmetic_fix {
    use polars_core::prelude::*;
    use polars_plan::prelude::{col, lit};

    #[test]
    fn test_u32_subtraction_underflow_fix() -> PolarsResult<()> {
        // Test basic u32 subtraction that would underflow
        let df = df![
            "a" => [2u32, 5u32, 1u32],
            "b" => [3u32, 2u32, 4u32],
        ]?;

        let result = df
            .lazy()
            .with_columns([
                (col("a") - col("b")).alias("diff"),
            ])
            .collect()?;

        let diff_col = result.column("diff")?;
        
        // Should be promoted to i64 to handle negative results
        assert_eq!(diff_col.dtype(), &DataType::Int64);
        
        // Check values are mathematically correct: [2-3, 5-2, 1-4] = [-1, 3, -3]
        let expected = [-1i64, 3i64, -3i64];
        for (i, &exp) in expected.iter().enumerate() {
            if let AnyValue::Int64(actual) = diff_col.get(i)? {
                assert_eq!(actual, exp, "Row {}: expected {}, got {}", i, exp, actual);
            } else {
                panic!("Expected Int64 at row {}", i);
            }
        }

        Ok(())
    }

    #[test]
    fn test_other_unsigned_types() -> PolarsResult<()> {
        // Test u16 and u8 also get promoted correctly
        let df16 = df![
            "a" => [1u16, 10u16],
            "b" => [5u16, 3u16],
        ]?;

        let result16 = df16
            .lazy()
            .with_columns([(col("a") - col("b")).alias("diff")])
            .collect()?;

        assert_eq!(result16.column("diff")?.dtype(), &DataType::Int32);

        let df8 = df![
            "a" => [1u8, 10u8],
            "b" => [5u8, 3u8],
        ]?;

        let result8 = df8
            .lazy()
            .with_columns([(col("a") - col("b")).alias("diff")])
            .collect()?;

        assert_eq!(result8.column("diff")?.dtype(), &DataType::Int16);

        Ok(())
    }

    #[test] 
    fn test_count_minus_literal() -> PolarsResult<()> {
        // Test the specific case of count() - literal that caused the original issue
        let df = df![
            "group" => ["A", "A", "B", "B", "C"],
            "value" => [1, 2, 3, 4, 5],
        ]?;

        let result = df
            .lazy()
            .group_by([col("group")])
            .agg([
                (col("value").count() - lit(3u32)).alias("count_minus_3"),
            ])
            .collect()?;

        let count_minus_3 = result.column("count_minus_3")?;
        
        // Should be promoted to i64 when count (u32) is subtracted from literal
        assert_eq!(count_minus_3.dtype(), &DataType::Int64);

        // All values should be negative since groups have ≤2 items, so count - 3 ≤ -1
        for i in 0..result.height() {
            if let AnyValue::Int64(actual) = count_minus_3.get(i)? {
                assert!(actual <= -1, "Expected negative value, got {}", actual);
            } else {
                panic!("Expected Int64 at row {}", i);
            }
        }

        Ok(())
    }
}
