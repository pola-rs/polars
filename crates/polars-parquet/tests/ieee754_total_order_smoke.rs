use std::fs::File;

use polars_parquet::parquet::metadata::ColumnOrder;
use polars_parquet::parquet::read::read_metadata;

/// A file using the `IEEE_754_TOTAL_ORDER` column order must decode instead of
/// failing with "ColumnOrder union has no variant set".
#[test]
fn reads_ieee754_total_order_column_orders() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/floating_orders_nan_count.parquet"
    );
    let mut file = File::open(path).unwrap();
    let metadata = read_metadata(&mut file).unwrap();

    let orders = metadata.column_orders.as_ref().unwrap();
    assert_eq!(orders.len(), 6);

    // Even columns are *_ieee754, odd are *_typedef.
    for (i, order) in orders.iter().enumerate() {
        if i % 2 == 0 {
            assert_eq!(*order, ColumnOrder::IEEE754TotalOrder);
        } else {
            assert!(matches!(order, ColumnOrder::TypeDefinedOrder(_)));
        }
    }
}
