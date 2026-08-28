# --8<-- [start:construction]
from uuid import UUID

import polars as pl

order_a = UUID("67e55044-10b1-426f-9247-bb680e5fe0c8")
order_b = UUID("1ec87c3c-44cb-4f12-8b48-a1f4e1d2d11e")

orders = pl.DataFrame(
    {
        "order_id": [order_a, order_b],
        "amount": [12.50, 31.00],
    }
)
print(orders)
print(orders.schema)

explicit = pl.Series("order_id", [order_a, order_b], dtype=pl.UUID)
# --8<-- [end:construction]

# --8<-- [start:parsing]
raw = pl.DataFrame({"order_id": [str(order_a), "not-a-uuid", None]})
parsed = raw.with_columns(pl.col("order_id").cast(pl.UUID, strict=False))
print(parsed)
# --8<-- [end:parsing]

# --8<-- [start:filtering]
selected = orders.filter(pl.col("order_id") == order_a)
print(selected)
# --8<-- [end:filtering]

# --8<-- [start:generation]
random_ids = pl.uuid4(3, eager=True)
time_ordered_ids = pl.uuid7(3, eager=True)
# --8<-- [end:generation]

# --8<-- [start:inspection]
print(time_ordered_ids.uuid.version())
print(time_ordered_ids.uuid.timestamp())
# --8<-- [end:inspection]
