import polars as pl

# df = pl.DataFrame(
#     {"y": [[], [], []]}, schema={"y": pl.List(pl.Int64)}
# )
# struct_df = df.select(pl.col("y").list.to_struct(fields = []).struct.unnest())
# print("Resulting empty struct df:", struct_df)
# empty_df = empty_df.select(pl.col("y").struct.unnest())
# print(empty_df)

# empty_df = df.select(
#     pl.col("y")
#     .list.to_struct(n_field_strategy="max_width")
#     .struct.unnest()
# )
# print(empty_df)
df = pl.DataFrame({"y": [[], [], []]}, schema={"y": pl.List(pl.Int64)})
empty_df = df.select(pl.col("y").list.to_struct(fields=[]).struct.unnest())
print("Resulting empty struct df:", empty_df)
