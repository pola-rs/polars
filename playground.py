import polars as pl

dfs = pl.read_csv_from_zip("/Users/yusufyudhistira/Documents/GitHub/polars/docs/assets/data/test_csv.zip")

print(dfs)