import polars as pl

df = pl.DataFrame({"IP": ["1.1.1.1", "2.2.2.2"]})

isp_names = {"1.1.1.1": "ABC", "2.2.2.2": "XYZ"}

df.with_column(pl.col("IP").apply(isp_names.get))


df = pl.DataFrame({"IP": ["1.1.1.1", "2.2.2.2"], "ISP": ["N/A", "N/A"]})
for i, row in enumerate(df.rows()):
    df[i, "ISP"] = isp_names[row[0]]
