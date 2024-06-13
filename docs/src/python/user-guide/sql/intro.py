# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:context]
ctx = pl.SQLContext()
# --8<-- [end:context]

# --8<-- [start:register_context]
df = pl.DataFrame({"a": [1, 2, 3]})
lf = pl.LazyFrame({"b": [4, 5, 6]})

# Register all dataframes in the global namespace: registers both "df" and "lf"
ctx = pl.SQLContext(register_globals=True)

# Register an explicit mapping of identifier name to frame
ctx = pl.SQLContext(frames={"table_one": df, "table_two": lf})

# Register frames using kwargs; dataframe df as "df" and lazyframe lf as "lf"
ctx = pl.SQLContext(df=df, lf=lf)
# --8<-- [end:register_context]

# --8<-- [start:register_pandas]
import pandas as pd

df_pandas = pd.DataFrame({"c": [7, 8, 9]})
ctx = pl.SQLContext(df_pandas=pl.from_pandas(df_pandas))
# --8<-- [end:register_pandas]

# --8<-- [start:execute]
# For local files use scan_csv instead
pokemon = pl.read_csv(
    "https://gist.githubusercontent.com/ritchie46/cac6b337ea52281aa23c049250a4ff03/raw/89a957ff3919d90e6ef2d34235e6bf22304f3366/pokemon.csv"
)
with pl.SQLContext(register_globals=True, eager=True) as ctx:
    df_small = ctx.execute("SELECT * from pokemon LIMIT 5")
    print(df_small)
# --8<-- [end:execute]

# --8<-- [start:prepare_multiple_sources]
with open("docs/data/products_categories.json", "w") as temp_file:
    json_data = """{"product_id": 1, "category": "Category 1"}
{"product_id": 2, "category": "Category 1"}
{"product_id": 3, "category": "Category 2"}
{"product_id": 4, "category": "Category 2"}
{"product_id": 5, "category": "Category 3"}"""

    temp_file.write(json_data)

with open("docs/data/products_masterdata.csv", "w") as temp_file:
    csv_data = """product_id,product_name
1,Product A
2,Product B
3,Product C
4,Product D
5,Product E"""

    temp_file.write(csv_data)

sales_data = pd.DataFrame(
    {
        "product_id": [1, 2, 3, 4, 5],
        "sales": [100, 200, 150, 250, 300],
    }
)
# --8<-- [end:prepare_multiple_sources]

# --8<-- [start:execute_multiple_sources]
# Input data:
# products_masterdata.csv with schema {'product_id': Int64, 'product_name': String}
# products_categories.json with schema {'product_id': Int64, 'category': String}
# sales_data is a Pandas DataFrame with schema {'product_id': Int64, 'sales': Int64}

with pl.SQLContext(
    products_masterdata=pl.scan_csv("docs/data/products_masterdata.csv"),
    products_categories=pl.scan_ndjson("docs/data/products_categories.json"),
    sales_data=pl.from_pandas(sales_data),
    eager=True,
) as ctx:
    query = """
    SELECT
        product_id,
        product_name,
        category,
        sales
    FROM
        products_masterdata
    LEFT JOIN products_categories USING (product_id)
    LEFT JOIN sales_data USING (product_id)
    """
    print(ctx.execute(query))
# --8<-- [end:execute_multiple_sources]

# --8<-- [start:clean_multiple_sources]
import os

os.remove("docs/data/products_categories.json")
os.remove("docs/data/products_masterdata.csv")
# --8<-- [end:clean_multiple_sources]
