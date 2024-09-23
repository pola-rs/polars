# --8<-- [start:dataframe]
import polars as pl
import polars.selectors as cs

path = "docs/assets/data/iris.csv"

df = (
    pl.scan_csv(path)
    .group_by("species")
    .agg(cs.starts_with("petal").mean().round(3))
    .collect()
)
print(df)
# --8<-- [end:dataframe]

# --8<-- [start:structure-header]
df.style.tab_header(title="Iris Data", subtitle="Mean measurement values per species")
# --8<-- [end:structure-header]

# --8<-- [start:structure-header-out]
print(
    df.style.tab_header(
        title="Iris Data", subtitle="Mean measurement values per species"
    ).as_raw_html()
)
# --8<-- [end:structure-header-out]


# --8<-- [start:structure-stub]
df.style.tab_stub(rowname_col="species")
# --8<-- [end:structure-stub]

# --8<-- [start:structure-stub-out]
print(df.style.tab_stub(rowname_col="species").as_raw_html())
# --8<-- [end:structure-stub-out]

# --8<-- [start:structure-spanner]
(
    df.style.tab_spanner("Petal", cs.starts_with("petal")).cols_label(
        petal_length="Length", petal_width="Width"
    )
)
# --8<-- [end:structure-spanner]

# --8<-- [start:structure-spanner-out]
print(
    df.style.tab_spanner("Petal", cs.starts_with("petal"))
    .cols_label(petal_length="Length", petal_width="Width")
    .as_raw_html()
)
# --8<-- [end:structure-spanner-out]

# --8<-- [start:format-number]
df.style.fmt_number("petal_width", decimals=1)
# --8<-- [end:format-number]


# --8<-- [start:format-number-out]
print(df.style.fmt_number("petal_width", decimals=1).as_raw_html())
# --8<-- [end:format-number-out]


# --8<-- [start:style-simple]
from great_tables import loc, style

df.style.tab_style(
    style.fill("yellow"),
    loc.body(
        rows=pl.col("petal_length") == pl.col("petal_length").max(),
    ),
)
# --8<-- [end:style-simple]

# --8<-- [start:style-simple-out]
from great_tables import loc, style

print(
    df.style.tab_style(
        style.fill("yellow"),
        loc.body(
            rows=pl.col("petal_length") == pl.col("petal_length").max(),
        ),
    ).as_raw_html()
)
# --8<-- [end:style-simple-out]


# --8<-- [start:style-bold-column]
from great_tables import loc, style

df.style.tab_style(
    style.text(weight="bold"),
    loc.body(columns="species"),
)
# --8<-- [end:style-bold-column]

# --8<-- [start:style-bold-column-out]
from great_tables import loc, style

print(
    df.style.tab_style(
        style.text(weight="bold"),
        loc.body(columns="species"),
    ).as_raw_html()
)
# --8<-- [end:style-bold-column-out]

# --8<-- [start:full-example]
from great_tables import loc, style

(
    df.style.tab_header(
        title="Iris Data", subtitle="Mean measurement values per species"
    )
    .tab_stub(rowname_col="species")
    .cols_label(petal_length="Length", petal_width="Width")
    .tab_spanner("Petal", cs.starts_with("petal"))
    .fmt_number("petal_width", decimals=2)
    .tab_style(
        style.fill("yellow"),
        loc.body(
            rows=pl.col("petal_length") == pl.col("petal_length").max(),
        ),
    )
)
# --8<-- [end:full-example]

# --8<-- [start:full-example-out]
from great_tables import loc, style

print(
    df.style.tab_header(
        title="Iris Data", subtitle="Mean measurement values per species"
    )
    .tab_stub(rowname_col="species")
    .cols_label(petal_length="Length", petal_width="Width")
    .tab_spanner("Petal", cs.starts_with("petal"))
    .fmt_number("petal_width", decimals=2)
    .tab_style(
        style.fill("yellow"),
        loc.body(
            rows=pl.col("petal_length") == pl.col("petal_length").max(),
        ),
    )
    .tab_style(
        style.text(weight="bold"),
        loc.body(columns="species"),
    )
    .as_raw_html()
)
# --8<-- [end:full-example-out]
