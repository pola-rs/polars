# --8<-- [start:setup]
# --8<-- [end:setup]
# --8<-- [start:selectors_df]
from datetime import date, datetime

import polars as pl

df = pl.DataFrame(
    {
        "id": [9, 4, 2],
        "place": ["Mars", "Earth", "Saturn"],
        "date": pl.date_range(date(2022, 1, 1), date(2022, 1, 3), "1d", eager=True),
        "sales": [33.4, 2142134.1, 44.7],
        "has_people": [False, True, False],
        "logged_at": pl.datetime_range(
            datetime(2022, 12, 1), datetime(2022, 12, 1, 0, 0, 2), "1s", eager=True
        ),
    }
).with_row_index("index")
print(df)
# --8<-- [end:selectors_df]

# --8<-- [start:all]
out = df.select(pl.col("*"))

# Is equivalent to
out = df.select(pl.all())
print(out)
# --8<-- [end:all]

# --8<-- [start:exclude]
out = df.select(pl.col("*").exclude("logged_at", "index"))
print(out)
# --8<-- [end:exclude]

# --8<-- [start:expansion_by_names]
out = df.select(pl.col("date", "logged_at").dt.to_string("%Y-%h-%d"))
print(out)
# --8<-- [end:expansion_by_names]

# --8<-- [start:expansion_by_regex]
out = df.select(pl.col("^.*(as|sa).*$"))
print(out)
# --8<-- [end:expansion_by_regex]

# --8<-- [start:expansion_by_dtype]
out = df.select(pl.col(pl.Int64, pl.UInt32, pl.Boolean).n_unique())
print(out)
# --8<-- [end:expansion_by_dtype]

# --8<-- [start:selectors_intro]
import polars.selectors as cs

out = df.select(cs.integer(), cs.string())
print(out)
# --8<-- [end:selectors_intro]

# --8<-- [start:selectors_diff]
out = df.select(cs.numeric() - cs.first())
print(out)
# --8<-- [end:selectors_diff]

# --8<-- [start:selectors_union]
out = df.select(cs.by_name("index") | ~cs.numeric())
print(out)
# --8<-- [end:selectors_union]

# --8<-- [start:selectors_by_name]
out = df.select(cs.contains("index"), cs.matches(".*_.*"))
print(out)
# --8<-- [end:selectors_by_name]

# --8<-- [start:selectors_to_expr]
out = df.select(cs.temporal().as_expr().dt.to_string("%Y-%h-%d"))
print(out)
# --8<-- [end:selectors_to_expr]

# --8<-- [start:selectors_is_selector_utility]
from polars.selectors import is_selector

out = cs.temporal()
print(is_selector(out))
# --8<-- [end:selectors_is_selector_utility]

# --8<-- [start:selectors_colnames_utility]
from polars.selectors import expand_selector

out = cs.temporal().as_expr().dt.to_string("%Y-%h-%d")
print(expand_selector(df, out))
# --8<-- [end:selectors_colnames_utility]
