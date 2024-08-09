"""
# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:df]
q1 = (
    pl.scan_csv("docs/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
)
# --8<-- [end:df]

# --8<-- [start:collect]
q4 = (
    pl.scan_csv(f"docs/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .collect()
)
# --8<-- [end:collect]
# --8<-- [start:stream]
q5 = (
    pl.scan_csv(f"docs/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .collect(streaming=True)
)
# --8<-- [end:stream]
# --8<-- [start:partial]
q9 = (
    pl.scan_csv(f"docs/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .fetch(n_rows=int(100))
)
# --8<-- [end:partial]

# --8<-- [start:background]
import time

background_query = (
    pl.scan_csv("docs/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .collect(background=True)
)

def wait_for_query(background_query, *, max_wait_time):
    start_time = time.monotonic()

    while True:
        result = background_query.fetch()
        if result is not None:
            return result  # query successfully completed

        current_time = time.monotonic()
        if (current_time - start_time) > max_wait_time:
            background_query.cancel()
            msg = f"Background query took more than {max_wait_time} s"
            raise RuntimeError(msg)

        # continue waiting
        time.sleep(1)

result = wait_for_query(background_query, max_wait_time=10)
# --8<-- [end:background]
"""
