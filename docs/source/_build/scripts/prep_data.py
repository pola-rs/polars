"""
Downloads data once when serving the docs so that subsequent
subsequent rebuilds do not have to access remote resources again.
"""

import requests


DATA = [
    (
        "https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/data/monopoly_props_groups.csv",
        "docs/assets/data/monopoly_props_groups.csv",
    ),
    (
        "https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/data/monopoly_props_prices.csv",
        "docs/assets/data/monopoly_props_prices.csv",
    ),
]


for url, dest in DATA:
    with open(dest, "wb") as f:
        try:
            f.write(requests.get(url, timeout=10).content)
        except Exception as e:
            print(f"WARNING: failed to download file {dest} ({e})")
        else:
            print(f"INFO: downloaded {dest}")
