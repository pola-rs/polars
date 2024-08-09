# --8<-- [start:dataframe]
import polars as pl

path = "docs/data/iris.csv"

df = pl.scan_csv(path).group_by("species").agg(pl.col("petal_length").mean()).collect()
print(df)
# --8<-- [end:dataframe]

"""
# --8<-- [start:hvplot_show_plot]
df.plot.bar(
    x="species",
    y="petal_length",
    width=650,
)
# --8<-- [end:hvplot_show_plot]
"""

# --8<-- [start:hvplot_make_plot]
import hvplot

plot = df.plot.bar(
    x="species",
    y="petal_length",
    width=650,
)
hvplot.save(plot, "docs/images/hvplot_bar.html")
with open("docs/images/hvplot_bar.html", "r") as f:
    chart_html = f.read()
    print(f"{chart_html}")
# --8<-- [end:hvplot_make_plot]

"""
# --8<-- [start:matplotlib_show_plot]
import matplotlib.pyplot as plt

plt.bar(x=df["species"], height=df["petal_length"])
# --8<-- [end:matplotlib_show_plot]
"""

# --8<-- [start:matplotlib_make_plot]
import base64

import matplotlib.pyplot as plt

plt.bar(x=df["species"], height=df["petal_length"])
plt.savefig("docs/images/matplotlib_bar.png")
with open("docs/images/matplotlib_bar.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:matplotlib_make_plot]

"""
# --8<-- [start:seaborn_show_plot]
import seaborn as sns
sns.barplot(
    df,
    x="species",
    y="petal_length",
)
# --8<-- [end:seaborn_show_plot]
"""

# --8<-- [start:seaborn_make_plot]
import seaborn as sns

sns.barplot(
    df,
    x="species",
    y="petal_length",
)
plt.savefig("docs/images/seaborn_bar.png")
with open("docs/images/seaborn_bar.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:seaborn_make_plot]

"""
# --8<-- [start:plotly_show_plot]
import plotly.express as px

px.bar(
    df,
    x="species",
    y="petal_length",
    width=400,
)
# --8<-- [end:plotly_show_plot]
"""

# --8<-- [start:plotly_make_plot]
import plotly.express as px

fig = px.bar(
    df,
    x="species",
    y="petal_length",
    width=650,
)
fig.write_html("docs/images/plotly_bar.html", full_html=False, include_plotlyjs="cdn")
with open("docs/images/plotly_bar.html", "r") as f:
    chart_html = f.read()
    print(f"{chart_html}")
# --8<-- [end:plotly_make_plot]

"""
# --8<-- [start:altair_show_plot]
import altair as alt

alt.Chart(df, width=700).mark_bar().encode(x="species:N", y="petal_length:Q")
# --8<-- [end:altair_show_plot]
"""

# --8<-- [start:altair_make_plot]
import altair as alt

chart = (
    alt.Chart(df, width=600)
    .mark_bar()
    .encode(
        x="species:N",
        y="petal_length:Q",
    )
)
chart.save("docs/images/altair_bar.html")
with open("docs/images/altair_bar.html", "r") as f:
    chart_html = f.read()
    print(f"{chart_html}")
# --8<-- [end:altair_make_plot]
