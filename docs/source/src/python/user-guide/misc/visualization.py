# --8<-- [start:dataframe]
import polars as pl

path = "docs/assets/data/iris.csv"

df = pl.read_csv(path)
print(df)
# --8<-- [end:dataframe]

"""
# --8<-- [start:hvplot_show_plot]
import hvplot.polars
df.hvplot.scatter(
    x="sepal_width",
    y="sepal_length",
    by="species",
    width=650,
    title="Irises",
    xlabel='Sepal Width',
    ylabel='Sepal Length',
)
# --8<-- [end:hvplot_show_plot]
"""

# --8<-- [start:hvplot_make_plot]
import hvplot.polars

plot = df.hvplot.scatter(
    x="sepal_width",
    y="sepal_length",
    by="species",
    width=650,
    title="Irises",
    xlabel="Sepal Width",
    ylabel="Sepal Length",
)
hvplot.save(plot, "docs/assets/images/hvplot_scatter.html", resources="cdn")
with open("docs/assets/images/hvplot_scatter.html", "r") as f:
    chart_html = f.read()
    print(f"{chart_html}")
# --8<-- [end:hvplot_make_plot]

"""
# --8<-- [start:matplotlib_show_plot]
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(
    x=df["sepal_width"],
    y=df["sepal_length"],
    c=df["species"].cast(pl.Categorical).to_physical(),
)
ax.set_title('Irises')
ax.set_xlabel('Sepal Width')
ax.set_ylabel('Sepal Length')
# --8<-- [end:matplotlib_show_plot]
"""

# --8<-- [start:matplotlib_make_plot]
import base64

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(
    x=df["sepal_width"],
    y=df["sepal_length"],
    c=df["species"].cast(pl.Categorical).to_physical(),
)
ax.set_title("Irises")
ax.set_xlabel("Sepal Width")
ax.set_ylabel("Sepal Length")
fig.savefig("docs/assets/images/matplotlib_scatter.png")
with open("docs/assets/images/matplotlib_scatter.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:matplotlib_make_plot]

"""
# --8<-- [start:plotnine_show_plot]
from plotnine import ggplot, aes, geom_point, labs

(
    ggplot(df, mapping=aes(x="sepal_width", y="sepal_length", color="species"))
    + geom_point()
    + labs(title="Irises", x="Sepal Width", y="Sepal Length")
)
# --8<-- [end:plotnine_show_plot]
"""

# --8<-- [start:plotnine_make_plot]
import base64

from plotnine import ggplot, aes, geom_point, labs

fig_path = "docs/assets/images/plotnine.png"

(
    ggplot(df, mapping=aes(x="sepal_width", y="sepal_length", color="species"))
    + geom_point()
    + labs(title="Irises", x="Sepal Width", y="Sepal Length")
).save(fig_path, dpi=300, verbose=False)

with open(fig_path, "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:plotnine_make_plot]

"""
# --8<-- [start:seaborn_show_plot]
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.scatterplot(
    df,
    x="sepal_width",
    y="sepal_length",
    hue="species",
    ax=ax,
)
ax.set_title('Irises')
ax.set_xlabel('Sepal Width')
ax.set_ylabel('Sepal Length')
# --8<-- [end:seaborn_show_plot]
"""

# --8<-- [start:seaborn_make_plot]
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.scatterplot(
    df,
    x="sepal_width",
    y="sepal_length",
    hue="species",
    ax=ax,
)
ax.set_title("Irises")
ax.set_xlabel("Sepal Width")
ax.set_ylabel("Sepal Length")
fig.savefig("docs/assets/images/seaborn_scatter.png")
with open("docs/assets/images/seaborn_scatter.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:seaborn_make_plot]

"""
# --8<-- [start:plotly_show_plot]
import plotly.express as px

px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    width=650,
    title="Irises",
    labels={'sepal_width': 'Sepal Width', 'sepal_length': 'Sepal Length'}
)
# --8<-- [end:plotly_show_plot]
"""

# --8<-- [start:plotly_make_plot]
import plotly.express as px

fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    width=650,
    title="Irises",
    labels={"sepal_width": "Sepal Width", "sepal_length": "Sepal Length"},
)
fig.write_html(
    "docs/assets/images/plotly_scatter.html", full_html=False, include_plotlyjs="cdn"
)
with open("docs/assets/images/plotly_scatter.html", "r") as f:
    chart_html = f.read()
    print(f"{chart_html}")
# --8<-- [end:plotly_make_plot]

"""
# --8<-- [start:altair_show_plot]
chart =  (
    df.plot.point(
        x="sepal_width",
        y="sepal_length",
        color="species",
    )
    .properties(width=500, title="Irises")
    .configure_scale(zero=False)
    .configure_axisX(tickMinStep=1)
)
chart.encoding.x.title = "Sepal Width"
chart.encoding.y.title = "Sepal Length"
chart
# --8<-- [end:altair_show_plot]
"""

# --8<-- [start:altair_make_plot]
chart = (
    df.plot.point(
        x="sepal_width",
        y="sepal_length",
        color="species",
    )
    .properties(width=500, title="Irises")
    .configure_scale(zero=False)
    .configure_axisX(tickMinStep=1)
)
chart.encoding.x.title = "Sepal Width"
chart.encoding.y.title = "Sepal Length"
chart.save("docs/assets/images/altair_scatter.html")
with open("docs/assets/images/altair_scatter.html", "r") as f:
    chart_html = f.read()
    print(f"{chart_html}")
# --8<-- [end:altair_make_plot]
