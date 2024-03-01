# Lazy / eager API

Polars supports two modes of operation: lazy and eager. In the eager API the query is executed immediately while in the lazy API the query is only evaluated once it is 'needed'. Deferring the execution to the last minute can have significant performance advantages and is why the Lazy API is preferred in most cases. Let us demonstrate this with an example:

{{code_block('user-guide/concepts/lazy-vs-eager','eager',['read_csv'])}}

In this example we use the eager API to:

1. Read the iris [dataset](https://archive.ics.uci.edu/dataset/53/iris).
1. Filter the dataset based on sepal length
1. Calculate the mean of the sepal width per species

Every step is executed immediately returning the intermediate results. This can be very wasteful as we might do work or load extra data that is not being used. If we instead used the lazy API and waited on execution until all the steps are defined then the query planner could perform various optimizations. In this case:

- Predicate pushdown: Apply filters as early as possible while reading the dataset, thus only reading rows with sepal length greater than 5.
- Projection pushdown: Select only the columns that are needed while reading the dataset, thus removing the need to load additional columns (e.g. petal length & petal width)

{{code_block('user-guide/concepts/lazy-vs-eager','lazy',['scan_csv'])}}

These will significantly lower the load on memory & CPU thus allowing you to fit bigger datasets in memory and process faster. Once the query is defined you call `collect` to inform Polars that you want to execute it. In the section on Lazy API we will go into more details on its implementation.

!!! info "Eager API"

    In many cases the eager API is actually calling the lazy API under the hood and immediately collecting the result. This has the benefit that within the query itself optimization(s) made by the query planner can still take place.

### When to use which

In general the lazy API should be preferred unless you are either interested in the intermediate results or are doing exploratory work and don't know yet what your query is going to look like.
