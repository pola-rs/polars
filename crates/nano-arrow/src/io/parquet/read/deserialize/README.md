# Design

## Non-nested types

Let's start with the design used for non-nested arrays. The (private) entry point of this
module for non-nested arrays is `simple::page_iter_to_arrays`.

This function expects

- a (fallible) streaming iterator of decompressed and encoded pages, `Pages`
- the source (parquet) column type, including its logical information
- the target (arrow) `DataType`
- the chunk size

and returns an iterator of `Array`, `ArrayIter`.

This design is shared among _all_ `(parquet, arrow)` implemented tuples. Their main
difference is how they are deserialized, which depends on the source and target types.

When the array iterator is pulled the first time, the following happens:

- a page from `Pages` is pulled
- a `PageState<'a>` is built from the page
- the `PageState` is consumed into a mutable array:
  - if `chunk_size` is larger than the number of rows in the page, the mutable array state is preserved and a new page is pulled and the process repeated until we fill a chunk.
  - if `chunk_size` is smaller than the number of rows in the page, the mutable array state
    is returned and the remaining of the page is consumed into multiple mutable arrays of length `chunk_size` into a FIFO queue.

Subsequent pulls of arrays will first try to pull from the FIFO queue. Once the queue is empty, the
a new page is pulled.

### `PageState`

As mentioned above, the iterator leverages the idea that we attach a state to a page. Recall
that a page is essentially `[header][data]`. The `data` part contains encoded
`[rep levels][def levels][non-null values]`. Some pages have an associated dictionary page,
in which case the `non-null values` represent the indices.

Irrespectively of the physical type, the main idea is to split the page in two iterators:

- An iterator over `def levels`
- An iterator over `non-null values`

and progress the iterators as needed. In particular, for non-nested types, `def levels` is
a bitmap with the same representation as Arrow, in which case the validity is extended directly.

The `non-null values` are "expanded" by filling null values with the default value of each physical
type.

## Nested types

For nested type with N+1 levels (1 is the primitive), we need to build the nest information of each
N levels + the non-nested Arrow array.

This is done by first transversing the parquet types and using it to initialize, per chunk, the N levels.

The per-chunk execution is then similar but `chunk_size` only drives the number of retrieved
rows from the outermost parquet group (the field). Each of these pulls knows how many items need
to be pulled from the inner groups, all the way to the primitive type. This works because
in parquet a row cannot be split between two pages and thus each page is guaranteed
to contain a full row.

The `PageState` of nested types is composed by 4 iterators:

- A (zipped) iterator over `rep levels` and `def levels`
- An iterator over `def levels`
- An iterator over `non-null values`

The idea is that an iterator of `rep, def` contain all the information to decode the
nesting structure of an arrow array. The other two iterators are equivalent to the non-nested
types with the exception that `def levels` are no equivalent to arrow bitmaps.
