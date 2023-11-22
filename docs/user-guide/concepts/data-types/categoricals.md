# Categorical Data

Categorical data represents string data where the values in the column have a finite set of values (usually way smaller than the length of the column). You can think about columns on gender, countries, currency pairings, etc. Storing these values as plain strings is a waste of memory and performance as we will be repeating the same string over and over again. Additionally, in the case of joins we are stuck with expensive string comparisons.

That is why Polars supports encoding string values in dictionary format. Working with categorical data in Polars can be done with two different DataTypes: `Enum`,`Categorical`. Both have their own use cases which we will explain further on this page.
First we will look at what a categorical is in Polars.

In Polars a categorical is a defined as a string column which is encoded by a dictionary. A string column would be split into two elements: encodings and the actual string values.

<table>
<tr><th>String Column </th><th>Categorical Column</th></tr>
<tr><td>
<table>
    <thead>
        <tr>
            <th>Series</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Polar Bear</td>
        </tr>
        <tr>
            <td>Panda Bear</td>
        </tr>
        <tr>
            <td>Brown Bear</td>
        </tr>
        <tr>
            <td>Panda Bear</td>
        </tr>
        <tr>
            <td>Brown Bear</td>
        </tr>
        <tr>
            <td>Brown Bear</td>
        </tr>
        <tr>
            <td>Polar Bear</td>
        </tr>
    </tbody>
</table>
</td>
<td>
<table>
<tr>
<td>

<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>0</td>
        </tr>
    </tbody>
</table>

</td>
<td>
<table>
    <thead>
        <tr>
            <th>Categories</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Polar Bear</td>
        </tr>
        <tr>
            <td>Panda Bear</td>
        </tr>
        <tr>
            <td>Brown Bear</td>
        </tr>
    </tbody>
</table>
</td>
</tr>
</table>
</td>
</tr>
</table>

The physical `0` in this case encodes (or maps) to the value 'Polar Bear', the value `1` encodes to 'Panda Bear' and the value `2` to 'Brown Bear'. This encoding has the benefit of only storing the string values once. Additionally, when we perform operations (e.g. sorting, counting) we can work directly on the physical representation which is much faster than the working with string data.

### `Enum` vs `Categorical`

Polars supports two different DataTypes for working with categorical data: `Enum` and `Categorical`. When the categories are known up front use `Enum`. When you don't know the categories or they are not fixed then you use `Categorical`. In case your requirements change along the way you can always cast from one to the other.

{{code_block('user-guide/concepts/data-types/categoricals','example',[])}}

From the code block above you can see that the `Enum` data type requires the upfront while the categorical data type infers the categories.

#### `Categorical` Data Type

The `Categorical` data type is a flexible one. Polars will add categories on the fly if it sees them. This sounds like a strictly better version compared to the `Enum` data type as we can simply infer the categories, however inferring comes at a cost. The main cost here is we have no control over our encodings.

Consider the following scenario where we append the following two categorical `Series`

{{code_block('user-guide/concepts/data-types/categoricals','append',[])}}

Polars encodes the string values in order as they appear. So the series would look like this:

<table>
<tr><th>cat_series </th><th>cat2_series</th></tr>
<tr><td>
<table>
<tr>
<td>
<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>0</td>
        </tr>
    </tbody>
</table>

</td>
<td>
<table>
    <thead>
        <tr>
            <th>Categories</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Polar</td>
        </tr>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
    </tbody>
</table>

</td>
</tr>
</table>
</td>
<td>
<table>
<tr>
<td>
<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
    </tbody>
</table>

</td>
<td>

<table>
    <thead>
        <tr>
            <th>Categories</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
        <tr>
            <td>Polar</td>
        </tr>
    </tbody>
</table>

</td>
</tr>
</table>
</td>
</tr>
</table>

Combining the `Series` becomes a non-trivial task which is expensive as the physical value of `0` represents something different in both `Series`. Polars does support these types of operations for convenience, however in general these should be avoided due to its slower performance as it requires making both encodings compatible first before doing any merge operations.

##### Using the global string Cache

One way to handle this problem is to enable a `StringCache`. When you enable the `StringCache` strings are no longer encoded in the order they appear on a per-column basis. Instead, the string cache ensures a single encoding for each string. The string `Polar` will always map the same physical for all categorical columns made under the string cache.
Merge operations (e.g. appends, joins) are cheap as there is no need to make the encodings compatible first, solving the problem we had above.

{{code_block('user-guide/concepts/data-types/categoricals','global_append',[])}}

However, the string cache does come at a small performance hit during construction of the `Series` as we need to look up / insert the string value in the cache. Therefore, it is preferred to use the `Enum` Data Type if you know your categories in advance.

#### `Enum Data Type`

In the `Enum` data type we specify the categories in advance. This way we ensure categoricals from different columns or different datasets have the same encoding and there is no need for expensive re-encoding or cache lookups.

{{code_block('user-guide/concepts/data-types/categoricals','enum_append',[])}}

Polars will raise an `OutOfBounds` error when a value is encountered which is not specified in the `Enum`.

{{code_block('user-guide/concepts/data-types/categoricals','enum_error',[])}}
