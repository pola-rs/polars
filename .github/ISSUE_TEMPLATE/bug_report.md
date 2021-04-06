---
name: Bug report
about: An issue with rust polars or python polars
title: ''
labels: ''
assignees: ''
---

#### Are you using Python or Rust?

Replace this text with the **Rust** or **Python**.

#### What version of polars are you using?

Replace this text with the version.

#### What operating system are you using polars on?

Replace this text with your operating system and version.

#### Describe your bug.

Give a high level description of the bug.

#### What are the steps to reproduce the behavior?

If possible, please include a minimal example on a dataset that is create through code:

Please use code instead of images, we don't like typing.

If the example is large, put it in a gist: https://gist.github.com/

If the example is small, put it in code fences:

```
your
code
goes
here
```

**Example**

```python
import polars as pl
import numpy as np

# Create a simple dataset on which we can reproduce the bug.
pl.DataFrame({
    "foo": [None, 1, 2],
    "bar": np.arange(3)
})
```

If we cannot reproduce the bug, it is unlikely that we will be able fix it.

#### What is the actual behavior?

Show the query you ran and the actual output. 

If the output is large, put it in a gist: https://gist.github.com/

If the output is small, put it in code fences:

```
your
output
goes
here
```

#### What is the expected behavior?

What do you think polars should have done?
