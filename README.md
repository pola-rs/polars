# Polars
[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
![Build and test](https://github.com/ritchie46/polars/workflows/Build%20and%20test/badge.svg)
[![](http://meritbadge.herokuapp.com/polars)](https://crates.io/crates/polars)
[![Gitter](https://badges.gitter.im/polars-rs/community.svg)](https://gitter.im/polars-rs/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Blazingly fast DataFrames in Rust & Python

Polars is a blazingly fast DataFrames library implemented in Rust using Apache Arrow(2) as memory model.

* Lazy | eager execution
* Multi-threaded
* SIMD
* Query optimization
* Powerful expression API
* Rust | Python | ...

To learn more, read the [User Guide](https://pola-rs.github.io/polars-book/).

## Performance
Polars is very fast, and in fact is one of the best performing solutions available. 
See the results in [h2oai's db-benchmark](https://h2oai.github.io/db-benchmark/).

# Rust setup
You can take latest release from `crates.io`, or if you want to use the latest features/ performance improvements
point to the `master` branch of this repo.

```toml
polars = {git = "https://github.com/ritchie46/polars", rev = "<optional git tag>" } 
```
## Rust version
Required Rust version `>=1.52`

# Python users read this!
Polars is currently transitioning from `py-polars` to `polars`. Some docs may still refer the old name. 

Install the latest polars version with: 
`$ pip3 install polars`

## Documentation
Want to know about all the features Polars support? Read the docs!

#### Rust
* [Documentation (master branch)](https://pola-rs.github.io/polars/polars/index.html). 
* [User Guide](https://pola-rs.github.io/polars-book/)
    
#### Python
* installation guide: `$ pip3 install polars`
* [User Guide](https://pola-rs.github.io/polars-book/)
* [Reference guide](https://pola-rs.github.io/polars/py-polars/html/reference/index.html)

## Contribution
Want to contribute? Read our [contribution guideline](https://github.com/ritchie46/polars/blob/master/CONTRIBUTING.md).

## \[Python\] compile py-polars from source
If you want a bleeding edge release or maximal performance you should compile **py-polars** from source.

This can be done by going through the following steps in sequence:

1. install the latest [rust compiler](https://www.rust-lang.org/tools/install)
2. `$ pip3 install maturin`
4.  Choose any of:
  * Very long compile times, fastest binary: `$ cd py-polars && maturin develop --rustc-extra-args="-C target-cpu=native" --release`
  * Shorter compile times, fast binary: `$ cd py-polars && maturin develop --rustc-extra-args="-C codegen-units=16 -C lto=thin -C target-cpu=native" --release
    `

Note that the Rust crate implementing the Python bindings is called `py-polars` to distinguish from the wrapped 
Rust crate `polars` itself. However, both the Python package and the Python module are named `polars`, so you
can `pip install polars` and `import polars` (previously, these were called `py-polars` and `pypolars`).

## Arrow2
Polars has a fully functional [arrow2](https://crates.io/crates/arrow2) branch and will ship the python binaries
from this branch. Arrow2 is a faster and safer implementation of the arrow spec. Arrow2 also has a more granular code base,
helping to reduce the compiler bloat.

## Acknowledgements
Development of Polars is proudly powered by

[![Xomnia](https://raw.githubusercontent.com/ritchie46/img/master/polars/xomnia_logo.png)](https://www.xomnia.com)