<h1 align="center">
  <a href="https://pola.rs">
    <img src="https://raw.githubusercontent.com/pola-rs/polars-static/master/banner/polars_github_banner.svg" alt="Polars logo">
  </a>
</h1>

<div align="center">
  <a href="https://crates.io/crates/polars">
    <img src="https://img.shields.io/crates/v/polars.svg" alt="crates.io Latest Release"/>
  </a>
  <a href="https://pypi.org/project/polars/">
    <img src="https://img.shields.io/pypi/v/polars.svg" alt="PyPi Latest Release"/>
  </a>
  <a href="https://www.npmjs.com/package/nodejs-polars">
    <img src="https://img.shields.io/npm/v/nodejs-polars.svg" alt="NPM Latest Release"/>
  </a>
  <a href="https://community.r-multiverse.org/polars">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcommunity.r-multiverse.org%2Fapi%2Fpackages%2Fpolars&query=%24.Version&label=r-multiverse" alt="R-multiverse Latest Release"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.7697217">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7697217.svg" alt="DOI Latest Release"/>
  </a>
</div>

<p align="center">
  <b>Документация</b>:
  <a href="https://docs.pola.rs/api/python/stable/reference/index.html">Python</a>
  -
  <a href="https://docs.rs/polars/latest/polars/">Rust</a>
  -
  <a href="https://pola-rs.github.io/nodejs-polars/index.html">Node.js</a>
  -
  <a href="https://pola-rs.github.io/r-polars/index.html">R</a>
  |
  <b>StackOverflow</b>:
  <a href="https://stackoverflow.com/questions/tagged/python-polars">Python</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/rust-polars">Rust</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/nodejs-polars">Node.js</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/r-polars">R</a>
  |
  <a href="https://docs.pola.rs/">Гайд пользователя</a>
  |
  <a href="https://discord.gg/4UfP5cfBE7">Discord</a>
</p>

## Polars: экстремально быстрый Query Engine для DataFrame, написанный на Rust

Polars - это аналитический движок запросов для DataFrame. Он спроектирован так, чтобы быть быстрым,
удобным и выразительным. Ключевые возможности:

- Lazy | Eager execution
- Streaming (датасеты больше, чем RAM)
- Query optimization
- Multi-threaded
- Написан на Rust
- SIMD
- Мощный expression API
- Frontend на Python | Rust | NodeJS | R | SQL
- [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html)

Подробнее - в [гайде пользователя](https://docs.pola.rs/).

## Производительность 🚀🚀

### Молниеносная скорость

Polars очень быстрый. Фактически это одно из самых производительных решений на рынке. Смотрите
результаты [бенчмарков PDS-H](https://www.pola.rs/benchmarks.html).

### Легковесный

Polars также очень легковесный. У него ноль обязательных зависимостей, и это отлично видно по
времени импорта:

- polars: 70ms
- numpy: 104ms
- pandas: 520ms

### Работает с данными больше RAM

Если данные не помещаются в память, query engine Polars умеет выполнять ваш запрос (или его части)
в streaming-режиме. Это радикально снижает требования к памяти, поэтому вы сможете обработать,
например, датасет на 250GB даже на ноутбуке. Чтобы запустить запрос в streaming-режиме,
используйте `collect(engine='streaming')`.

## Установка

### Python

Установите последнюю версию Polars:

```sh
pip install polars
```

Подробности об optional dependencies смотрите в
[гайде пользователя](https://docs.pola.rs/user-guide/installation/#feature-flags).

Чтобы посмотреть текущую версию Polars и полный список дополнительных зависимостей, выполните:

```python
pl.show_versions()
```

## Contributing

Хотите внести вклад? Прочитайте наш
[гайд по contrib'у](https://docs.pola.rs/development/contributing/).

## Managed/Distributed Polars

Нужен managed-вариант или масштабирование на distributed-кластеры? Обратите внимание на наше
[предложение](https://cloud.pola.rs/) и поддержите проект!

## Python: сборка Polars из исходников

Если вам нужен bleeding-edge релиз или максимальная производительность, собирайте Polars из
исходников.

Для этого последовательно выполните следующие шаги:

1. Установите свежий [компилятор Rust](https://www.rust-lang.org/tools/install)
2. Установите [maturin](https://maturin.rs/): `pip install maturin`
3. Выполните `cd py-polars` и выберите один из вариантов:
   - `make build` - медленный бинарник с debug assertions и символами, быстрая компиляция
   - `make build-release` - быстрый бинарник без debug assertions, минимум debug symbols, долгая
     компиляция
   - `make build-nodebug-release` - то же, что build-release, но без debug symbols; компилируется
     чуть быстрее
   - `make build-debug-release` - то же, что build-release, но с полными debug symbols;
     компилируется чуть медленнее
   - `make build-dist-release` - самый быстрый бинарник, экстремально долгая компиляция

По умолчанию бинарник компилируется с оптимизациями под современный CPU. Если у вас более старый
процессор и нет поддержки, например, AVX2, добавьте к команде `LTS_CPU=1`.

Обратите внимание: Rust crate с Python bindings называется `py-polars`, чтобы отличаться от
оборачиваемого Rust crate `polars`. При этом и Python package, и Python module называются `polars`,
поэтому вы можете просто `pip install polars` и `import polars`.

## Использование кастомных Rust-функций в Python

Расширять Polars UDF-функциями, скомпилированными на Rust, просто. Мы предоставляем PyO3
extensions для структур данных `DataFrame` и `Series`. Подробнее:
https://github.com/pola-rs/polars/tree/main/pyo3-polars.

## Когда нужен big scale...

Ожидаете больше 2^32 (~4.2 млрд) строк? Соберите Polars с feature flag `bigidx` или, если вы
используете Python, установите `pip install polars[rt64]`.

Не используйте это, пока не упрётесь в лимит по числу строк: дефолтная сборка Polars быстрее и
потребляет меньше памяти.

## Legacy

Хотите запускать Polars на старом CPU (например, выпущенном до 2011 года) или на `x86-64` сборке
Python под Rosetta на Apple Silicon? Установите `pip install polars[rtcompat]`. Эта версия Polars
собрана без target-фич [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions).
