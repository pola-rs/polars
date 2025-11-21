# Plugins

Polars allows you to extend its functionality with either Expression plugins or IO plugins.

- [Expression plugins](./expr_plugins.md)
- [IO plugins](./io_plugins.md)

## Community plugins

Here is a curated (non-exhaustive) list of community-implemented plugins.

### Various

- [polars-xdt](https://github.com/pola-rs/polars-xdt) Polars plugin with extra datetime-related
  functionality which isn't quite in-scope for the main library
- [polars-hash](https://github.com/ion-elgreco/polars-hash) Stable non-cryptographic and
  cryptographic hashing functions for Polars

### Data science

- [polars-distance](https://github.com/ion-elgreco/polars-distance) Polars plugin for pairwise
  distance functions
- [polars-ds](https://github.com/abstractqqq/polars_ds_extension) Polars extension aiming to
  simplify common numerical/string data analysis procedures

### Geo

- [polars-st](https://github.com/Oreilles/polars-st) Polars ST provides spatial operations on Polars
  DataFrames, Series and Expressions. Just like Shapely and Geopandas.
- [polars-reverse-geocode](https://github.com/MarcoGorelli/polars-reverse-geocode) Offline reverse
  geocoder for finding the closest city to a given (latitude, longitude) pair.
- [polars-h3](https://github.com/Filimoa/polars-h3) This is a Polars extension that adds support for
  the H3 discrete global grid system, so you can index points and geometries to hexagons directly in
  Polars.

## Other material

- [Ritchie Vink - Keynote on Polars Plugins](https://youtu.be/jKW-CBV7NUM)
- [Polars plugins tutorial](https://marcogorelli.github.io/polars-plugins-tutorial/) Learn how to
  write a plugin by going through some very simple and minimal examples
- [cookiecutter-polars-plugin](https://github.com/MarcoGorelli/cookiecutter-polars-plugins) Project
  template for Polars Plugins
