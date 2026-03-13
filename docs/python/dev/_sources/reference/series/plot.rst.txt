====
Plot
====

Polars does not implement plotting logic itself, but instead defers to
hvplot. Please see the `hvplot reference gallery <https://hvplot.holoviz.org/reference/index.html>`_
for more information and documentation.

Examples
--------
Histogram:

.. code-block:: python

   s = pl.Series([1, 4, 2])
   s.plot.hist()

KDE plot (note: in addition to ``hvplot``, this one also requires ``scipy``):

.. code-block:: python

   s.plot.kde()

For more info on what you can pass, you can use ``hvplot.help``:

.. code-block:: python

   import hvplot
   hvplot.help("hist")
