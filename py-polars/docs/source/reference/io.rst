
============
Input/output
============
.. currentmodule:: polars


CSV
~~~
.. autosummary::
   :toctree: api/

   read_csv
   read_csv_batched
   scan_csv
   DataFrame.write_csv

Feather/ IPC
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_ipc
   scan_ipc
   read_ipc_schema
   DataFrame.write_ipc

Parquet
~~~~~~~
.. autosummary::
   :toctree: api/

   read_parquet
   scan_parquet
   read_parquet_schema
   DataFrame.write_parquet

SQL
~~~
.. autosummary::
   :toctree: api/

   read_sql

JSON
~~~~
.. autosummary::
   :toctree: api/

   read_json
   read_ndjson
   scan_ndjson
   DataFrame.write_json
   DataFrame.write_ndjson

AVRO
~~~~
.. autosummary::
   :toctree: api/

   read_avro
   DataFrame.write_avro

Excel
~~~~~
.. autosummary::
   :toctree: api/

   read_excel

Delta Lake
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   scan_delta
   read_delta

Datasets
~~~~~~~~
Connect to pyarrow datasets.

.. autosummary::
   :toctree: api/

   scan_ds


BatchedCsvReader
~~~~~~~~~~~~~~~~
This reader comes available by calling `pl.read_csv_batched`.

.. currentmodule:: polars.internals.batched

.. autosummary::
   :toctree: api/

    BatchedCsvReader.next_batches
