
============
Input/output
============
.. currentmodule:: polars


CSV
~~~
.. autosummary::
   :toctree: api/

   read_csv
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
   DataFrame.write_json

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

Datasets
~~~~~~~~
Connect to pyarrow datasets.

.. autosummary::
   :toctree: api/

   scan_ds
