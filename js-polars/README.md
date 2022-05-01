## Running locally.

### Prerequisites
- node
- wasm-pack
- python
- rust nightly
- yarn 2+


since polars must use workers for multithreading, you must run the polars code inside of a worker. 

1. 
   `wasm-pack build -t web`

2. upload a csv file to `./examples`
3. change worker.js line:2 to point to your csv file. 

4.  run `python server.py`

5. open your browser to `localhost:8000/index.html`

6. open the browser console & you should see the output from `worker.js`

