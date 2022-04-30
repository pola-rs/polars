## Running locally.

### Prerequisites
- node
- wasm-pack
- python
- rust nightly
- yarn 2+


1. 
   `wasm-pack build -t web`
2. wasm-rayon will generate a `.js` file in `pkg/snippets/*/workerHelpers.js`
  you will need to update `line 54` from

    `const pkg = await import('../../..');`
    
    to 
  
    `const pkg = await import('../../../polars.js');`

3. run `yarn webpack` 

4.  run `python server.py`

5. open your browser to `localhost:8000/index.html`

6. open the browser console & you should see the output from `index.js` & `worker.js`

