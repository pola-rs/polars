const path = require('path')
const WasmPackPlugin = require('@wasm-tool/wasm-pack-plugin')

module.exports = {
  entry: './pkg/index.js', // input file of the JS bundle
  output: {
    library: {
      // name: 'polars',
      type: 'module'
    },
    // filename: 'bundle.js', // output filename
    path: path.resolve(__dirname, 'dist'), // directory of where the bundle will be created at
    // le
  },
  // plugins: [
  //   new WasmPackPlugin({
  //     crateDirectory: __dirname, 
  //     outDir: 'pkg',
  //     outName: 'index',
  //     forceMode: "development",
  //   }),
  // ],
  experiments: {
    // asyncWebAssembly: true,
    outputModule: true,
    syncWebAssembly: true,
    futureDefaults: true,
  },  
}