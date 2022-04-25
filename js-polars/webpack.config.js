const path = require("path");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

const dist = path.resolve(__dirname, "dist");

module.exports = {
  devtool: 'source-map',
  entry: {
    index: "./index.js"
  },
  output: {
    publicPath: '/dist/',
    path: dist,

    filename: "polars.js"
  },
  devServer: {
    contentBase: dist,
  },
  // plugins: [
  //   new WasmPackPlugin({
  //     crateDirectory: __dirname,
  //   }),
  // ],
  experiments: {
    // asyncWebAssembly: true,
    // outputModule: true,
    syncWebAssembly: true,
    futureDefaults: true,
  },
};
