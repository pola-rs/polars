
// Typescript has some difficulties detecting '.node' files. 
// it likely can be done, I need to look into it further.
const polarsInternal = require("../bin/libpolars.node")
module.exports = polarsInternal;