const pl = require("./pkg")
const assert = require("assert")

let s = new pl.Series("a", [1, 2, 3])
assert(s.mean() === 2)
console.log(s.mean())
