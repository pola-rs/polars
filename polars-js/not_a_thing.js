const pln = require("./index")

const data = [
  {foo: "a", bar: 1},
  {foo: "a", bar: 1},
  {foo: "a", bar: "a"}
]
const df = pln.(data)
console.log(df.toString())