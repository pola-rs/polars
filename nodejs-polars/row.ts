import pl from "./bin/index";

const data = [
  {
    "id": "a",
    "name": "fred",
    "country": "france",
    "age": 30,
    "city": "paris" // there's no "city" property elsewhere
  },
  {
    "id": "b",
    "name": "alexandra",
    "country": "usa",
    "age": 40n
  },
  {
    "id": "c",
    "name": "george",
    "country": "argentina",
  }
];

const df = pl.DataFrame(data);
console.log(df)