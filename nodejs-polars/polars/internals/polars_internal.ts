/* eslint-disable camelcase */
import {loadBinding} from "@node-rs/helper";
import {join} from "path";

// eslint-disable-next-line no-undef
const up2 = join(__dirname, "../", "../");
const bindings =  loadBinding("up2", "nodejs-polars", "nodejs-polars");
export default bindings;
