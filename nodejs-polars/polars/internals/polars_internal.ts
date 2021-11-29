/* eslint-disable camelcase */
import {loadBinding} from "@node-rs/helper";
import {join} from "path";

// eslint-disable-next-line no-undef
const up1 = join(__dirname, "../", "../");
const polars_internal = loadBinding(up1, "nodejs-polars", "nodejs-polars");
export default polars_internal;
