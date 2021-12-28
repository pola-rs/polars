import { pathsToModuleNameMapper } from "ts-jest";
import {compilerOptions} from "./tsconfig.json";

export default {
  globals: {
    "ts-jest": {
      tsconfig: "tsconfig.test.json",
    },
  },
  preset: "ts-jest",
  testEnvironment: "node",
  clearMocks: true,
  collectCoverage: true,
  moduleDirectories: ["node_modules", "./polars"],
  moduleFileExtensions: ["js", "ts"],
  setupFilesAfterEnv : ["<rootDir>/__tests__/setup.ts"],
  moduleNameMapper: pathsToModuleNameMapper(compilerOptions.paths, { prefix: "<rootDir>/polars" }),
  testPathIgnorePatterns: ["<rootDir>/__tests__/setup.ts"]
};
