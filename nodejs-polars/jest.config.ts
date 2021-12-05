export default {
  clearMocks: true,
  collectCoverage: true,
  coverageDirectory: "coverage",
  coveragePathIgnorePatterns: ["/node_modules/"],
  coverageProvider: "v8",


  moduleDirectories: ["node_modules"],

  moduleFileExtensions: ["js", "ts", "node"],
  setupFilesAfterEnv : ["<rootDir>/__tests__/setup.ts"],
  moduleNameMapper: {
    "@polars": "<rootDir>/polars/index.ts",
    "@polars/(.*)$": "<rootDir>/polars/*",
  },
  testPathIgnorePatterns: ["<rootDir>/__tests__/setup.ts"]
};
