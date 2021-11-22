export default {
  clearMocks: true,
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coveragePathIgnorePatterns: ['/node_modules/'],
  coverageProvider: 'v8',


  moduleDirectories: ['node_modules'],

  moduleFileExtensions: ['js', 'ts', 'node'],

  moduleNameMapper: {
    '@polars': '<rootDir>/polars/index.ts',
    '@polars/(.*)$': '<rootDir>/polars/*',
  },
  testMatch: ['__tests__/**/*.[jt]s?(x)'],

};
