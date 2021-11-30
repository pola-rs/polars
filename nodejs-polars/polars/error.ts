export class InvalidOperationError extends Error {

  constructor(method, dtype) {
    super(`Invalid operation: ${method} is not supported for ${dtype}`);
  }
}