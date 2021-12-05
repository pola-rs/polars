export class InvalidOperationError extends Error {

  constructor(method, dtype) {
    super(`Invalid operation: ${method} is not supported for ${dtype}`);
  }
}

export class NotImplemented extends Error {

  constructor(method, dtype) {
    super(`Invalid operation: ${method} is not supported for ${dtype}`);
  }
}
