export class InvalidOperationError extends RangeError {

  constructor(method, dtype) {
    super(`Invalid operation: ${method} is not supported for ${dtype}`);
  }
}

export class NotImplemented extends Error {

  constructor(method, dtype) {
    super(`Invalid operation: ${method} is not supported for ${dtype}`);
  }
}

export const todo = () => new Error("not yet implemented");
