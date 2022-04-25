
build-dev:
	RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals'; wasm-pack build --dev --target nodejs -d pkg/node
build-web:
	RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals';  wasm-pack build --dev --target web -d pkg/browser
build-prod:
	wasm-pack build --target nodejs 
