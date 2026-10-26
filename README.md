# Ail

**Ail** is an ultra-fast, AI-native programming language with multiple backends:

- **Cranelift JIT** (dev): instant compile + run
- **LLVM IR** (`--target llvm`): max native speed, no C layer for core logic
- **C** (`--target c`): GCC -O3 -march=native fallback
- **WASM** (`--target wasm`): runs in browsers

No garbage collector. No interpreter at runtime (AOT to native binary). Binary cache avoids recompiling unchanged code.

## Build

```sh
cd ailc
cargo build --release
```

## Usage

```sh
ailc file.ail                  # Cranelift JIT
ailc --target llvm file.ail    # LLVM IR (fastest native)
ailc --target c file.ail       # C backend
ailc --target wasm file.ail    # WebAssembly + HTML harness
```

## Example

```lisp
(fn fib (n i64) i64
  (if (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))

(fn main () void
  (print (fib 35)))
```

## Download

Prebuilt binaries are attached to each [GitHub Release](https://github.com/nathent2023-glitch/ail/releases). Or build from source with Rust + (optionally) LLVM/Clang and GCC.

## Cache

Compiled binaries are cached in `~/.ail/cache/<hash>/` keyed by source + target. Repeat runs are instant.

## License

MIT
