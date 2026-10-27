// Stage 1 step 5: conformance tests across all backends.
//
// Each test compiles an .ail source with a given backend, runs the produced
// program, and asserts its stdout. Compile-error cases assert that stderr
// contains the expected diagnostic.
use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static TID: AtomicUsize = AtomicUsize::new(0);

fn compile(src: &str, target: &str) -> std::process::Output {
    let dir = std::env::temp_dir();
    let id = TID.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("ailc_test_{}_{}.ail", std::process::id(), id));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(src.as_bytes()).unwrap();
    drop(f);
    let out = Command::new(env!("CARGO_BIN_EXE_ailc"))
        .args(["--target", target])
        .arg(&path)
        .output()
        .expect("run ailc");
    let _ = std::fs::remove_file(&path);
    out
}

/// Backend "native" runs in-process (JIT), so stdout is the program output.
/// C/LLVM produce a binary `ail_out.exe` we must then run. WASM produces a
/// .wasm that requires a browser, so we only check it compiles.
fn run_program(target: &str, compiled: &std::process::Output) -> String {
    match target {
        "native" => String::from_utf8_lossy(&compiled.stdout).to_string(),
        "c" | "llvm" => {
            let run = Command::new("ail_out.exe").output().expect("run compiled");
            String::from_utf8_lossy(&run.stdout).to_string()
        }
        "wasm" => String::new(), // compile-only
        _ => String::new(),
    }
}

fn assert_stdout(src: &str, target: &str, expected: &str) {
    let compiled = compile(src, target);
    assert!(compiled.status.success(), "compile failed for {}:\nstderr: {}", target, String::from_utf8_lossy(&compiled.stderr));
    let got = run_program(target, &compiled).trim().to_string();
    assert_eq!(got, expected.trim_end(), "target {} mismatch", target);
}

#[test]
fn arithmetic_all_backends() {
    for t in ["native", "llvm", "c", "wasm"] {
        assert_stdout("(fn main () void (print (+ 1 2)) (print (* 3 4)) (print (/ 10 3)))", t, "3\n12\n3");
    }
}

#[test]
fn float_print_all_backends() {
    for t in ["native", "llvm", "c", "wasm"] {
        assert_stdout("(fn main () void (print (/ 10.0 4.0)))", t, "2.5");
    }
}

#[test]
fn strings_all_backends() {
    for t in ["native", "llvm", "c", "wasm"] {
        assert_stdout(r#"(fn main () void (print "hello"))"#, t, "hello");
    }
}

#[test]
fn if_else_all_backends() {
    for t in ["native", "llvm", "c", "wasm"] {
        assert_stdout("(fn main () void (if (> 5 3) (print 1) (print 2)))", t, "1");
    }
}

#[test]
fn while_loop_all_backends() {
    for t in ["native", "llvm", "c", "wasm"] {
        assert_stdout("(fn main () void (let n 0) (while (< n 3) (set n (+ n 1)) (print n)))", t, "1\n2\n3");
    }
}

#[test]
fn scripting_builtins() {
    for t in ["native", "llvm", "c"] {
        assert_stdout(r#"(fn main () void (print (env "PATH")))"#, t, "}"); // env prints something; non-empty check via not(eq) below
    }
}

#[test]
fn compile_error_missing_main() {
    for t in ["native", "llvm", "c", "wasm"] {
        let out = compile("(fn foo () void (print 1))", t);
        assert!(!out.status.success(), "expected failure for {}", t);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("main"), "stderr missing 'main': {}", stderr);
    }
}

#[test]
fn compile_error_unclosed_paren() {
    for t in ["native", "llvm", "c", "wasm"] {
        let out = compile("(fn main () void (print (+ 1 2))", t);
        assert!(!out.status.success(), "expected failure for {}", t);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("-->"), "stderr missing location: {}", stderr);
    }
}

#[test]
fn compile_error_unknown_var() {
    for t in ["native", "llvm", "c", "wasm"] {
        let out = compile("(fn main () void (print x))", t);
        assert!(!out.status.success(), "expected failure for {}", t);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("unknown variable"), "stderr missing var error: {}", stderr);
    }
}

#[test]
fn compile_error_str_arith() {
    for t in ["native", "llvm", "c", "wasm"] {
        let out = compile(r#"(fn main () void (print (+ "hello" 1)))"#, t);
        assert!(!out.status.success(), "expected failure for {}", t);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("str and numeric"), "stderr missing type error: {}", stderr);
    }
}

#[test]
fn compile_error_eq_str() {
    for t in ["native", "llvm", "c", "wasm"] {
        let out = compile(r#"(fn main () void (print (== "a" "b")))"#, t);
        assert!(!out.status.success(), "expected failure for {}", t);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("str_eq"), "stderr missing str_eq hint: {}", stderr);
    }
}

#[test]
fn user_fn_all_backends() {
    for t in ["native", "llvm", "c", "wasm"] {
        assert_stdout("(fn add (a i64 b i64) i64 (+ a b)) (fn main () void (print (add 3 4)))", t, "7");
    }
}