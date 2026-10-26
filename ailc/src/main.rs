use std::collections::HashMap;
use std::env;
use std::fs;

use cranelift::frontend::Variable;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use cranelift_module::{DataDescription, Linkage};
use cranelift_module::Module as _;

// ---------- host functions ----------
#[no_mangle]
pub extern "C" fn ail_print_i64(x: i64) { println!("{}", x); }
#[no_mangle]
pub extern "C" fn ail_print_f64(x: f64) { println!("{}", x); }
#[no_mangle]
pub extern "C" fn ail_print_bool(x: i64) { println!("{}", x != 0); }
#[no_mangle]
pub extern "C" fn ail_print_str(p: *const u8) {
    if p.is_null() { return; }
    let mut len = 0usize;
    unsafe { while *p.add(len) != 0 { len += 1; } }
    let s = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(p, len)) };
    println!("{}", s);
}

#[no_mangle]
pub extern "C" fn ail_str_len(p: *const u8) -> i64 {
    if p.is_null() { return 0; }
    let mut len = 0usize;
    unsafe { while *p.add(len) != 0 { len += 1; } }
    len as i64
}

static ARRAYS: std::sync::LazyLock<Mutex<Vec<Vec<i64>>>> = std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

#[no_mangle]
pub extern "C" fn ail_array_new() -> i64 {
    let mut arrs = ARRAYS.lock().unwrap();
    arrs.push(Vec::new());
    arrs.len() as i64
}

#[no_mangle]
pub extern "C" fn ail_array_push(h: i64, v: i64) {
    if h <= 0 { return; }
    let mut arrs = ARRAYS.lock().unwrap();
    let idx = (h - 1) as usize;
    if idx < arrs.len() { arrs[idx].push(v); }
}

#[no_mangle]
pub extern "C" fn ail_array_get(h: i64, i: i64) -> i64 {
    if h <= 0 { return 0; }
    let arrs = ARRAYS.lock().unwrap();
    let idx = (h - 1) as usize;
    if idx < arrs.len() {
        let v = &arrs[idx];
        if i >= 0 && (i as usize) < v.len() { return v[i as usize]; }
    }
    0
}

#[no_mangle]
pub extern "C" fn ail_array_len(h: i64) -> i64 {
    if h <= 0 { return 0; }
    let arrs = ARRAYS.lock().unwrap();
    let idx = (h - 1) as usize;
    if idx < arrs.len() { arrs[idx].len() as i64 } else { 0 }
}

struct Mat { rows: usize, cols: usize, data: Vec<f64> }
static MATS: std::sync::LazyLock<Mutex<Vec<Option<Mat>>>> = std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

#[no_mangle]
pub extern "C" fn ail_mat_new(rows: i64, cols: i64) -> i64 {
    let r = rows.max(0) as usize;
    let c = cols.max(0) as usize;
    let mut mats = MATS.lock().unwrap();
    mats.push(Some(Mat { rows: r, cols: c, data: vec![0.0; r*c] }));
    mats.len() as i64
}

#[no_mangle]
pub extern "C" fn ail_mat_set(h: i64, r: i64, c: i64, v: f64) {
    if h <= 0 { return; }
    let mut mats = MATS.lock().unwrap();
    let idx = (h-1) as usize;
    if let Some(Some(m)) = mats.get_mut(idx) {
        let ri = r as usize; let ci = c as usize;
        if ri < m.rows && ci < m.cols { m.data[ri*m.cols+ci] = v; }
    }
}

#[no_mangle]
pub extern "C" fn ail_mat_get(h: i64, r: i64, c: i64) -> f64 {
    if h <= 0 { return 0.0; }
    let mats = MATS.lock().unwrap();
    let idx = (h-1) as usize;
    if let Some(Some(m)) = mats.get(idx) {
        let ri = r as usize; let ci = c as usize;
        if ri < m.rows && ci < m.cols { return m.data[ri*m.cols+ci]; }
    }
    0.0
}

#[no_mangle]
pub extern "C" fn ail_mat_rows(h: i64) -> i64 {
    if h <= 0 { return 0; }
    let mats = MATS.lock().unwrap();
    mats.get((h-1) as usize).and_then(|m| m.as_ref()).map(|m| m.rows as i64).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn ail_mat_cols(h: i64) -> i64 {
    if h <= 0 { return 0; }
    let mats = MATS.lock().unwrap();
    mats.get((h-1) as usize).and_then(|m| m.as_ref()).map(|m| m.cols as i64).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn ail_mat_add(a: i64, b: i64) -> i64 {
    if a <= 0 || b <= 0 { return 0; }
    let mats = MATS.lock().unwrap();
    let ma = match mats.get((a-1) as usize).and_then(|m| m.as_ref()) { Some(m)=>m, None=>return 0 };
    let mb = match mats.get((b-1) as usize).and_then(|m| m.as_ref()) { Some(m)=>m, None=>return 0 };
    if ma.rows != mb.rows || ma.cols != mb.cols { return 0; }
    let (rows, cols) = (ma.rows, ma.cols);
    let mut data = vec![0.0; ma.data.len()];
    for i in 0..data.len() { data[i] = ma.data[i] + mb.data[i]; }
    drop(mats);
    let mut mats = MATS.lock().unwrap();
    mats.push(Some(Mat { rows, cols, data }));
    mats.len() as i64
}

#[no_mangle]
pub extern "C" fn ail_mat_mul(a: i64, b: i64) -> i64 {
    if a <= 0 || b <= 0 { return 0; }
    let mats = MATS.lock().unwrap();
    let ma = match mats.get((a-1) as usize).and_then(|m| m.as_ref()) { Some(m)=>m, None=>return 0 };
    let mb = match mats.get((b-1) as usize).and_then(|m| m.as_ref()) { Some(m)=>m, None=>return 0 };
    if ma.cols != mb.rows { return 0; }
    let (m, k, n) = (ma.rows, ma.cols, mb.cols);
    let mut data = vec![0.0; m*n];
    // rayon parallel over rows
    use rayon::prelude::*;
    data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += ma.data[i*k+t] * mb.data[t*n+j];
            }
            row[j] = s;
        }
    });
    let rows = m; let cols = n;
    drop(mats);
    let mut mats = MATS.lock().unwrap();
    mats.push(Some(Mat { rows, cols, data }));
    mats.len() as i64
}

#[no_mangle]
pub extern "C" fn ail_mat_relu(h: i64) -> i64 {
    if h <= 0 { return 0; }
    let mut mats = MATS.lock().unwrap();
    let idx = (h-1) as usize;
    if let Some(Some(m)) = mats.get_mut(idx) {
        for v in m.data.iter_mut() { if *v < 0.0 { *v = 0.0; } }
    }
    h
}

struct Gfx { window: minifb::Window, buffer: Vec<u32>, w: usize, h: usize }
thread_local! {
    static GFXS: std::cell::RefCell<Vec<Option<Gfx>>> = std::cell::RefCell::new(Vec::new());
}

#[no_mangle]
pub extern "C" fn ail_gfx_open(title: *const u8, w: i64, h: i64) -> i64 {
    let mut t = String::new();
    if !title.is_null() { unsafe { let mut i=0; while *title.add(i)!=0 { t.push(*title.add(i) as char); i+=1; } } }
    if t.is_empty() { t = "Ail".into(); }
    let wi = w.max(1) as usize;
    let hi = h.max(1) as usize;
    let mut opts = minifb::WindowOptions::default();
    opts.resize = false;
    let window = match minifb::Window::new(&t, wi, hi, opts) { Ok(w)=>w, Err(_)=>return 0 };
    GFXS.with(|cell| {
        let mut gfxs = cell.borrow_mut();
        gfxs.push(Some(Gfx { window, buffer: vec![0; wi*hi], w: wi, h: hi }));
        gfxs.len() as i64
    })
}

#[no_mangle]
pub extern "C" fn ail_gfx_pixel(h: i64, x: i64, y: i64, color: i64) {
    if h <= 0 { return; }
    GFXS.with(|cell| {
        let mut gfxs = cell.borrow_mut();
        let idx = (h-1) as usize;
        if let Some(Some(g)) = gfxs.get_mut(idx) {
            let xi = x as usize; let yi = y as usize;
            if xi < g.w && yi < g.h { g.buffer[yi*g.w+xi] = color as u32; }
        }
    });
}

#[no_mangle]
pub extern "C" fn ail_gfx_show(h: i64) {
    if h <= 0 { return; }
    GFXS.with(|cell| {
        let mut gfxs = cell.borrow_mut();
        let idx = (h-1) as usize;
        if let Some(Some(g)) = gfxs.get_mut(idx) {
            let _ = g.window.update_with_buffer(&g.buffer, g.w, g.h);
        }
    });
}

#[no_mangle]
pub extern "C" fn ail_gfx_poll(h: i64) -> i64 {
    if h <= 0 { return 0; }
    GFXS.with(|cell| {
        let mut gfxs = cell.borrow_mut();
        let idx = (h-1) as usize;
        if let Some(Some(g)) = gfxs.get_mut(idx) {
            if !g.window.is_open() { return 0; }
            let _ = g.window.update_with_buffer(&g.buffer, g.w, g.h);
            if g.window.is_open() { 1 } else { 0 }
        } else { 0 }
    })
}

#[no_mangle]
pub extern "C" fn ail_gfx_close(h: i64) {
    if h <= 0 { return; }
    GFXS.with(|cell| {
        let mut gfxs = cell.borrow_mut();
        let idx = (h-1) as usize;
        if idx < gfxs.len() { gfxs[idx] = None; }
    });
}

#[no_mangle]
pub extern "C" fn ail_js_eval(_p: *const u8) -> i64 { 0 }

#[no_mangle]
pub extern "C" fn ail_dom_set_inner_html(_a: *const u8, _b: *const u8) -> i64 { 0 }

#[no_mangle]
pub extern "C" fn ail_str_concat(a: *const u8, b: *const u8) -> *const u8 {
    let mut sa = Vec::new();
    if !a.is_null() { unsafe { let mut i=0; while *a.add(i)!=0 { sa.push(*a.add(i)); i+=1; } } }
    if !b.is_null() { unsafe { let mut i=0; while *b.add(i)!=0 { sa.push(*b.add(i)); i+=1; } } }
    sa.push(0);
    let boxed = sa.into_boxed_slice();
    Box::into_raw(boxed) as *const u8
}

// ---------- AST ----------
#[derive(Debug, Clone)]
enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Var(String),
    Block(Vec<Expr>),
    Let(String, Box<Expr>),
    Set(String, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    While(Box<Expr>, Vec<Expr>),
    Call(String, Vec<Expr>),
    Fn { name: String, params: Vec<(String, String)>, ret: String, body: Vec<Expr> },
    Program(Vec<Expr>),
}

// ---------- tokenizer ----------
#[derive(Debug, Clone, PartialEq)]
enum Tok { LParen, RParen, Atom(String) }

fn tokenize(src: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\n' | '\r' => { chars.next(); }
            '(' => { chars.next(); out.push(Tok::LParen); }
            ')' => { chars.next(); out.push(Tok::RParen); }
            '"' => {
                chars.next();
                let mut s = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == '"' { chars.next(); break; }
                    if ch == '\\' {
                        chars.next();
                        if let Some(&e) = chars.peek() {
                            chars.next();
                            match e { 'n'=>'\n', 't'=>'\t', '"' =>'"', '\\'=>'\\', _=>e }.to_string().chars().for_each(|x| s.push(x));
                        }
                    } else { chars.next(); s.push(ch); }
                }
                out.push(Tok::Atom(format!("\"{}\"", s.replace('\\',"\\\\").replace('"',"\\\""))));
            }
            _ => {
                let mut s = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_whitespace() || ch=='(' || ch==')' || ch=='"' { break; }
                    s.push(ch); chars.next();
                }
                if !s.is_empty() { out.push(Tok::Atom(s)); }
            }
        }
    }
    out
}

// ---------- parser ----------
struct Parser { toks: Vec<Tok>, pos: usize }
impl Parser {
    fn new(toks: Vec<Tok>) -> Self { Self { toks, pos: 0 } }
    fn parse_program(&mut self) -> Expr {
        let mut fns = Vec::new();
        while self.pos < self.toks.len() {
            fns.push(self.parse_expr());
        }
        Expr::Program(fns)
    }
    fn parse_expr(&mut self) -> Expr {
        if self.pos >= self.toks.len() { panic!("unexpected eof"); }
        match &self.toks[self.pos] {
            Tok::LParen => {
                self.pos += 1;
                let head = match &self.toks[self.pos] {
                    Tok::Atom(a) => { let a = a.clone(); self.pos += 1; a }
                    _ => panic!("expected head atom"),
                };
                match head.as_str() {
                    "fn" => {
                        let name = atom_str(self);
                        expect_lp(self);
                        let mut params = Vec::new();
                        while self.pos < self.toks.len() && self.toks[self.pos] != Tok::RParen {
                            let pname = atom_str(self);
                            let ptype = atom_str(self);
                            params.push((pname, ptype));
                        }
                        expect_rp(self);
                        let ret = atom_str(self);
                        let mut body = Vec::new();
                        while self.pos < self.toks.len() && self.toks[self.pos] != Tok::RParen {
                            body.push(self.parse_expr());
                        }
                        expect_rp(self);
                        Expr::Fn { name, params, ret, body }
                    }
                    "let" => {
                        let name = atom_str(self);
                        let val = self.parse_expr();
                        expect_rp(self);
                        Expr::Let(name, Box::new(val))
                    }
                    "set" => {
                        let name = atom_str(self);
                        let val = self.parse_expr();
                        expect_rp(self);
                        Expr::Set(name, Box::new(val))
                    }
                    "if" => {
                        let c = self.parse_expr();
                        let t = self.parse_expr();
                        let e = self.parse_expr();
                        expect_rp(self);
                        Expr::If(Box::new(c), Box::new(t), Box::new(e))
                    }
                    "while" => {
                        let c = self.parse_expr();
                        let mut body = Vec::new();
                        while self.pos < self.toks.len() && self.toks[self.pos] != Tok::RParen {
                            body.push(self.parse_expr());
                        }
                        expect_rp(self);
                        Expr::While(Box::new(c), body)
                    }
                    "block" => {
                        let mut v = Vec::new();
                        while self.pos < self.toks.len() && self.toks[self.pos] != Tok::RParen {
                            v.push(self.parse_expr());
                        }
                        expect_rp(self);
                        Expr::Block(v)
                    }
                    _ => {
                        let mut args = Vec::new();
                        while self.pos < self.toks.len() && self.toks[self.pos] != Tok::RParen {
                            args.push(self.parse_expr());
                        }
                        expect_rp(self);
                        Expr::Call(head, args)
                    }
                }
            }
            Tok::Atom(a) => {
                self.pos += 1;
                if let Ok(n) = a.parse::<i64>() { return Expr::Int(n); }
                if let Ok(f) = a.parse::<f64>() { return Expr::Float(f); }
                if a == "true" { return Expr::Bool(true); }
                if a == "false" { return Expr::Bool(false); }
                if a.starts_with('"') {
                    // unquote
                    let inner = &a[1..a.len()-1];
                    return Expr::Str(inner.to_string());
                }
                Expr::Var(a.clone())
            }
            Tok::RParen => { self.pos += 1; panic!("unexpected )"); }
        }
    }
}

fn atom_str(p: &mut Parser) -> String {
    match &p.toks[p.pos] {
        Tok::Atom(a) => { let a = a.clone(); p.pos += 1; a }
        _ => panic!("expected atom"),
    }
}
fn expect_lp(p: &mut Parser) {
    assert!(p.toks[p.pos] == Tok::LParen, "expected (");
    p.pos += 1;
}
fn expect_rp(p: &mut Parser) {
    assert!(p.toks[p.pos] == Tok::RParen, "expected )");
    p.pos += 1;
}

// ---------- type helpers ----------
fn type_of(e: &Expr) -> String {
    match e {
        Expr::Int(_) => "i64".into(),
        Expr::Float(_) => "f64".into(),
        Expr::Bool(_) => "i64".into(),
        Expr::Str(_) => "str".into(),
        Expr::Var(v) => VAR_TYPES.lock().unwrap().get(v).cloned().unwrap_or_else(|| "i64".into()),
        Expr::Let(_, v) => type_of(v),
        Expr::Set(_, v) => type_of(v),
        Expr::If(_, t, _) => type_of(t),
        Expr::Block(v) => v.last().map(type_of).unwrap_or_else(|| "i64".into()),
        Expr::While(_, _) => "i64".into(),
        Expr::Call(name, args) => {
            match name.as_str() {
                "+" | "-" | "*" | "/" | "%" => {
                    if args.iter().any(|a| type_of(a) == "str") { "str".into() }
                    else if args.iter().any(|a| type_of(a) == "f64") { "f64".into() }
                    else { "i64".into() }
                }
                "<" | ">" | "<=" | ">=" | "==" | "!=" | "and" | "or" | "not" => "i64".into(),
                "print" => "i64".into(),
                "array" => "array".into(),
                "get" => "i64".into(),
                "push" => "i64".into(),
                "mat_new" => "mat".into(),
                "mat_get" => "f64".into(),
                "mat_rows" => "i64".into(),
                "mat_cols" => "i64".into(),
                "mat_add" => "mat".into(),
                "mat_mul" => "mat".into(),
                "mat_relu" => "mat".into(),
                "gfx_open" => "gfx".into(),
                "gfx_poll" => "i64".into(),
                "js_eval" => "i64".into(),
                "dom_set_inner_html" => "i64".into(),
                _ => "i64".into(), // user fn assumed i64 for v0
            }
        }
        Expr::Fn { ret, .. } => ret.clone(),
        Expr::Program(_) => "void".into(),
    }
}

fn cl_type(t: &str) -> Type {
    match t { "f64" => types::F64, _ => types::I64 }
}

// ---------- compiler ----------
struct Ctx {
    module: JITModule,
    funcs: HashMap<String, (cranelift_module::FuncId, Vec<String>, String)>, // name -> (id, param types, ret)
    host: HashMap<String, cranelift_module::FuncId>,
    strs: HashMap<String, cranelift_module::DataId>,
    str_counter: usize,
}

impl Ctx {
    fn new() -> Self {
        let mut builder = JITBuilder::with_flags(&[("opt_level", "speed"), ("enable_verifier", "false")], Box::new(cranelift_module::default_libcall_names())).unwrap();
        builder.symbol("ail_print_i64", ail_print_i64 as *const u8);
        builder.symbol("ail_print_f64", ail_print_f64 as *const u8);
        builder.symbol("ail_print_bool", ail_print_bool as *const u8);
        builder.symbol("ail_print_str", ail_print_str as *const u8);
        builder.symbol("ail_str_len", ail_str_len as *const u8);
        builder.symbol("ail_str_concat", ail_str_concat as *const u8);
        builder.symbol("ail_array_new", ail_array_new as *const u8);
        builder.symbol("ail_array_push", ail_array_push as *const u8);
        builder.symbol("ail_array_get", ail_array_get as *const u8);
        builder.symbol("ail_array_len", ail_array_len as *const u8);
        builder.symbol("ail_mat_new", ail_mat_new as *const u8);
        builder.symbol("ail_mat_set", ail_mat_set as *const u8);
        builder.symbol("ail_mat_get", ail_mat_get as *const u8);
        builder.symbol("ail_mat_rows", ail_mat_rows as *const u8);
        builder.symbol("ail_mat_cols", ail_mat_cols as *const u8);
        builder.symbol("ail_mat_add", ail_mat_add as *const u8);
        builder.symbol("ail_mat_mul", ail_mat_mul as *const u8);
        builder.symbol("ail_mat_relu", ail_mat_relu as *const u8);
        builder.symbol("ail_gfx_open", ail_gfx_open as *const u8);
        builder.symbol("ail_gfx_pixel", ail_gfx_pixel as *const u8);
        builder.symbol("ail_gfx_show", ail_gfx_show as *const u8);
        builder.symbol("ail_gfx_poll", ail_gfx_poll as *const u8);
        builder.symbol("ail_gfx_close", ail_gfx_close as *const u8);
        builder.symbol("ail_js_eval", ail_js_eval as *const u8);
        builder.symbol("ail_dom_set_inner_html", ail_dom_set_inner_html as *const u8);
        let module = JITModule::new(builder);
        let mut ctx = Ctx { module, funcs: HashMap::new(), host: HashMap::new(), strs: HashMap::new(), str_counter: 0 };
        // declare host imports
        for (name, params, rets) in [
            ("ail_print_i64", vec![types::I64], vec![]),
            ("ail_print_f64", vec![types::F64], vec![]),
            ("ail_print_bool", vec![types::I64], vec![]),
            ("ail_print_str", vec![types::I64], vec![]),
            ("ail_str_len", vec![types::I64], vec![types::I64]),
            ("ail_str_concat", vec![types::I64, types::I64], vec![types::I64]),
            ("ail_array_new", vec![], vec![types::I64]),
            ("ail_array_push", vec![types::I64, types::I64], vec![]),
            ("ail_array_get", vec![types::I64, types::I64], vec![types::I64]),
            ("ail_array_len", vec![types::I64], vec![types::I64]),
            ("ail_mat_new", vec![types::I64, types::I64], vec![types::I64]),
            ("ail_mat_set", vec![types::I64, types::I64, types::I64, types::F64], vec![]),
            ("ail_mat_get", vec![types::I64, types::I64, types::I64], vec![types::F64]),
            ("ail_mat_rows", vec![types::I64], vec![types::I64]),
            ("ail_mat_cols", vec![types::I64], vec![types::I64]),
            ("ail_mat_add", vec![types::I64, types::I64], vec![types::I64]),
            ("ail_mat_mul", vec![types::I64, types::I64], vec![types::I64]),
            ("ail_mat_relu", vec![types::I64], vec![types::I64]),
            ("ail_gfx_open", vec![types::I64, types::I64, types::I64], vec![types::I64]),
            ("ail_gfx_pixel", vec![types::I64, types::I64, types::I64, types::I64], vec![]),
            ("ail_gfx_show", vec![types::I64], vec![]),
            ("ail_gfx_poll", vec![types::I64], vec![types::I64]),
            ("ail_gfx_close", vec![types::I64], vec![]),
            ("ail_js_eval", vec![types::I64], vec![types::I64]),
            ("ail_dom_set_inner_html", vec![types::I64, types::I64], vec![types::I64]),
        ] {
            let mut sig = ctx.module.make_signature();
            for p in params { sig.params.push(AbiParam::new(p)); }
            for r in rets { sig.returns.push(AbiParam::new(r)); }
            let fid = ctx.module.declare_function(name, Linkage::Import, &sig).unwrap();
            ctx.host.insert(name.to_string(), fid);
        }
        ctx
    }

    fn get_string(&mut self, s: &str) -> cranelift_module::DataId {
        if let Some(d) = self.strs.get(s) { return *d; }
        let name = format!("str{}", self.str_counter);
        self.str_counter += 1;
        let did = self.module.declare_data(&name, Linkage::Local, true, false).unwrap();
        let mut desc = DataDescription::new();
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        desc.define(bytes.into_boxed_slice());
        self.module.define_data(did, &desc).unwrap();
        self.strs.insert(s.to_string(), did);
        did
    }

    fn declare_funcs(&mut self, prog: &Expr) {
        if let Expr::Program(fns) = prog {
            for f in fns {
                if let Expr::Fn { name, params, ret, .. } = f {
                    let mut sig = self.module.make_signature();
                    for (_, t) in params {
                        sig.params.push(AbiParam::new(cl_type(t)));
                    }
                    if ret != "void" {
                        sig.returns.push(AbiParam::new(cl_type(ret)));
                    }
                    let fid = self.module.declare_function(name, Linkage::Export, &sig).unwrap();
                    let ptypes: Vec<String> = params.iter().map(|(_, t)| t.clone()).collect();
                    self.funcs.insert(name.clone(), (fid, ptypes, ret.clone()));
                }
            }
        }
    }

    fn compile_fn(&mut self, f: &Expr) {
        if let Expr::Fn { name, params, ret, body } = f {
            let (fid, _, _) = self.funcs[name].clone();
            let mut ctx = self.module.make_context();
            // signature
            let mut sig = self.module.make_signature();
            for (_, t) in params { sig.params.push(AbiParam::new(cl_type(t))); }
            if ret != "void" { sig.returns.push(AbiParam::new(cl_type(ret))); }
            ctx.func.signature = sig;

            let mut bctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut bctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let mut vars: HashMap<String, Variable> = HashMap::new();
            VAR_TYPES.lock().unwrap().clear();
            for (pname, ptype) in params {
                VAR_TYPES.lock().unwrap().insert(pname.clone(), ptype.clone());
            }
            for (i, (pname, ptype)) in params.iter().enumerate() {
                let v = new_var();
                builder.declare_var(v, cl_type(ptype));
                let pv = builder.block_params(entry)[i];
                builder.def_var(v, pv);
                vars.insert(pname.clone(), v);
            }

            // compile body
            let mut last_val: Option<Value> = None;
            for (i, e) in body.iter().enumerate() {
                let v = self.gen_expr(&mut builder, &mut vars, e);
                if i == body.len().saturating_sub(1) {
                    last_val = v;
                }
            }

            if ret != "void" {
                let v = last_val.unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                builder.ins().return_(&[v]);
            } else {
                builder.ins().return_(&[]);
            }
            builder.finalize();

            self.module.define_function(fid, &mut ctx).unwrap();
        }
    }

    fn gen_expr(&mut self, builder: &mut FunctionBuilder, vars: &mut HashMap<String, Variable>, e: &Expr) -> Option<Value> {
        match e {
            Expr::Int(n) => Some(builder.ins().iconst(types::I64, *n)),
            Expr::Float(f) => Some(builder.ins().f64const(*f)),
            Expr::Bool(b) => Some(builder.ins().iconst(types::I64, if *b { 1 } else { 0 })),
            Expr::Str(s) => {
                let did = self.get_string(s);
                let gv = self.module.declare_data_in_func(did, builder.func);
                Some(builder.ins().global_value(types::I64, gv))
            }
            Expr::Var(v) => {
                if let Some(var) = vars.get(v) {
                    Some(builder.use_var(*var))
                } else {
                    Some(builder.ins().iconst(types::I64, 0))
                }
            }
            Expr::Let(name, val) => {
                let vval = self.gen_expr(builder, vars, val).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                let ty = type_of(val);
                let variable = new_var();
                builder.declare_var(variable, cl_type(&ty));
                builder.def_var(variable, vval);
                vars.insert(name.clone(), variable);
                VAR_TYPES.lock().unwrap().insert(name.clone(), ty);
                None
            }
            Expr::Set(name, val) => {
                let vval = self.gen_expr(builder, vars, val).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                if let Some(var) = vars.get(name) {
                    builder.def_var(*var, vval);
                }
                None
            }
            Expr::Block(v) => {
                let mut last = None;
                for e in v {
                    last = self.gen_expr(builder, vars, e);
                }
                last
            }
            Expr::If(c, t, e) => {
                let cond_val = self.gen_expr(builder, vars, c).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                let cond_i8 = builder.ins().icmp_imm(IntCC::NotEqual, cond_val, 0);
                let then_b = builder.create_block();
                let else_b = builder.create_block();
                let merge_b = builder.create_block();
                let rty = type_of(t);
                builder.append_block_param(merge_b, cl_type(&rty));
                builder.ins().brif(cond_i8, then_b, &[], else_b, &[]);

                builder.switch_to_block(then_b);
                builder.seal_block(then_b);
                let then_val = self.gen_expr(builder, vars, t).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                builder.ins().jump(merge_b, &[then_val]);

                builder.switch_to_block(else_b);
                builder.seal_block(else_b);
                let else_val = self.gen_expr(builder, vars, e).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                builder.ins().jump(merge_b, &[else_val]);

                builder.switch_to_block(merge_b);
                builder.seal_block(merge_b);
                Some(builder.block_params(merge_b)[0])
            }
            Expr::While(c, body) => {
                let header = builder.create_block();
                let body_b = builder.create_block();
                let exit_b = builder.create_block();
                builder.ins().jump(header, &[]);
                builder.switch_to_block(header);
                let cond_val = self.gen_expr(builder, vars, c).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                let cond_i8 = builder.ins().icmp_imm(IntCC::NotEqual, cond_val, 0);
                builder.ins().brif(cond_i8, body_b, &[], exit_b, &[]);
                builder.switch_to_block(body_b);
                builder.seal_block(body_b);
                for e in body {
                    self.gen_expr(builder, vars, e);
                }
                builder.ins().jump(header, &[]);
                builder.seal_block(header);
                builder.switch_to_block(exit_b);
                builder.seal_block(exit_b);
                Some(builder.ins().iconst(types::I64, 0))
            }
            Expr::Call(name, args) => {
                if name == "+" && args.len() == 2 && (type_of(&args[0]) == "str" || type_of(&args[1]) == "str") {
                    let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let b = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_str_concat"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[a, b]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "len" && !args.is_empty() {
                    let arg = args[0].clone();
                    let v = self.gen_expr(builder, vars, &arg).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    if type_of(&arg) == "str" {
                        let fid = self.host["ail_str_len"];
                        let fref = self.module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[v]);
                        let res = builder.inst_results(call);
                        return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                    }
                    if type_of(&arg) == "array" {
                        let fid = self.host["ail_array_len"];
                        let fref = self.module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[v]);
                        let res = builder.inst_results(call);
                        return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                    }
                    return Some(v);
                }
                if name == "array" {
                    let fid = self.host["ail_array_new"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "push" && args.len() == 2 {
                    let arr = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let val = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_array_push"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    builder.ins().call(fref, &[arr, val]);
                    return Some(builder.ins().iconst(types::I64, 0));
                }
                // matrix builtins
                if name == "mat_new" && args.len() == 2 {
                    let r = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let c = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_mat_new"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[r, c]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "mat_set" && args.len() == 4 {
                    let m = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let r = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let c = self.gen_expr(builder, vars, &args[2]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let v = self.gen_expr(builder, vars, &args[3]).unwrap_or_else(|| builder.ins().f64const(0.0));
                    let fid = self.host["ail_mat_set"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    builder.ins().call(fref, &[m, r, c, v]);
                    return Some(builder.ins().iconst(types::I64, 0));
                }
                if name == "mat_get" && args.len() == 3 {
                    let m = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let r = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let c = self.gen_expr(builder, vars, &args[2]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_mat_get"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[m, r, c]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().f64const(0.0) } else { res[0] });
                }
                if name == "mat_rows" && args.len() == 1 {
                    let m = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_mat_rows"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[m]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "mat_cols" && args.len() == 1 {
                    let m = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_mat_cols"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[m]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "mat_add" && args.len() == 2 {
                    let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let b = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_mat_add"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[a, b]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "mat_mul" && args.len() == 2 {
                    let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let b = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_mat_mul"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[a, b]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "mat_relu" && args.len() == 1 {
                    let m = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_mat_relu"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[m]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                // gfx builtins
                if name == "gfx_open" && args.len() == 3 {
                    let title = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let w = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let h = self.gen_expr(builder, vars, &args[2]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_gfx_open"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[title, w, h]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "gfx_pixel" && args.len() == 4 {
                    let h = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let x = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let y = self.gen_expr(builder, vars, &args[2]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let color = self.gen_expr(builder, vars, &args[3]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_gfx_pixel"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    builder.ins().call(fref, &[h, x, y, color]);
                    return Some(builder.ins().iconst(types::I64, 0));
                }
                if name == "gfx_show" && args.len() == 1 {
                    let h = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_gfx_show"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    builder.ins().call(fref, &[h]);
                    return Some(builder.ins().iconst(types::I64, 0));
                }
                if name == "gfx_poll" && args.len() == 1 {
                    let h = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_gfx_poll"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[h]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "gfx_close" && args.len() == 1 {
                    let h = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_gfx_close"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    builder.ins().call(fref, &[h]);
                    return Some(builder.ins().iconst(types::I64, 0));
                }
                if name == "get" && args.len() == 2 {
                    let arr = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let idx = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_array_get"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[arr, idx]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "js_eval" && args.len() == 1 {
                    let code = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_js_eval"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[code]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                if name == "dom_set_inner_html" && args.len() == 2 {
                    let sel = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let html = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                    let fid = self.host["ail_dom_set_inner_html"];
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[sel, html]);
                    let res = builder.inst_results(call);
                    return Some(if res.is_empty() { builder.ins().iconst(types::I64, 0) } else { res[0] });
                }
                match name.as_str() {
                    "+" | "-" | "*" | "/" | "%" => {
                        if args.len() == 1 && name == "-" {
                            let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                            return Some(builder.ins().ineg(a));
                        }
                        let is_float = type_of(&args[0]) == "f64" || type_of(&args[1]) == "f64";
                        let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        let b = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        let res = if is_float {
                            match name.as_str() {
                                "+" => builder.ins().fadd(a, b),
                                "-" => builder.ins().fsub(a, b),
                                "*" => builder.ins().fmul(a, b),
                                "/" => builder.ins().fdiv(a, b),
                                _ => a,
                            }
                        } else {
                            match name.as_str() {
                                "+" => builder.ins().iadd(a, b),
                                "-" => builder.ins().isub(a, b),
                                "*" => builder.ins().imul(a, b),
                                "/" => builder.ins().sdiv(a, b),
                                "%" => builder.ins().srem(a, b),
                                _ => a,
                            }
                        };
                        // float handling: if either f64, use f ops instead (recompute)
                        Some(res)
                    }
                    "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                        let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        let b = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        let cc = match name.as_str() {
                            "<" => IntCC::SignedLessThan,
                            ">" => IntCC::SignedGreaterThan,
                            "<=" => IntCC::SignedLessThanOrEqual,
                            ">=" => IntCC::SignedGreaterThanOrEqual,
                            "==" => IntCC::Equal,
                            "!=" => IntCC::NotEqual,
                            _ => IntCC::Equal,
                        };
                        let cmp = builder.ins().icmp(cc, a, b);
                        Some(builder.ins().uextend(types::I64, cmp))
                    }
                    "and" => {
                        let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        let b = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        Some(builder.ins().band(a, b))
                    }
                    "or" => {
                        let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        let b = self.gen_expr(builder, vars, &args[1]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        Some(builder.ins().bor(a, b))
                    }
                    "not" => {
                        let a = self.gen_expr(builder, vars, &args[0]).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        Some(builder.ins().icmp_imm(IntCC::Equal, a, 0).pipe(|v| builder.ins().uextend(types::I64, v)))
                    }
                    "print" => {
                        let arg = args.get(0).cloned().unwrap_or(Expr::Int(0));
                        let t = type_of(&arg);
                        let v = self.gen_expr(builder, vars, &arg).unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                        let fname = match t.as_str() {
                            "f64" => "ail_print_f64",
                            "str" => "ail_print_str",
                            "bool" => "ail_print_bool",
                            _ => "ail_print_i64",
                        };
                        let fid = self.host[fname];
                        let fref = self.module.declare_func_in_func(fid, builder.func);
                        if t == "f64" {
                            builder.ins().call(fref, &[v]);
                        } else {
                            builder.ins().call(fref, &[v]);
                        }
                        Some(builder.ins().iconst(types::I64, 0))
                    }
                    _ => {
                        // user function call
                        if let Some((fid, _ptypes, ret)) = self.funcs.get(name).cloned() {
                            let mut argvals = Vec::new();
                            for a in args {
                                argvals.push(self.gen_expr(builder, vars, a).unwrap_or_else(|| builder.ins().iconst(types::I64, 0)));
                            }
                            let fref = self.module.declare_func_in_func(fid, builder.func);
                            let call = builder.ins().call(fref, &argvals);
                            if ret == "void" {
                                Some(builder.ins().iconst(types::I64, 0))
                            } else {
                                let res = builder.inst_results(call);
                                if res.is_empty() { Some(builder.ins().iconst(types::I64, 0)) } else { Some(res[0]) }
                            }
                        } else {
                            Some(builder.ins().iconst(types::I64, 0))
                        }
                    }
                }
            }
            _ => Some(builder.ins().iconst(types::I64, 0)),
        }
    }
}

trait Pipe<T> { fn pipe<F: FnOnce(T)->T>(self, f: F) -> T; }
impl<T> Pipe<T> for T { fn pipe<F: FnOnce(T)->T>(self, f: F) -> T { f(self) } }

static VAR_COUNTER: AtomicUsize = AtomicUsize::new(0);
static VAR_TYPES: std::sync::LazyLock<Mutex<std::collections::HashMap<String, String>>> = std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));
fn new_var() -> Variable { Variable::new(VAR_COUNTER.fetch_add(1, Ordering::Relaxed)) }

// ---------- wasm backend ----------
use wasm_encoder::{BlockType, CodeSection, DataSection, ExportKind, ExportSection, Function, FunctionSection, ImportSection, Instruction, MemArg, MemorySection, MemoryType, Module, TypeSection, ValType};

fn cl_vt(t: &str) -> ValType {
    if t == "f64" { ValType::F64 } else { ValType::I64 }
}

fn collect_strings(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Str(s) => { if !out.contains(s) { out.push(s.clone()); } }
        Expr::Block(v) | Expr::Program(v) => { for x in v { collect_strings(x, out); } }
        Expr::Fn { body, .. } => { for x in body { collect_strings(x, out); } }
        Expr::Let(_, v) | Expr::Set(_, v) => collect_strings(v, out),
        Expr::If(c,t,e) => { collect_strings(c,out); collect_strings(t,out); collect_strings(e,out); }
        Expr::While(c,body) => { collect_strings(c,out); for x in body { collect_strings(x,out); } }
        Expr::Call(_, args) => { for a in args { collect_strings(a,out); } }
        _ => {}
    }
}

fn scan_lets(e: &Expr, var_map: &mut std::collections::HashMap<String,u32>, local_types: &mut Vec<ValType>) {
    match e {
        Expr::Let(name, val) => {
            if !var_map.contains_key(name) {
                let ty = type_of(val);
                let vt = if ty == "f64" { ValType::F64 } else { ValType::I64 };
                let idx = local_types.len() as u32; // offset added later by caller? we store relative index; final index = params.len() + idx
                // We'll store placeholder; caller adjusts? Instead store absolute later. For now store params.len()+local_types.len()
                var_map.insert(name.clone(), 0); // placeholder, fix below
                // Actually we need params count. We'll handle in compile step; here just record type.
                local_types.push(vt);
                // fix index: remove placeholder, set after
                var_map.insert(name.clone(), (local_types.len() as u32)); // temporary; corrected in compile
            }
            scan_lets(val, var_map, local_types);
        }
        Expr::Block(v) | Expr::Program(v) => { for x in v { scan_lets(x, var_map, local_types); } }
        Expr::Fn { body, .. } => { for x in body { scan_lets(x, var_map, local_types); } }
        Expr::If(c,t,e) => { scan_lets(c,var_map,local_types); scan_lets(t,var_map,local_types); scan_lets(e,var_map,local_types); }
        Expr::While(c,body) => { scan_lets(c,var_map,local_types); for x in body { scan_lets(x,var_map,local_types); } }
        Expr::Call(_, args) => { for a in args { scan_lets(a,var_map,local_types); } }
        Expr::Set(_, v) => scan_lets(v, var_map, local_types),
        _ => {}
    }
}

fn gen_wasm(f: &mut Function, e: &Expr, var_map: &std::collections::HashMap<String,u32>, str_offsets: &std::collections::HashMap<String,i32>, func_idx: &std::collections::HashMap<String,u32>) {
    match e {
        Expr::Int(n) => { f.instruction(&Instruction::I64Const(*n)); }
        Expr::Float(x) => { f.instruction(&Instruction::F64Const(*x)); }
        Expr::Bool(b) => { f.instruction(&Instruction::I64Const(if *b {1} else {0})); }
        Expr::Str(s) => {
            let off = *str_offsets.get(s).unwrap_or(&0);
            f.instruction(&Instruction::I64Const(off as i64));
        }
        Expr::Var(v) => {
            let idx = var_map.get(v).copied().unwrap_or(0);
            f.instruction(&Instruction::LocalGet(idx));
        }
        Expr::Let(name, val) => {
            gen_wasm(f, val, var_map, str_offsets, func_idx);
            if let Some(idx) = var_map.get(name) {
                f.instruction(&Instruction::LocalSet(*idx));
            }
            f.instruction(&Instruction::I64Const(0));
        }
        Expr::Set(name, val) => {
            gen_wasm(f, val, var_map, str_offsets, func_idx);
            if let Some(idx) = var_map.get(name) {
                f.instruction(&Instruction::LocalSet(*idx));
            }
            f.instruction(&Instruction::I64Const(0));
        }
        Expr::Block(v) => {
            for (i, e) in v.iter().enumerate() {
                gen_wasm(f, e, var_map, str_offsets, func_idx);
                if i < v.len().saturating_sub(1) {
                    f.instruction(&Instruction::Drop);
                }
            }
            if v.is_empty() { f.instruction(&Instruction::I64Const(0)); }
        }
        Expr::If(c, t, e) => {
            gen_wasm(f, c, var_map, str_offsets, func_idx);
            f.instruction(&Instruction::I32WrapI64);
            let rty = type_of(t);
            let bt = if rty == "f64" { BlockType::Result(ValType::F64) } else { BlockType::Result(ValType::I64) };
            f.instruction(&Instruction::If(bt));
            gen_wasm(f, t, var_map, str_offsets, func_idx);
            f.instruction(&Instruction::Else);
            gen_wasm(f, e, var_map, str_offsets, func_idx);
            f.instruction(&Instruction::End);
        }
        Expr::While(c, body) => {
            f.instruction(&Instruction::Block(BlockType::Empty));
            f.instruction(&Instruction::Loop(BlockType::Empty));
            gen_wasm(f, c, var_map, str_offsets, func_idx);
            f.instruction(&Instruction::I32WrapI64);
            f.instruction(&Instruction::I32Eqz);
            f.instruction(&Instruction::BrIf(1));
            for e in body {
                gen_wasm(f, e, var_map, str_offsets, func_idx);
                f.instruction(&Instruction::Drop);
            }
            f.instruction(&Instruction::Br(0));
            f.instruction(&Instruction::End);
            f.instruction(&Instruction::End);
            f.instruction(&Instruction::I64Const(0));
        }
        Expr::Call(name, args) => {
            // str concat
            if name == "+" && args.len() == 2 && (type_of(&args[0]) == "str" || type_of(&args[1]) == "str") {
                gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                gen_wasm(f, &args[1], var_map, str_offsets, func_idx);
                f.instruction(&Instruction::Call(4)); // str_concat
                // fix below
                return;
            }

            if name == "js_eval" && args.len() == 1 {
                gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                f.instruction(&Instruction::Call(5));
                return;
            }
            if name == "dom_set_inner_html" && args.len() == 2 {
                gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                gen_wasm(f, &args[1], var_map, str_offsets, func_idx);
                f.instruction(&Instruction::Call(6));
                return;
            }
            // arithmetic
            match name.as_str() {
                "+" | "-" | "*" | "/" => {
                    let isf = type_of(&args[0]) == "f64" || type_of(&args[1]) == "f64";
                    if isf {
                        gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                        if type_of(&args[0]) != "f64" { f.instruction(&Instruction::F64ConvertI64S); }
                        gen_wasm(f, &args[1], var_map, str_offsets, func_idx);
                        if type_of(&args[1]) != "f64" { f.instruction(&Instruction::F64ConvertI64S); }
                        let instr = match name.as_str() { "+" => Instruction::F64Add, "-" => Instruction::F64Sub, "*" => Instruction::F64Mul, "/" => Instruction::F64Div, _ => Instruction::F64Add };
                        f.instruction(&instr);
                    } else {
                        gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                        gen_wasm(f, &args[1], var_map, str_offsets, func_idx);
                        let instr = match name.as_str() { "+" => Instruction::I64Add, "-" => Instruction::I64Sub, "*" => Instruction::I64Mul, "/" => Instruction::I64DivS, _ => Instruction::I64Add };
                        f.instruction(&instr);
                    }
                }
                "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                    let isf = type_of(&args[0]) == "f64" || type_of(&args[1]) == "f64";
                    if isf {
                        gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                        if type_of(&args[0]) != "f64" { f.instruction(&Instruction::F64ConvertI64S); }
                        gen_wasm(f, &args[1], var_map, str_offsets, func_idx);
                        if type_of(&args[1]) != "f64" { f.instruction(&Instruction::F64ConvertI64S); }
                        let instr = match name.as_str() { "<" => Instruction::F64Lt, ">" => Instruction::F64Gt, "<=" => Instruction::F64Le, ">=" => Instruction::F64Ge, "==" => Instruction::F64Eq, "!=" => Instruction::F64Ne, _ => Instruction::F64Eq };
                        f.instruction(&instr);
                    } else {
                        gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                        gen_wasm(f, &args[1], var_map, str_offsets, func_idx);
                        let instr = match name.as_str() { "<" => Instruction::I64LtS, ">" => Instruction::I64GtS, "<=" => Instruction::I64LeS, ">=" => Instruction::I64GeS, "==" => Instruction::I64Eq, "!=" => Instruction::I64Ne, _ => Instruction::I64Eq };
                        f.instruction(&instr);
                    }
                    f.instruction(&Instruction::I64ExtendI32U);
                }
                "print" => {
                    if !args.is_empty() {
                        let t = type_of(&args[0]);
                        gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                        match t.as_str() {
                            "f64" => { f.instruction(&Instruction::Call(1)); }
                            "str" => { f.instruction(&Instruction::Call(2)); }
                            _ => { f.instruction(&Instruction::Call(0)); }
                        }
                    }
                    f.instruction(&Instruction::I64Const(0));
                }
                "len" => {
                    if !args.is_empty() {
                        let t = type_of(&args[0]);
                        if t == "str" {
                            gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                            f.instruction(&Instruction::Call(3)); // str_len
                        } else {
                            gen_wasm(f, &args[0], var_map, str_offsets, func_idx);
                        }
                    } else { f.instruction(&Instruction::I64Const(0)); }
                }
                _ => {
                    // user function
                    if let Some(idx) = func_idx.get(name) {
                        for a in args { gen_wasm(f, a, var_map, str_offsets, func_idx); }
                        f.instruction(&Instruction::Call(*idx));
                        // if void, push 0? We don't know; assume function returns value. If void, call leaves nothing; push 0.
                        // Heuristic: if name not in func_idx? already. We'll push 0 only if? Can't know. Skip; assume non-void.
                    } else {
                        f.instruction(&Instruction::I64Const(0));
                    }
                }
            }
            // str concat handling (moved): if we detected earlier, we returned. But we put str concat inside match? We need handle before match. Let's handle str concat separately above match.
        }
        _ => { f.instruction(&Instruction::I64Const(0)); }
    }
}

fn compile_wasm(prog: &Expr) -> (Vec<u8>, String) {
    // collect strings
    let mut strs = Vec::new();
    collect_strings(prog, &mut strs);
    let mut str_offsets = std::collections::HashMap::new();
    let mut off = 1024i32;
    for s in &strs {
        str_offsets.insert(s.clone(), off);
        off += s.len() as i32 + 1;
    }

    // ail functions
    let ail_fns: Vec<&Expr> = if let Expr::Program(fns) = prog {
        fns.iter().filter(|e| matches!(e, Expr::Fn{..})).collect()
    } else { vec![] };

    // type section
    let mut types = TypeSection::new();
    // imports: 0 print_i64,1 print_f64,2 print_str,3 str_len,4 str_concat,5 js_eval,6 dom_set_inner_html
    types.ty().function([ValType::I64], []);
    types.ty().function([ValType::F64], []);
    types.ty().function([ValType::I64], []);
    types.ty().function([ValType::I64], [ValType::I64]);
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]);
    types.ty().function([ValType::I64], [ValType::I64]);
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]);
    // ail fns
    for f in &ail_fns {
        if let Expr::Fn { params, ret, .. } = *f {
            let pt: Vec<ValType> = params.iter().map(|(_,t)| cl_vt(t)).collect();
            let rt: Vec<ValType> = if ret == "void" { vec![] } else { vec![cl_vt(ret)] };
            types.ty().function(pt, rt);
        }
    }

    // import section
    let mut imports = ImportSection::new();
    imports.import("env", "print_i64", wasm_encoder::EntityType::Function(0));
    imports.import("env", "print_f64", wasm_encoder::EntityType::Function(1));
    imports.import("env", "print_str", wasm_encoder::EntityType::Function(2));
    imports.import("env", "str_len", wasm_encoder::EntityType::Function(3));
    imports.import("env", "str_concat", wasm_encoder::EntityType::Function(4));
    imports.import("env", "js_eval", wasm_encoder::EntityType::Function(5));
    imports.import("env", "dom_set_inner_html", wasm_encoder::EntityType::Function(6));

    // function section (ail funcs only; imports already counted)
    let mut functions = FunctionSection::new();
    for i in 0..ail_fns.len() {
        functions.function(7 + i as u32);
    }

    // memory section
    let mut memories = MemorySection::new();
    memories.memory(MemoryType { minimum: 10, maximum: None, memory64: false, shared: false, page_size_log2: None });

    // export section
    let mut exports = ExportSection::new();
    // find main
    let mut main_idx: Option<u32> = None;
    for (i, f) in ail_fns.iter().enumerate() {
        if let Expr::Fn { name, .. } = *f {
            if name == "main" { main_idx = Some(7 + i as u32); }
        }
    }
    if let Some(mi) = main_idx {
        exports.export("main", ExportKind::Func, mi);
    }
    exports.export("memory", ExportKind::Memory, 0);

    // code section
    let mut code = CodeSection::new();
    // func_idx map
    let mut func_idx = std::collections::HashMap::new();
    for (i, f) in ail_fns.iter().enumerate() {
        if let Expr::Fn { name, .. } = *f {
            func_idx.insert(name.clone(), 7 + i as u32);
        }
    }

    for f in &ail_fns {
        if let Expr::Fn { name, params, ret, body } = *f {
            // populate VAR_TYPES
            crate::VAR_TYPES.lock().unwrap().clear(); // not crate, just use static
            for (pname, ptype) in params {
                VAR_TYPES.lock().unwrap().insert(pname.clone(), ptype.clone());
            }
            // scan lets
            let mut var_map = std::collections::HashMap::new();
            for (i, (pname, _)) in params.iter().enumerate() {
                var_map.insert(pname.clone(), i as u32);
            }
            let mut local_types = Vec::new();
            for e in body {
                scan_lets(e, &mut var_map, &mut local_types);
            }
            // fix var_map indices for lets: our scan stored placeholder (local_types.len() at scan time). Need recompute: params.len() + position. We stored temporary local_types.len() (1-based after push) which equals params.len()? Not exactly. Let's recompute: For each let, index = params.len() + index_in_local_types. But var_map currently has wrong. We'll rebuild by scanning again with proper counting.
            // Rebuild var_map cleanly:
            let mut var_map2 = std::collections::HashMap::new();
            for (i, (pname, _)) in params.iter().enumerate() {
                var_map2.insert(pname.clone(), i as u32);
            }
            let mut lt2 = Vec::new();
            // scan to populate var_map2 with correct indices
            fn scan2(e: &Expr, vm: &mut std::collections::HashMap<String,u32>, lt: &mut Vec<ValType>, params_len: usize) {
                match e {
                    Expr::Let(name, val) => {
                        if !vm.contains_key(name) {
                            let ty = type_of(val);
                            let vt = if ty == "f64" { ValType::F64 } else { ValType::I64 };
                            let idx = (params_len + lt.len()) as u32;
                            vm.insert(name.clone(), idx);
                            lt.push(vt);
                        }
                        scan2(val, vm, lt, params_len);
                    }
                    Expr::Block(v) | Expr::Program(v) => { for x in v { scan2(x, vm, lt, params_len); } }
                    Expr::If(c,t,e) => { scan2(c,vm,lt,params_len); scan2(t,vm,lt,params_len); scan2(e,vm,lt,params_len); }
                    Expr::While(c,body) => { scan2(c,vm,lt,params_len); for x in body { scan2(x,vm,lt,params_len); } }
                    Expr::Call(_, args) => { for a in args { scan2(a,vm,lt,params_len); } }
                    Expr::Set(_, v) => scan2(v, vm, lt, params_len),
                    _ => {}
                }
            }
            for e in body { scan2(e, &mut var_map2, &mut lt2, params.len()); }
            // compress locals
            let mut compressed: Vec<(u32, ValType)> = Vec::new();
            for vt in &lt2 {
                if let Some(last) = compressed.last_mut() {
                    if last.1 == *vt { last.0 += 1; continue; }
                }
                compressed.push((1, *vt));
            }
            let mut func = Function::new(compressed);
            // update VAR_TYPES for lets? type_of already uses global; but locals types known. Insert let types into VAR_TYPES for nested type_of? We'll insert after determining? type_of called during gen uses VAR_TYPES; for a let, type_of(val) called before insert. For nested references after let, VAR_TYPES should include let. We can insert during a pre-pass. Let's insert param+let types now.
            for (pname, ptype) in params { VAR_TYPES.lock().unwrap().insert(pname.clone(), ptype.clone()); }
            // determine let types by scanning again
            fn insert_lets(e: &Expr) {
                match e {
                    Expr::Let(name, val) => {
                        VAR_TYPES.lock().unwrap().insert(name.clone(), type_of(val));
                        insert_lets(val);
                    }
                    Expr::Block(v) | Expr::Program(v) => { for x in v { insert_lets(x); } }
                    Expr::If(c,t,e) => { insert_lets(c); insert_lets(t); insert_lets(e); }
                    Expr::While(c,body) => { insert_lets(c); for x in body { insert_lets(x); } }
                    Expr::Call(_, args) => { for a in args { insert_lets(a); } }
                    Expr::Set(_, v) => insert_lets(v),
                    _ => {}
                }
            }
            for e in body { insert_lets(e); }

            // gen body
            for (i, e) in body.iter().enumerate() {
                gen_wasm(&mut func, e, &var_map2, &str_offsets, &func_idx);
                if i < body.len().saturating_sub(1) {
                    func.instruction(&Instruction::Drop);
                }
            }
            if body.is_empty() {
                if ret != "void" { 
                    if ret == "f64" { func.instruction(&Instruction::F64Const(0.0)); } else { func.instruction(&Instruction::I64Const(0)); }
                }
            } else if ret == "void" {
                func.instruction(&Instruction::Drop);
            }
            func.instruction(&Instruction::End);
            code.function(&func);
            let _ = name;
        }
    }

    // data section
    let mut data = DataSection::new();
    for (s, off) in &str_offsets {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        data.active(0, &wasm_encoder::ConstExpr::i32_const(*off), bytes);
    }

    // assemble
    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&functions);
    module.section(&memories);
    module.section(&exports);
    module.section(&code);
    module.section(&data);

    let wasm = module.finish();

    // html harness
    let html = r#"<!DOCTYPE html>
<html><body>
<script>
const importObject = {
  env: {
    print_i64: (x) => console.log(Number(x)),
    print_f64: (x) => console.log(x),
    print_str: (p) => {
      const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
      let s = '';
      for (let i = Number(p); mem[i] !== 0; i++) s += String.fromCharCode(mem[i]);
      console.log(s);
    },
    str_len: (p) => {
      const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
      let n = 0; const pp = Number(p); while (mem[pp+n] !== 0) n++; return BigInt(n);
    },
    str_concat: (a, b) => {
      const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
      const aa = Number(a), bb = Number(b);
      let la = 0; while (mem[aa+la] !== 0) la++;
      let lb = 0; while (mem[bb+lb] !== 0) lb++;
      if (!wasmInstance._bump) wasmInstance._bump = 65536;
      const dest = wasmInstance._bump;
      wasmInstance._bump += la + lb + 1;
      for (let i = 0; i < la; i++) mem[dest+i] = mem[aa+i];
      for (let i = 0; i < lb; i++) mem[dest+la+i] = mem[bb+i];
      mem[dest+la+lb] = 0;
      return BigInt(dest);
    },
    js_eval: (codePtr) => {
      const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
      let code = ''; const pp = Number(codePtr); for (let i = pp; mem[i] !== 0; i++) code += String.fromCharCode(mem[i]);
      try { return BigInt(eval(code) || 0); } catch (e) { console.error(e); return 0n; }
    },
    dom_set_inner_html: (selPtr, htmlPtr) => {
      const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
      let sel = ''; let pp = Number(selPtr); for (let i = pp; mem[i] !== 0; i++) sel += String.fromCharCode(mem[i]);
      let html = ''; pp = Number(htmlPtr); for (let i = pp; mem[i] !== 0; i++) html += String.fromCharCode(mem[i]);
      const el = document.querySelector(sel);
      if (el) el.innerHTML = html;
      return 0n;
    }
  }
};
let wasmInstance;
WebAssembly.instantiateStreaming(fetch('PLACEHOLDER.wasm'), importObject).then(r => {
  wasmInstance = r.instance;
  if (wasmInstance.exports.main) wasmInstance.exports.main();
});
</script>
</body></html>"#.replace("PLACEHOLDER", "PLACEHOLDER");

    (wasm, html)
}

// ---------- c backend ----------
fn ctype(t: &str) -> &'static str {
    match t { "i64" => "long long", "f64" => "double", "bool" => "int", "str" => "char*", "array" => "Arr*", "mat" => "Mat*", _ => "long long" }
}

fn c_escape(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn collect_lets(e: &Expr, vars: &mut std::collections::HashMap<String, String>) {
    match e {
        Expr::Let(name, val) => {
            let ty = type_of(val);
            if !vars.contains_key(name) { vars.insert(name.clone(), ty); }
            collect_lets(val, vars);
        }
        Expr::Block(v) | Expr::Program(v) | Expr::Fn { body: v, .. } => { for x in v { collect_lets(x, vars); } }
        Expr::If(c,t,ee) => { collect_lets(c,vars); collect_lets(t,vars); collect_lets(ee,vars); }
        Expr::While(c,body) => { collect_lets(c,vars); for x in body { collect_lets(x,vars); } }
        Expr::Call(_, args) => { for a in args { collect_lets(a,vars); } }
        Expr::Set(_, v) => collect_lets(v, vars),
        _ => {}
    }
}

fn gen_c_expr(e: &Expr) -> String {
    match e {
        Expr::Int(n) => format!("{}", n),
        Expr::Float(x) => {
            let s = format!("{}", x);
            if s.contains('.') || s.contains('e') { s } else { format!("{}.0", s) }
        }
        Expr::Bool(b) => if *b { "1".into() } else { "0".into() },
        Expr::Str(s) => format!("\"{}\"", c_escape(s)),
        Expr::Var(v) => v.clone(),
        Expr::Let(name, val) => format!("({} = {})", name, gen_c_expr(val)),
        Expr::Set(name, val) => format!("({} = {})", name, gen_c_expr(val)),
        Expr::If(c, t, ee) => {
            let ty = type_of(e);
            let ct = ctype(&ty);
            format!("(({}) ? ({}) : ({}))", gen_c_expr(c), gen_c_expr(t), gen_c_expr(ee))
        }
        Expr::While(c, body) => {
            let mut bs = String::new();
            for x in body { bs.push_str(&format!("{}; ", gen_c_stmt(x))); }
            format!("({{ while({}) {{ {} }} 0; }})", gen_c_expr(c), bs)
        }
        Expr::Block(v) => {
            let mut s = String::from("({ ");
            for (i, x) in v.iter().enumerate() {
                if i < v.len().saturating_sub(1) {
                    s.push_str(&format!("{}; ", gen_c_stmt(x)));
                } else {
                    s.push_str(&format!("{}; ", gen_c_expr(x)));
                }
            }
            if v.is_empty() { s.push_str("0; "); }
            s.push_str("})");
            s
        }
        Expr::Call(name, args) => {
            match name.as_str() {
                "+" | "-" | "*" | "/" | "%" => {
                    let isf = type_of(&args[0]) == "f64" || (args.len()>1 && type_of(&args[1]) == "f64");
                    if isf {
                        let a = gen_c_expr(&args[0]);
                        let b = if args.len()>1 { gen_c_expr(&args[1]) } else { "0".into() };
                        let op = match name.as_str() { "+" => "+", "-" => "-", "*" => "*", "/" => "/", "%" => "%", _ => "+" };
                        format!("({} {} {})", a, op, b)
                    } else if type_of(e) == "str" {
                        // concat handled below? Actually + for str
                        if args.len() == 2 && (type_of(&args[0]) == "str" || type_of(&args[1]) == "str") {
                            format!("ail_str_concat({}, {})", gen_c_expr(&args[0]), gen_c_expr(&args[1]))
                        } else {
                            let a = gen_c_expr(&args[0]);
                            let b = if args.len()>1 { gen_c_expr(&args[1]) } else { "0".into() };
                            let op = match name.as_str() { "+" => "+", "-" => "-", "*" => "*", "/" => "/", "%" => "%", _ => "+" };
                            format!("({} {} {})", a, op, b)
                        }
                    } else {
                        let a = gen_c_expr(&args[0]);
                        let b = if args.len()>1 { gen_c_expr(&args[1]) } else { "0".into() };
                        let op = match name.as_str() { "+" => "+", "-" => "-", "*" => "*", "/" => "/", "%" => "%", _ => "+" };
                        format!("({} {} {})", a, op, b)
                    }
                }
                "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                    let a = gen_c_expr(&args[0]);
                    let b = if args.len()>1 { gen_c_expr(&args[1]) } else { "0".into() };
                    let op = match name.as_str() { "<" => "<", ">" => ">", "<=" => "<=", ">=" => ">=", "==" => "==", "!=" => "!=", _ => "==" };
                    format!("({} {} {})", a, op, b)
                }
                "print" => {
                    if args.is_empty() { "0".into() } else {
                        let t = type_of(&args[0]);
                        let a = gen_c_expr(&args[0]);
                        match t.as_str() { "f64" => format!("(ail_print_f64({}), 0)", a), "str" => format!("(ail_print_str({}), 0)", a), "bool" => format!("(ail_print_bool({}), 0)", a), _ => format!("(ail_print_i64({}), 0)", a) }
                    }
                }
                "len" => {
                    if args.is_empty() { "0".into() } else {
                        let t = type_of(&args[0]);
                        let a = gen_c_expr(&args[0]);
                        if t == "str" { format!("ail_str_len({})", a) } else if t == "array" { format!("ail_array_len({})", a) } else { "0".into() }
                    }
                }
                "array" => "ail_array_new()".into(),
                "push" => { if args.len() >= 2 { format!("(ail_array_push({}, {}), 0)", gen_c_expr(&args[0]), gen_c_expr(&args[1])) } else { "0".into() } }
                "get" => { if args.len() >= 2 { format!("ail_array_get({}, {})", gen_c_expr(&args[0]), gen_c_expr(&args[1])) } else { "0".into() } }
                "mat_new" => { if args.len() >= 2 { format!("ail_mat_new({}, {})", gen_c_expr(&args[0]), gen_c_expr(&args[1])) } else { "0".into() } }
                "mat_set" => { if args.len() >= 4 { format!("(ail_mat_set({}, {}, {}, {}), 0)", gen_c_expr(&args[0]), gen_c_expr(&args[1]), gen_c_expr(&args[2]), gen_c_expr(&args[3])) } else { "0".into() } }
                "mat_get" => { if args.len() >= 3 { format!("ail_mat_get({}, {}, {})", gen_c_expr(&args[0]), gen_c_expr(&args[1]), gen_c_expr(&args[2])) } else { "0".into() } }
                "mat_rows" => { if !args.is_empty() { format!("ail_mat_rows({})", gen_c_expr(&args[0])) } else { "0".into() } }
                "mat_cols" => { if !args.is_empty() { format!("ail_mat_cols({})", gen_c_expr(&args[0])) } else { "0".into() } }
                "mat_add" => { if args.len() >= 2 { format!("ail_mat_add({}, {})", gen_c_expr(&args[0]), gen_c_expr(&args[1])) } else { "0".into() } }
                "mat_mul" => { if args.len() >= 2 { format!("ail_mat_mul({}, {})", gen_c_expr(&args[0]), gen_c_expr(&args[1])) } else { "0".into() } }
                "mat_relu" => { if !args.is_empty() { format!("ail_mat_relu({})", gen_c_expr(&args[0])) } else { "0".into() } }
                _ => {
                    // user function call
                    let a: Vec<String> = args.iter().map(gen_c_expr).collect();
                    format!("{}({})", name, a.join(", "))
                }
            }
        }
        Expr::Fn { .. } | Expr::Program(_) => "0".into(),
    }
}

fn gen_c_stmt(e: &Expr) -> String {
    match e {
        Expr::Let(name, val) => format!("{} = {};", name, gen_c_expr(val)),
        Expr::Set(name, val) => format!("{} = {};", name, gen_c_expr(val)),
        Expr::If(c, t, ee) => format!("if ({}) {{ {} }} else {{ {} }}", gen_c_expr(c), gen_c_stmt(t), gen_c_stmt(ee)),
        Expr::While(c, body) => {
            let mut bs = String::new();
            for x in body { bs.push_str(&format!("{} ", gen_c_stmt(x))); }
            format!("while ({}) {{ {} }}", gen_c_expr(c), bs)
        }
        Expr::Block(v) => {
            let mut s = String::from("{ ");
            for x in v { s.push_str(&format!("{} ", gen_c_stmt(x))); }
            s.push('}');
            s
        }
        _ => format!("{};", gen_c_expr(e)),
    }
}

fn clang_path() -> String {
    if std::process::Command::new("clang").arg("--version").status().map(|s| s.success()).unwrap_or(false) {
        "clang".to_string()
    } else {
        "C:\\Program Files\\LLVM\\bin\\clang.exe".to_string()
    }
}

fn gcc_path() -> String {
    if std::process::Command::new("gcc").arg("--version").status().map(|s| s.success()).unwrap_or(false) {
        "gcc".to_string()
    } else {
        "C:\\Users\\sophi\\.rustup\\toolchains\\stable-x86_64-pc-windows-gnu\\bin\\gcc.exe".to_string()
    }
}

fn compile_c(prog: &Expr) {
    // collect functions
    let fns: Vec<&Expr> = if let Expr::Program(fns) = prog { fns.iter().filter(|e| matches!(e, Expr::Fn{..})).collect() } else { vec![] };

    let mut code = String::new();
    code.push_str("#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n");

    // structs
    code.push_str("typedef struct { long long len, cap; long long *data; } Arr;\n");
    code.push_str("typedef struct { long long rows, cols; double *data; } Mat;\n\n");

    // helpers
    code.push_str("void ail_print_i64(long long x){ printf(\"%lld\\n\", x); }\n");
    code.push_str("void ail_print_f64(double x){ printf(\"%g\\n\", x); }\n");
    code.push_str("void ail_print_str(const char* s){ printf(\"%s\\n\", s); }\n");
    code.push_str("void ail_print_bool(int x){ printf(\"%s\\n\", x?\"true\":\"false\"); }\n");
    code.push_str("long long ail_str_len(const char* s){ return (long long)strlen(s); }\n");
    code.push_str("char* ail_str_concat(const char* a, const char* b){ size_t la=strlen(a), lb=strlen(b); char* r=malloc(la+lb+1); memcpy(r,a,la); memcpy(r+la,b,lb); r[la+lb]=0; return r; }\n");

    // array helpers
    code.push_str("Arr* ail_array_new(){ Arr* a=malloc(sizeof(Arr)); a->len=0; a->cap=8; a->data=malloc(8*sizeof(long long)); return a; }\n");
    code.push_str("void ail_array_push(Arr* a,long long v){ if(a->len==a->cap){ a->cap*=2; a->data=realloc(a->data,a->cap*sizeof(long long)); } a->data[a->len++]=v; }\n");
    code.push_str("long long ail_array_get(Arr* a,long long i){ if(i<0||i>=a->len) return 0; return a->data[i]; }\n");
    code.push_str("long long ail_array_len(Arr* a){ return a->len; }\n");

    // mat helpers
    code.push_str("Mat* ail_mat_new(long long r,long long c){ Mat* m=malloc(sizeof(Mat)); m->rows=r; m->cols=c; m->data=calloc((size_t)(r*c),sizeof(double)); return m; }\n");
    code.push_str("void ail_mat_set(Mat* m,long long r,long long c,double v){ if(r<0||r>=m->rows||c<0||c>=m->cols) return; m->data[r*m->cols+c]=v; }\n");
    code.push_str("double ail_mat_get(Mat* m,long long r,long long c){ if(r<0||r>=m->rows||c<0||c>=m->cols) return 0; return m->data[r*m->cols+c]; }\n");
    code.push_str("long long ail_mat_rows(Mat* m){ return m->rows; }\n");
    code.push_str("long long ail_mat_cols(Mat* m){ return m->cols; }\n");
    code.push_str("Mat* ail_mat_add(Mat* a,Mat* b){ if(a->rows!=b->rows||a->cols!=b->cols) return NULL; Mat* r=ail_mat_new(a->rows,a->cols); for(long long i=0;i<a->rows*a->cols;i++) r->data[i]=a->data[i]+b->data[i]; return r; }\n");
    code.push_str("Mat* ail_mat_mul(Mat* a,Mat* b){ if(a->cols!=b->rows) return NULL; Mat* r=ail_mat_new(a->rows,b->cols); for(long long i=0;i<a->rows;i++) for(long long j=0;j<b->cols;j++){ double s=0; for(long long k=0;k<a->cols;k++) s+=a->data[i*a->cols+k]*b->data[k*b->cols+j]; r->data[i*r->cols+j]=s; } return r; }\n");
    code.push_str("Mat* ail_mat_relu(Mat* m){ for(long long i=0;i<m->rows*m->cols;i++) if(m->data[i]<0) m->data[i]=0; return m; }\n\n");

    // forward decls
    for f in &fns {
        if let Expr::Fn { name, params, ret, .. } = *f {
            let ps: Vec<String> = params.iter().map(|(n,t)| format!("{} {}", ctype(t), n)).collect();
            code.push_str(&format!("{} {}({});\n", ctype(ret), name, ps.join(", ")));
        }
    }
    code.push('\n');

    // definitions
    for f in &fns {
        if let Expr::Fn { name, params, ret, body } = *f {
            // collect lets
            let mut lets = std::collections::HashMap::new();
            for e in body { collect_lets(e, &mut lets); }
            // params types
            let ps: Vec<String> = params.iter().map(|(n,t)| format!("{} {}", ctype(t), n)).collect();
            code.push_str(&format!("{} {}({}) {{\n", ctype(ret), name, ps.join(", ")));
            // declare lets
            for (vname, vty) in &lets {
                code.push_str(&format!("    {} {} = 0;\n", ctype(vty), vname));
            }
            // body
            for e in body {
                code.push_str(&format!("    {}\n", gen_c_stmt(e)));
            }
            if ret != "void" && body.is_empty() {
                // return default
                if ret == "f64" { code.push_str("    return 0.0;\n"); } else { code.push_str("    return 0;\n"); }
            }
            code.push_str("}\n\n");
        }
    }

    // main entry: if main returns int, call it
    // ensure main exists
    let has_main = fns.iter().any(|f| if let Expr::Fn{name,..}=*f { name=="main" } else { false });

    // write file
    let out_base = "ail_out";
    let c_path = format!("{}.c", out_base);
    std::fs::write(&c_path, &code).expect("write c");

    // compile with clang
    let bin = "ail_out.exe";
    let mut cmd = std::process::Command::new(gcc_path());
    cmd.arg("-O3").arg("-march=native").arg("-o").arg(bin).arg(&c_path);
    // try openmp
    cmd.arg("-fopenmp");
    let status = cmd.status();
    match status {
        Ok(s) if s.success() => { println!("built {} (gcc -O3 -march=native)", bin); }
        _ => {
            let mut cmd2 = std::process::Command::new(gcc_path());
            cmd2.arg("-O3").arg("-march=native").arg("-o").arg(bin).arg(&c_path);
            let s2 = cmd2.status().expect("gcc");
            if s2.success() { println!("built {} (gcc -O3 -march=native, no openmp)", bin); }
            else { eprintln!("gcc failed"); std::process::exit(1); }
        }
    }
}

// ---------- llvm backend ----------
fn llvm_type(t: &str) -> &'static str {
    match t { "i64" => "i64", "f64" => "double", "bool" => "i1", "str" => "i8*", "array" => "i8*", "mat" => "i8*", _ => "i64" }
}

fn gen_llvm_expr(e: &Expr, ctx: &mut LlvmCtx, code: &mut String) -> String {
    match e {
        Expr::Int(n) => format!("{}", n),
        Expr::Float(x) => {
            let s = format!("{}", x);
            if s.contains('.') || s.contains('e') { s } else { format!("{}.0", s) }
        }
        Expr::Bool(b) => if *b { "1".into() } else { "0".into() },
        Expr::Str(s) => {
            let gname = ctx.str_globals.get(s).cloned().unwrap_or_else(|| "null".to_string());
            let len = s.len() + 1;
            let val = ctx.temp();
            code.push_str(&format!("  {} = getelementptr inbounds [{} x i8], ptr {}, i64 0, i64 0\n", val, len, gname));
            val
        }
        Expr::Var(v) => {
            let alloca = ctx.var_alloca.get(v).cloned().unwrap_or_else(|| "null".to_string());
            let val = ctx.temp();
            code.push_str(&format!("  {} = load {}, ptr {}\n", val, llvm_type(&ctx.var_types.get(v).cloned().unwrap_or_else(|| "i64".into())), alloca));
            val
        }
        Expr::Let(name, val) => {
            let v = gen_llvm_expr(val, ctx, code);
            let alloca = ctx.var_alloca.get(name).cloned().unwrap_or_default();
            let ty = ctx.var_types.get(name).cloned().unwrap_or_else(|| "i64".into());
            code.push_str(&format!("  store {} {}, ptr {}\n", llvm_type(&ty), v, alloca));
            v
        }
        Expr::Set(name, val) => {
            let v = gen_llvm_expr(val, ctx, code);
            let alloca = ctx.var_alloca.get(name).cloned().unwrap_or_default();
            let ty = ctx.var_types.get(name).cloned().unwrap_or_else(|| "i64".into());
            code.push_str(&format!("  store {} {}, ptr {}\n", llvm_type(&ty), v, alloca));
            v
        }
        Expr::If(c, t, ee) => {
            let ty = type_of(e);
            let lt = llvm_type(&ty);
            let cond = gen_llvm_expr(c, ctx, code);
            let cond_bool = ctx.temp();
            code.push_str(&format!("  {} = icmp ne i64 {}, 0\n", cond_bool, cond));
            let then_lbl = ctx.label("then");
            let else_lbl = ctx.label("else");
            let merge_lbl = ctx.label("merge");
            code.push_str(&format!("  br i1 {}, label %{}, label %{}\n", cond_bool, then_lbl, else_lbl));
            code.push_str(&format!("{}:\n", then_lbl));
            let tval = gen_llvm_expr(t, ctx, code);
            code.push_str(&format!("  br label %{}\n", merge_lbl));
            code.push_str(&format!("{}:\n", else_lbl));
            let eval = gen_llvm_expr(ee, ctx, code);
            code.push_str(&format!("  br label %{}\n", merge_lbl));
            code.push_str(&format!("{}:\n", merge_lbl));
            let phi = ctx.temp();
            code.push_str(&format!("  {} = phi {} [ {}, %{} ], [ {}, %{} ]\n", phi, lt, tval, then_lbl, eval, else_lbl));
            phi
        }
        Expr::While(c, body) => {
            let header = ctx.label("while_hdr");
            let bodylbl = ctx.label("while_body");
            let exit = ctx.label("while_exit");
            code.push_str(&format!("  br label %{}\n", header));
            code.push_str(&format!("{}:\n", header));
            let cond = gen_llvm_expr(c, ctx, code);
            let cond_bool = ctx.temp();
            code.push_str(&format!("  {} = icmp ne i64 {}, 0\n", cond_bool, cond));
            code.push_str(&format!("  br i1 {}, label %{}, label %{}\n", cond_bool, bodylbl, exit));
            code.push_str(&format!("{}:\n", bodylbl));
            for x in body { gen_llvm_stmt(x, ctx, code); }
            code.push_str(&format!("  br label %{}\n", header));
            code.push_str(&format!("{}:\n", exit));
            "0".into()
        }
        Expr::Block(v) => {
            let mut last = "0".to_string();
            for (i, x) in v.iter().enumerate() {
                if i < v.len().saturating_sub(1) {
                    gen_llvm_stmt(x, ctx, code);
                } else {
                    last = gen_llvm_expr(x, ctx, code);
                }
            }
            if v.is_empty() { "0".into() } else { last }
        }
        Expr::Call(name, args) => {
            if name == "+" && args.len() == 2 && (type_of(&args[0]) == "str" || type_of(&args[1]) == "str") {
                let a = gen_llvm_expr(&args[0], ctx, code);
                let b = gen_llvm_expr(&args[1], ctx, code);
                let v = ctx.temp();
                code.push_str(&format!("  {} = call i8* @ail_str_concat(i8* {}, i8* {})\n", v, a, b));
                return v;
            }
            match name.as_str() {
                "+" | "-" | "*" | "/" | "%" => {
                    let isf = type_of(&args[0]) == "f64" || (args.len()>1 && type_of(&args[1]) == "f64");
                    if isf {
                        let a = gen_llvm_expr(&args[0], ctx, code);
                        let a = if type_of(&args[0]) != "f64" { let t = ctx.temp(); code.push_str(&format!("  {} = sitofp i64 {} to double\n", t, a)); t } else { a };
                        let b = if args.len()>1 { gen_llvm_expr(&args[1], ctx, code) } else { "0".into() };
                        let b = if args.len()>1 && type_of(&args[1]) != "f64" { let t = ctx.temp(); code.push_str(&format!("  {} = sitofp i64 {} to double\n", t, b)); t } else { b };
                        let op = match name.as_str() { "+" => "fadd", "-" => "fsub", "*" => "fmul", "/" => "fdiv", "%" => "frem", _ => "fadd" };
                        let val = ctx.temp();
                        code.push_str(&format!("  {} = {} double {}, {}\n", val, op, a, b));
                        val
                    } else {
                        let a = gen_llvm_expr(&args[0], ctx, code);
                        let b = if args.len()>1 { gen_llvm_expr(&args[1], ctx, code) } else { "0".into() };
                        let op = match name.as_str() { "+" => "add", "-" => "sub", "*" => "mul", "/" => "sdiv", "%" => "srem", _ => "add" };
                        let val = ctx.temp();
                        code.push_str(&format!("  {} = {} i64 {}, {}\n", val, op, a, b));
                        val
                    }
                }
                "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                    let isf = type_of(&args[0]) == "f64" || (args.len()>1 && type_of(&args[1]) == "f64");
                    let a = gen_llvm_expr(&args[0], ctx, code);
                    let b = if args.len()>1 { gen_llvm_expr(&args[1], ctx, code) } else { "0".into() };
                    let cmp = if isf {
                        let (a,b) = if type_of(&args[0]) != "f64" { (format!("{}.0", a), b) } else { (a,b) }; // not converting; simplify
                        let op = match name.as_str() { "<" => "fcmp olt", ">" => "fcmp ogt", "<=" => "fcmp ole", ">=" => "fcmp oge", "==" => "fcmp oeq", "!=" => "fcmp one", _ => "fcmp oeq" };
                        let v = ctx.temp();
                        code.push_str(&format!("  {} = {} double {}, {}\n", v, op, a, b));
                        v
                    } else {
                        let op = match name.as_str() { "<" => "icmp slt", ">" => "icmp sgt", "<=" => "icmp sle", ">=" => "icmp sge", "==" => "icmp eq", "!=" => "icmp ne", _ => "icmp eq" };
                        let v = ctx.temp();
                        code.push_str(&format!("  {} = {} i64 {}, {}\n", v, op, a, b));
                        v
                    };
                    let z = ctx.temp();
                    code.push_str(&format!("  {} = zext i1 {} to i64\n", z, cmp));
                    z
                }
                "print" => {
                    if !args.is_empty() {
                        let t = type_of(&args[0]);
                        let a = gen_llvm_expr(&args[0], ctx, code);
                        match t.as_str() {
                            "f64" => { code.push_str(&format!("  call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], ptr @fmt_f64, i64 0, i64 0), double {})\n", a)); }
                            "str" => { code.push_str(&format!("  call i32 @puts(i8* {})\n", a)); }
                            "bool" => { code.push_str(&format!("  call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], ptr @fmt_bool, i64 0, i64 0), i64 {})\n", a)); } // not perfect
                            _ => { code.push_str(&format!("  call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], ptr @fmt_i64, i64 0, i64 0), i64 {})\n", a)); }
                        }
                    }
                    "0".into()
                }
                "len" => {
                    if !args.is_empty() && type_of(&args[0]) == "str" {
                        let a = gen_llvm_expr(&args[0], ctx, code);
                        let v = ctx.temp();
                        code.push_str(&format!("  {} = call i64 @ail_str_len(i8* {})\n", v, a));
                        v
                    } else { "0".into() }
                }
                "str_concat" if args.len() == 2 => {
                    // not a builtin name in Ail; handled as function? skip
                    "0".into()
                }
                _ => {
                    // user function or unknown
                    let a: Vec<String> = args.iter().map(|x| {
                        let v = gen_llvm_expr(x, ctx, code);
                        let t = llvm_type(&type_of(x));
                        format!("{} {}", t, v)
                    }).collect();
                    let v = ctx.temp();
                    let argstr = a.join(", ");
                    let rt = "i64";
                    code.push_str(&format!("  {} = call {} @{}({})\n", v, rt, name, argstr));
                    v
                }
            }
        }
        Expr::Fn { .. } | Expr::Program(_) => "0".into(),
    }
}

fn gen_llvm_stmt(e: &Expr, ctx: &mut LlvmCtx, code: &mut String) {
    match e {
        Expr::Let(name, val) => {
            let v = gen_llvm_expr(val, ctx, code);
            let alloca = ctx.var_alloca.get(name).cloned().unwrap_or_default();
            let ty = ctx.var_types.get(name).cloned().unwrap_or_else(|| "i64".into());
            code.push_str(&format!("  store {} {}, ptr {}\n", llvm_type(&ty), v, alloca));
        }
        Expr::Set(name, val) => {
            let v = gen_llvm_expr(val, ctx, code);
            let alloca = ctx.var_alloca.get(name).cloned().unwrap_or_default();
            let ty = ctx.var_types.get(name).cloned().unwrap_or_else(|| "i64".into());
            code.push_str(&format!("  store {} {}, ptr {}\n", llvm_type(&ty), v, alloca));
        }
        Expr::If(c, t, ee) => { let _ = gen_llvm_expr(e, ctx, code); }
        Expr::While(c, body) => { let _ = gen_llvm_expr(e, ctx, code); }
        Expr::Block(v) => { for x in v { gen_llvm_stmt(x, ctx, code); } }
        _ => { let _ = gen_llvm_expr(e, ctx, code); }
    }
}

struct LlvmCtx {
    temp_counter: usize,
    label_counter: usize,
    var_alloca: std::collections::HashMap<String, String>,
    var_types: std::collections::HashMap<String, String>,
    str_globals: std::collections::HashMap<String, String>,
}

impl LlvmCtx {
    fn temp(&mut self) -> String { let t = format!("%t{}", self.temp_counter); self.temp_counter += 1; t }
    fn label(&mut self, prefix: &str) -> String { let l = format!("{}{}", prefix, self.label_counter); self.label_counter += 1; l }
}

fn compile_llvm(prog: &Expr) {
    // collect strings
    let mut strs = Vec::new();
    collect_strings(prog, &mut strs);
    let mut str_globals = std::collections::HashMap::new();
    let mut global_decls = String::new();
    for (i, s) in strs.iter().enumerate() {
        let gname = format!("@.str{}", i);
        str_globals.insert(s.clone(), gname.clone());
        let len = s.len() + 1;
        let escaped = s.replace('\\', "\\\\").replace('"', "\\22").replace('\n', "\\0A");
        global_decls.push_str(&format!("{} = private unnamed_addr constant [{} x i8] c\"{}\\00\", align 1\n", gname, len, escaped));
    }

    let fns: Vec<&Expr> = if let Expr::Program(fns) = prog { fns.iter().filter(|e| matches!(e, Expr::Fn{..})).collect() } else { vec![] };

    // build function code
    let mut functions_ir = String::new();
    for f in &fns {
        if let Expr::Fn { name, params, ret, body } = *f {
            // var types
            let mut var_types = std::collections::HashMap::new();
            for (pname, ptype) in params {
                var_types.insert(pname.clone(), ptype.clone());
            }
            // collect lets
            let mut lets = std::collections::HashMap::new();
            for e in body { collect_lets(e, &mut lets); }
            for (lname, lty) in &lets {
                var_types.insert(lname.clone(), lty.clone());
            }
            // populate global type map for type_of()
            VAR_TYPES.lock().unwrap().clear();
            for (k, v) in &var_types {
                VAR_TYPES.lock().unwrap().insert(k.clone(), v.clone());
            }

            // allocas
            let mut ctx = LlvmCtx {
                temp_counter: 0,
                label_counter: 0,
                var_alloca: std::collections::HashMap::new(),
                var_types: var_types.clone(),
                str_globals: str_globals.clone(),
            };
            let mut allocas = String::new();
            for (lname, lty) in &lets {
                let ptr = format!("%{}.addr", lname);
                ctx.var_alloca.insert(lname.clone(), ptr.clone());
                allocas.push_str(&format!("  {} = alloca {}\n", ptr, llvm_type(lty)));
            }
            // params are SSA args, not alloca; but for uniformity store params into allocas? We can keep params as SSA values and reference by name? In LLVM, param names are %0, %1... or named. We'll name params in signature and use them directly. But gen_llvm_expr for Var expects alloca. So allocate alloca for params too and store incoming arg.
            // Adjust: add allocas for params
            let mut param_names: Vec<String> = Vec::new();
            for (i, (pname, ptype)) in params.iter().enumerate() {
                let argname = format!("%arg{}", i);
                param_names.push(argname.clone());
                let ptr = format!("%{}.addr", pname);
                ctx.var_alloca.insert(pname.clone(), ptr.clone());
                allocas.push_str(&format!("  {} = alloca {}\n", ptr, llvm_type(ptype)));
            }

            let mut body_code = String::new();
            // store params
            for (i, (pname, ptype)) in params.iter().enumerate() {
                let argname = format!("%arg{}", i);
                let ptr = format!("%{}.addr", pname);
                body_code.push_str(&format!("  store {} {}, ptr {}\n", llvm_type(ptype), argname, ptr));
            }

            // generate body
            for (i, e) in body.iter().enumerate() {
                if i == body.len().saturating_sub(1) && ret != "void" {
                    let val = gen_llvm_expr(e, &mut ctx, &mut body_code);
                    body_code.push_str(&format!("  ret {} {}\n", llvm_type(ret), val));
                } else {
                    gen_llvm_stmt(e, &mut ctx, &mut body_code);
                }
            }
            if body.is_empty() {
                if ret != "void" {
                    let def = if ret == "f64" { "0.0" } else { "0" };
                    body_code.push_str(&format!("  ret {} {}\n", llvm_type(ret), def));
                } else {
                    body_code.push_str("  ret void\n");
                }
            } else if ret == "void" {
                // ensure last stmt didn't leave value; gen_llvm_stmt discards. Add ret void if not already.
                if !body_code.trim_end().ends_with("ret void") {
                    body_code.push_str("  ret void\n");
                }
            }

            // signature
            let ps: Vec<String> = params.iter().enumerate().map(|(i, (_, t))| format!("{} %arg{}", llvm_type(t), i)).collect();
            let rt = if ret == "void" { "void" } else { llvm_type(ret) };
            functions_ir.push_str(&format!("define {} @{}({}) {{\n", rt, name, ps.join(", ")));
            functions_ir.push_str("entry:\n");
            functions_ir.push_str(&allocas);
            functions_ir.push_str(&body_code);
            functions_ir.push_str("}\n\n");
        }
    }

    // assemble module
    let mut ir = String::new();
    ir.push_str("; ModuleID = 'ail'\n");
    ir.push_str("source_filename = \"ail.ll\"\n");
    ir.push_str("target triple = \"x86_64-w64-windows-gnu\"\n\n");

    // declarations
    ir.push_str("declare i32 @printf(i8*, ...)\n");
    ir.push_str("declare i32 @puts(i8*)\n");
    ir.push_str("declare i64 @strlen(i8*)\n");
    ir.push_str("declare i8* @malloc(i64)\n");
    ir.push_str("declare i8* @memcpy(i8*, i8*, i64)\n");
    ir.push_str("declare void @ail_print_i64(i64)\n");
    ir.push_str("declare void @ail_print_f64(double)\n");


    // format strings
    ir.push_str("@fmt_i64 = private unnamed_addr constant [6 x i8] c\"%lld\\0A\\00\", align 1\n");
    ir.push_str("@fmt_f64 = private unnamed_addr constant [4 x i8] c\"%g\\0A\\00\", align 1\n\n");

    // str globals
    ir.push_str(&global_decls);
    if !global_decls.is_empty() { ir.push('\n'); }

    // functions
    ir.push_str(&functions_ir);

    // str_concat definition in LLVM IR
    ir.push_str("define i8* @ail_str_concat(i8* %a, i8* %b) {\n");
    ir.push_str("entry:\n");
    ir.push_str("  %la = call i64 @strlen(i8* %a)\n");
    ir.push_str("  %lb = call i64 @strlen(i8* %b)\n");
    ir.push_str("  %total = add i64 %la, %lb\n");
    ir.push_str("  %size = add i64 %total, 1\n");
    ir.push_str("  %mem = call i8* @malloc(i64 %size)\n");
    ir.push_str("  call i8* @memcpy(i8* %mem, i8* %a, i64 %la)\n");
    ir.push_str("  %dest2 = getelementptr i8, ptr %mem, i64 %la\n");
    ir.push_str("  call i8* @memcpy(i8* %dest2, i8* %b, i64 %lb)\n");
    ir.push_str("  %nullpos = getelementptr i8, ptr %mem, i64 %total\n");
    ir.push_str("  store i8 0, ptr %nullpos\n");
    ir.push_str("  ret i8* %mem\n");
    ir.push_str("}\n\n");

    ir.push_str("define i64 @ail_str_len(i8* %s) {\n");
    ir.push_str("entry:\n");
    ir.push_str("  %l = call i64 @strlen(i8* %s)\n");
    ir.push_str("  ret i64 %l\n");
    ir.push_str("}\n");

    // write .ll
    let ll_path = "ail_out.ll";
    std::fs::write(ll_path, &ir).expect("write ll");

    // compile IR to object with clang, link with gcc
    let obj = "ail_out.o";
    let mut cmd = std::process::Command::new(clang_path());
    cmd.arg("-O3").arg("-march=native").arg("-c").arg(ll_path).arg("-o").arg(obj);
    let status = cmd.status();
    if !status.map(|s| s.success()).unwrap_or(false) {
        eprintln!("clang (llvm) compile failed"); std::process::exit(1);
    }
    let bin = "ail_out.exe";
    let mut cmd2 = std::process::Command::new(gcc_path());
    cmd2.arg("-O3").arg("-march=native").arg("-o").arg(bin).arg(obj);
    let status2 = cmd2.status();
    if status2.map(|s| s.success()).unwrap_or(false) {
        println!("built {} (llvm ir -O3 -march=native)", bin);
    } else {
        eprintln!("gcc link failed"); std::process::exit(1);
    }
}

use std::hash::{Hash, Hasher};

fn ail_cache_root() -> std::path::PathBuf {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".ail").join("cache")
}

fn ail_cache_dir(target: &str, src: &str) -> std::path::PathBuf {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    target.hash(&mut h);
    src.hash(&mut h);
    let key = format!("{:x}", h.finish());
    ail_cache_root().join(key)
}

fn cache_binary_path(cache_dir: &std::path::Path, target: &str) -> std::path::PathBuf {
    let name = match target {
        "wasm" => "out.wasm",
        _ => "out.exe",
    };
    cache_dir.join(name)
}

fn cache_store(cache_bin: &std::path::Path, produced: &std::path::Path) {
    if let Some(parent) = cache_bin.parent() { let _ = std::fs::create_dir_all(parent); }
    let _ = std::fs::copy(produced, cache_bin);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut target = "native".to_string();
    let mut file: Option<String> = None;
    let mut out: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--target" { if i+1 < args.len() { target = args[i+1].clone(); i += 2; continue; } }
        if args[i] == "-o" { if i+1 < args.len() { out = Some(args[i+1].clone()); i += 2; continue; } }
        if file.is_none() { file = Some(args[i].clone()); }
        i += 1;
    }
    let fpath = file.unwrap_or_else(|| { eprintln!("usage: ail [--target native|wasm] <file.ail> [-o out]"); std::process::exit(1); });
    let src = fs::read_to_string(&fpath).expect("read file");

    // binary cache: skip recompile if cached
    let cache_dir = ail_cache_dir(&target, &src);
    let cache_bin = cache_binary_path(&cache_dir, &target);
    if target != "native" && cache_bin.exists() {
        let status = std::process::Command::new(&cache_bin).status();
        std::process::exit(status.map(|s| s.code().unwrap_or(0)).unwrap_or(1));
    }

    let toks = tokenize(&src);
    let mut parser = Parser::new(toks);
    let prog = parser.parse_program();

    if target == "wasm" {
        let (wasm, html) = compile_wasm(&prog);
        let base = out.clone().unwrap_or_else(|| fpath.trim_end_matches(".ail").to_string());
        let wasm_path = format!("{}.wasm", base);
        let html_path = format!("{}.html", base);
        fs::write(&wasm_path, &wasm).expect("write wasm");
        fs::write(&html_path, &html).expect("write html");
        cache_store(&cache_bin, std::path::Path::new(&wasm_path));
        println!("built {} and {}", wasm_path, html_path);
        return;
    }
    if target == "c" {
        compile_c(&prog);
        cache_store(&cache_bin, std::path::Path::new("ail_out.exe"));
        return;
    }
    if target == "llvm" {
        compile_llvm(&prog);
        cache_store(&cache_bin, std::path::Path::new("ail_out.exe"));
        return;
    }

    let mut cx = Ctx::new();
    cx.declare_funcs(&prog);
    if let Expr::Program(fns) = &prog {
        for f in fns {
            cx.compile_fn(f);
        }
    }
    cx.module.finalize_definitions().unwrap();

    // find main
    let main_fid = cx.funcs.get("main").map(|x| x.0).expect("no main");
    let code_ptr = cx.module.get_finalized_function(main_fid);
    let main_fn: extern "C" fn() = unsafe { std::mem::transmute(code_ptr) };
    main_fn();
}
