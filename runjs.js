const fs = require('fs');
const wasm = fs.readFileSync('js.wasm');
const importObject = {
  env: {
    print_i64: (x) => { console.log('i64', Number(x)); },
    print_f64: (x) => { console.log('f64', x); },
    print_str: (p) => { const m=new Uint8Array(wasmInstance.exports.memory.buffer); let s=''; const pp=Number(p); for(let i=pp;m[i]!==0;i++) s+=String.fromCharCode(m[i]); console.log('str',JSON.stringify(s)); },
    str_len: (p) => 0n,
    str_concat: (a,b) => 0n,
    js_eval: (p) => { const m=new Uint8Array(wasmInstance.exports.memory.buffer); let code=''; const pp=Number(p); for(let i=pp;m[i]!==0;i++) code+=String.fromCharCode(m[i]); try{return BigInt(eval(code)||0);}catch(e){console.error(e);return 0n;} },
    dom_set_inner_html: (a,b) => 0n
  }
};
let wasmInstance;
WebAssembly.instantiate(wasm, importObject).then(r => {
  wasmInstance = r.instance;
  if (wasmInstance.exports.main) wasmInstance.exports.main();
}).catch(e => console.error('ERR', e));
