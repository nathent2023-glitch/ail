const fs = require('fs');
const wasm = fs.readFileSync('str.wasm');
const mem = new WebAssembly.Memory({ initial: 10 });
let bump = 65536;
const importObject = {
  env: {
    print_i64: (x) => console.log(Number(x)),
    print_f64: (x) => console.log(x),
    print_str: (p) => {
      const m = new Uint8Array(mem.buffer);
      let s=''; const pp=Number(p); for(let i=pp;m[i]!==0;i++) s+=String.fromCharCode(m[i]);
      console.log(s);
    },
    str_len: (p) => {
      const m = new Uint8Array(mem.buffer);
      let n=0; const pp=Number(p); while(m[pp+n]!==0) n++; return BigInt(n);
    },
    str_concat: (a,b) => {
      const m = new Uint8Array(mem.buffer);
      const aa=Number(a), bb=Number(b);
      let la=0; while(m[aa+la]!==0) la++;
      let lb=0; while(m[bb+lb]!==0) lb++;
      const dest=bump; bump+=la+lb+1;
      for(let i=0;i<la;i++) m[dest+i]=m[aa+i];
      for(let i=0;i<lb;i++) m[dest+la+i]=m[bb+i];
      m[dest+la+lb]=0;
      return BigInt(dest);
    },
    memory: mem
  }
};
WebAssembly.instantiate(wasm, importObject).then(r => {
  if (r.instance.exports.main) r.instance.exports.main();
}).catch(e => console.error(e));
