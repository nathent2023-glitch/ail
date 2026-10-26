const fs = require('fs');
const wasm = fs.readFileSync('fib.wasm');
const mem = new WebAssembly.Memory({ initial: 10 });
let bump = 65536;
const importObject = {
  env: {
    print_i64: (x) => console.log(x),
    print_f64: (x) => console.log(x),
    print_str: (p) => {
      const m = new Uint8Array(mem.buffer);
      let s=''; for(let i=p;m[i]!==0;i++) s+=String.fromCharCode(m[i]);
      console.log(s);
    },
    str_len: (p) => {
      const m = new Uint8Array(mem.buffer); let n=0; while(m[p+n]!==0) n++; return BigInt(n);
    },
    str_concat: (a,b) => {
      const m = new Uint8Array(mem.buffer);
      let la=0; while(m[a+la]!==0) la++;
      let lb=0; while(m[b+lb]!==0) lb++;
      const dest=bump; bump+=la+lb+1;
      for(let i=0;i<la;i++) m[dest+i]=m[a+i];
      for(let i=0;i<lb;i++) m[dest+la+i]=m[b+i];
      m[dest+la+lb]=0;
      return BigInt(dest);
    },
    memory: mem
  }
};
WebAssembly.instantiate(wasm, importObject).then(r => {
  if (r.instance.exports.main) r.instance.exports.main();
}).catch(e => console.error(e));
