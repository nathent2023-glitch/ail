const fs = require('fs');
const wasm = fs.readFileSync('str.wasm');
const importObject = {
  env: {
    print_i64: (x) => { console.log('i64', Number(x)); },
    print_f64: (x) => { console.log('f64', x); },
    print_str: (p) => {
      const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
      let s=''; const pp=Number(p); console.log('print_str ptr', pp, 'byte', mem[pp]);
      for(let i=pp; mem[i]!==0; i++) s+=String.fromCharCode(mem[i]);
      console.log('str', JSON.stringify(s));
    },
    str_len: (p) => { const mem=new Uint8Array(wasmInstance.exports.memory.buffer); let n=0; const pp=Number(p); while(mem[pp+n]!==0)n++; console.log('str_len ptr',pp,'len',n); return BigInt(n); },
    str_concat: (a,b) => {
      const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
      const aa=Number(a), bb=Number(b);
      let la=0; while(mem[aa+la]!==0) la++;
      let lb=0; while(mem[bb+lb]!==0) lb++;
      if (!wasmInstance._bump) wasmInstance._bump = 65536;
      const dest = wasmInstance._bump;
      wasmInstance._bump += la+lb+1;
      for(let i=0;i<la;i++) mem[dest+i]=mem[aa+i];
      for(let i=0;i<lb;i++) mem[dest+la+i]=mem[bb+i];
      mem[dest+la+lb]=0;
      console.log('concat', aa, bb, '->', dest);
      return BigInt(dest);
    }
  }
};
let wasmInstance;
WebAssembly.instantiate(wasm, importObject).then(r => {
  wasmInstance = r.instance;
  if (wasmInstance.exports.main) wasmInstance.exports.main();
}).catch(e => console.error('ERR', e));
