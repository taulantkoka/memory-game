import { useState, useEffect, useCallback, useRef } from "react";

const EMOJIS = ["🍎","🍊","🍋","🍇","🍓","🍑","🍒","🥝","🌸","🌺","🌻","🌹","🎲","🎯","🎨","🎭","🐶","🐱"];
const DELAY = { flip: 450, think: 300, match: 650, noMatch: 800 };
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

// ── DP move table ──
function computeMoveTable(nMax, M) {
  if (M === null) M = 2 * nMax + 10;
  const e = new Map(); e.set("0,0", 0);
  const opt = new Map();
  for (let n = 1; n <= nMax; n++) {
    if (n <= M) { e.set(`${n},${n}`, n); opt.set(`${n},${n}`, 1); }
    const startK = n <= M ? n - 1 : M;
    for (let k = startK; k >= 0; k--) {
      const pDen = 2*n-k; if (pDen === 0) continue;
      const p = k/pDen, q = 2*(n-k)/pDen;
      let v1 = null;
      if (k >= 1) {
        if (k < M) v1 = p*(1+(e.get(`${n-1},${Math.min(k-1,M)}`)||0)) - q*(e.get(`${n},${Math.min(k+1,M)}`)||0);
        else { const d=1+q; v1=d!==0?p*(1+(e.get(`${n-1},${M-1}`)||0))/d:0; }
      }
      let v2 = 0; const d2 = 2*n-k-1;
      if (d2 > 0) {
        const kP=Math.min(k+1,M),kL=k<M?k:M-1,kA=k<M?k:M-1,kN=Math.min(k+2,M);
        const fl=1/d2,fa=kP>=1?(kP-1)/d2:0,nk1=n-k-1,fn=nk1>0?2*nk1/d2:0;
        const first=k>=1?p*(1+(e.get(`${n-1},${Math.min(k-1,M)}`)||0)):0;
        const ik=fl*(1+(e.get(`${n-1},${Math.min(kL,M)}`)||0));
        const ia=fa*(1+(e.get(`${n-1},${Math.min(kA,M)}`)||0));
        if(kN!==k&&e.has(`${n},${kN}`))v2=first+q*(ik-ia-fn*e.get(`${n},${kN}`));
        else if(k<M)v2=first+q*(ik-ia-fn*(e.get(`${n},${kN}`)||0));
        else{const rhs=first+q*(ik-ia),dm=1+q*fn;v2=dm!==0?rhs/dm:0;}
      } else if(d2===0) v2=1+(k>=1?(e.get(`${n-1},${Math.min(k-1,M)}`)||0):0);
      const v1v=v1!==null?v1:-99999;
      if(k===0){e.set(`${n},${k}`,v2);opt.set(`${n},${k}`,2)}
      else if(k===1){v1v>=v2?(e.set(`${n},${k}`,v1v),opt.set(`${n},${k}`,1)):(e.set(`${n},${k}`,v2),opt.set(`${n},${k}`,2))}
      else{if(v1v>0&&v1v>=v2){e.set(`${n},${k}`,v1v);opt.set(`${n},${k}`,1)}else if(v2>=0&&v2>=v1v){e.set(`${n},${k}`,v2);opt.set(`${n},${k}`,2)}else if(v1v<=0&&v2<=0){e.set(`${n},${k}`,0);opt.set(`${n},${k}`,0)}else{e.set(`${n},${k}`,v1v>v2?v1v:v2);opt.set(`${n},${k}`,v1v>v2?1:2)}}
    }
  }
  return opt;
}

// ── LRU Memory ──
class LRUMemory {
  constructor(cap){this.cap=cap;this.store=new Map()}
  observe(pos,val){if(this.store.has(pos))this.store.delete(pos);else while(this.store.size>=this.cap){this.store.delete(this.store.keys().next().value)}this.store.set(pos,val)}
  findMatch(val,excl){for(const[p,v]of this.store)if(v===val&&p!==excl)return p;return null}
  knows(pos){return this.store.has(pos)}
  knownAlive(alive){let c=0;for(const p of this.store.keys())if(alive.has(p))c++;return c}
  forgetPos(pos){this.store.delete(pos)}
  getEntries(){return[...this.store.entries()]}
  clone(){const m=new LRUMemory(this.cap);for(const[k,v]of this.store)m.store.set(k,v);return m}
  findKnownPairs(alive){
    const byVal=new Map();
    for(const[pos,val]of this.store){if(!alive.has(pos))continue;if(byVal.has(val))return[byVal.get(val),pos];byVal.set(val,pos)}
    return null;
  }
  findTwoKnownNonMatch(alive){
    const entries=[];for(const[pos,val]of this.store)if(alive.has(pos))entries.push([pos,val]);
    for(let i=0;i<entries.length;i++)for(let j=i+1;j<entries.length;j++)if(entries[i][1]!==entries[j][1])return[entries[i][0],entries[j][0]];
    return null;
  }
}

// ── Headless game for simulation ──
function playHeadlessGame(nPairs, mem1Cap, mem2Cap, table1, table2) {
  const vals=EMOJIS.slice(0,nPairs);const deck=[...vals,...vals];
  for(let i=deck.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[deck[i],deck[j]]=[deck[j],deck[i]]}
  const alive=new Set(Array.from({length:2*nPairs},(_,i)=>i));
  const mem=[new LRUMemory(mem1Cap),new LRUMemory(mem2Cap)];
  const tables=[table1,table2];const scores=[0,0];
  const flip=(pos)=>{const v=deck[pos];mem[0].observe(pos,v);mem[1].observe(pos,v);return v};
  const remove=(a,b,p)=>{alive.delete(a);alive.delete(b);mem[0].forgetPos(a);mem[0].forgetPos(b);mem[1].forgetPos(a);mem[1].forgetPos(b);scores[p]++};
  let cur=0,passes=0;
  for(let iter=0;iter<50000&&alive.size>0;iter++){
    // Greedy match
    let took=true;while(took&&alive.size>0){took=false;const pr=mem[cur].findKnownPairs(alive);if(pr){flip(pr[0]);flip(pr[1]);if(deck[pr[0]]===deck[pr[1]]){remove(pr[0],pr[1],cur);took=true}else break}}
    if(alive.size===0)break;
    const n=alive.size/2,k=Math.min(mem[cur].knownAlive(alive),mem[cur].cap);
    let move=tables[cur].get(`${n},${k}`);if(move===undefined)move=2;
    if(move===0){const p0=mem[cur].findTwoKnownNonMatch(alive);if(p0){flip(p0[0]);flip(p0[1])}passes++;if(passes>=4)break;cur=1-cur;continue}
    passes=0;const aa=[...alive],unk=aa.filter(p=>!mem[cur].knows(p));
    const c1=unk.length>0?pick(unk):pick(aa);const v1=flip(c1);
    const mp=mem[cur].findMatch(v1,c1);
    if(mp!==null&&alive.has(mp)){flip(mp);if(deck[mp]===v1){remove(c1,mp,cur);continue}cur=1-cur;continue}
    if(move===1){const kn=aa.filter(p=>mem[cur].knows(p)&&p!==c1);if(kn.length>0)flip(pick(kn));cur=1-cur}
    else{const u2=unk.filter(p=>p!==c1);const c2=u2.length>0?pick(u2):pick(aa.filter(p=>p!==c1));const v2=flip(c2);
      if(v1===v2){remove(c1,c2,cur);continue}
      const opp=1-cur,om=mem[opp].findMatch(v2,c2);if(om!==null&&alive.has(om)){flip(om);if(deck[om]===v2)remove(c2,om,opp)}
      cur=1-cur}
  }
  return scores;
}

// ── UI Components ──
function Sel({label,value,onChange,options,className=""}){
  return(<label className={`flex items-center gap-1.5 bg-stone-800 px-2 py-1 rounded text-xs ${className}`}>
    <span className="text-stone-400 whitespace-nowrap">{label}</span>
    <select value={value} onChange={e=>onChange(e.target.value)} className="bg-stone-700 text-stone-200 rounded px-1 py-0.5 text-xs">
      {options.map(([v,l])=><option key={v} value={v}>{l}</option>)}
    </select>
  </label>);
}

function BotCfg({label,strat,setStrat,stratM,setStratM,mem,setMem,nPairs,color}){
  return(<div className={`bg-stone-800 rounded p-2.5 border-l-2 ${color}`}>
    <div className="text-xs font-bold text-stone-300 mb-1.5">{label}</div>
    <div className="flex flex-wrap gap-1.5">
      <Sel label="Strategy:" value={strat} onChange={setStrat} options={[["bounded","Bounded"],["zwick","Zwick (∞)"]]}/>
      {strat==="bounded"&&<label className="flex items-center gap-1 bg-stone-700 px-2 py-1 rounded text-xs">
        <span className="text-stone-400">Table M={stratM}</span>
        <input type="range" min={3} max={20} value={stratM} onChange={e=>setStratM(+e.target.value)} className="w-14 accent-amber-500"/>
      </label>}
      <label className="flex items-center gap-1 bg-stone-700 px-2 py-1 rounded text-xs">
        <span className="text-stone-400">Memory={mem}</span>
        <input type="range" min={3} max={Math.max(20,nPairs*2)} value={mem} onChange={e=>setMem(+e.target.value)} className="w-14 accent-rose-500"/>
      </label>
    </div>
  </div>);
}

// ── Main ──
export default function MemoryGame() {
  const [mode, setMode] = useState("play");
  const [nPairs, setNPairs] = useState(8);
  // Play mode
  const [humanStarts, setHumanStarts] = useState(true);
  const [showHints, setShowHints] = useState(true);
  const [botStrat, setBotStrat] = useState("bounded");
  const [botStratM, setBotStratM] = useState(7);
  const [botMem, setBotMem] = useState(7);
  // Sim mode
  const [b1s, setB1s] = useState("bounded"); const [b1sm, setB1sm] = useState(7); const [b1m, setB1m] = useState(7);
  const [b2s, setB2s] = useState("zwick");   const [b2sm, setB2sm] = useState(7); const [b2m, setB2m] = useState(7);
  const [simN, setSimN] = useState(1000);
  const [simRes, setSimRes] = useState(null); const [simRunning, setSimRunning] = useState(false);
  // Game state
  const [cards, setCards] = useState([]); const [flipped, setFlipped] = useState(new Set());
  const [matched, setMatched] = useState(new Set()); const [scores, setScores] = useState([0,0]);
  const [turn, setTurn] = useState(0); const [selected, setSelected] = useState([]);
  const [botMemState, setBotMemState] = useState(null);
  const [gameLog, setGameLog] = useState([]); const [gameOver, setGameOver] = useState(false);
  const [botBusy, setBotBusy] = useState(false); const [hl, setHl] = useState(new Set());

  const tblRef = useRef(null); const memRef = useRef(null);
  const mRef = useRef(new Set()); const cRef = useRef([]);

  const log = useCallback((m) => setGameLog(p => [m,...p].slice(0,30)), []);

  const initGame = useCallback(() => {
    const vals=EMOJIS.slice(0,nPairs);const deck=[...vals,...vals];
    for(let i=deck.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[deck[i],deck[j]]=[deck[j],deck[i]]}
    tblRef.current = computeMoveTable(nPairs, botStrat==="zwick"?null:botStratM);
    const m = new LRUMemory(botMem); memRef.current=m; mRef.current=new Set(); cRef.current=deck;
    setCards(deck);setFlipped(new Set());setMatched(new Set());setScores([0,0]);setSelected([]);
    setBotMemState(m.clone());setGameLog([]);setGameOver(false);setBotBusy(false);setHl(new Set());
    setTurn(humanStarts?0:1);
    log(humanStarts?"Your turn! Flip a card.":"Bot goes first...");
  }, [nPairs,botStrat,botStratM,botMem,humanStarts,log]);

  useEffect(()=>{if(mode==="play")initGame()},[initGame,mode]);

  const aliveSet=useCallback(()=>{const s=new Set();for(let i=0;i<cRef.current.length;i++)if(!mRef.current.has(i))s.add(i);return s},[]);

  // Human click
  const handleClick=useCallback((idx)=>{
    if(turn!==0||botBusy||gameOver||mode!=="play")return;
    if(matched.has(idx)||flipped.has(idx)||selected.includes(idx))return;
    if(selected.length>=2)return;
    const nf=new Set(flipped);nf.add(idx);const ns=[...selected,idx];
    setFlipped(nf);setSelected(ns);
    memRef.current.observe(idx,cRef.current[idx]);setBotMemState(memRef.current.clone());
    if(ns.length===2){const[a,b]=ns;
      setTimeout(()=>{
        if(cRef.current[a]===cRef.current[b]){
          const nm=new Set(mRef.current);nm.add(a);nm.add(b);mRef.current=nm;
          setMatched(nm);setScores(s=>[s[0]+1,s[1]]);setSelected([]);
          memRef.current.forgetPos(a);memRef.current.forgetPos(b);setBotMemState(memRef.current.clone());
          setHl(new Set([a,b]));setTimeout(()=>setHl(new Set()),500);
          log("Match! Go again.");if(nm.size===cRef.current.length){setGameOver(true);log("Game over!")}
        } else {
          log("No match. Bot's turn.");
          setTimeout(()=>{const f2=new Set(flipped);f2.delete(a);f2.delete(b);setFlipped(f2);setSelected([]);setTurn(1)},DELAY.noMatch);
        }
      },DELAY.match);
    }
  },[turn,botBusy,gameOver,mode,matched,flipped,selected,log]);

  // Bot turn
  useEffect(()=>{
    if(turn!==1||gameOver||botBusy||mode!=="play")return;
    setBotBusy(true);
    const mem=memRef.current;const al=aliveSet();const n=al.size/2;
    if(n===0){setGameOver(true);setBotBusy(false);return}

    // Greedy match
    const pair=mem.findKnownPairs(al);
    if(pair){const[p1,p2]=pair;
      log(`Bot takes known pair (${cRef.current[p1]})`);
      setTimeout(()=>{setFlipped(f=>new Set([...f,p1]));
        setTimeout(()=>{setFlipped(f=>new Set([...f,p2]));
          setTimeout(()=>{
            const nm=new Set(mRef.current);nm.add(p1);nm.add(p2);mRef.current=nm;setMatched(nm);setScores(s=>[s[0],s[1]+1]);
            mem.forgetPos(p1);mem.forgetPos(p2);setBotMemState(mem.clone());
            setHl(new Set([p1,p2]));setTimeout(()=>setHl(new Set()),500);
            if(nm.size===cRef.current.length){setGameOver(true);setBotBusy(false);log("Game over!")}
            else setBotBusy(false);
          },DELAY.match);
        },DELAY.flip);
      },DELAY.think);return}

    const k=Math.min(mem.knownAlive(al),mem.cap);
    let move=tblRef.current.get(`${n},${k}`);if(move===undefined)move=2;
    const aa=[...al],unk=aa.filter(p=>!mem.knows(p)),kn=aa.filter(p=>mem.knows(p));

    // 0-move: flip two known non-matching cards
    if(move===0){
      const p0=mem.findTwoKnownNonMatch(al);
      if(p0){const[c1,c2]=p0;
        log(`Bot wastes turn (0-move): two known non-matching. n=${n}, k=${k}`);
        setTimeout(()=>{setFlipped(f=>new Set([...f,c1]));mem.observe(c1,cRef.current[c1]);setBotMemState(mem.clone());
          setTimeout(()=>{setFlipped(f=>new Set([...f,c2]));mem.observe(c2,cRef.current[c2]);setBotMemState(mem.clone());
            setTimeout(()=>{setFlipped(f=>{const nf=new Set(f);nf.delete(c1);nf.delete(c2);return nf});setBotBusy(false);setTurn(0);log("Your turn!")},DELAY.noMatch);
          },DELAY.flip);
        },DELAY.think);return}
      move=1; // fallback
    }

    const moveName=move===1?"1-move (conservative)":"2-move (aggressive)";
    log(`Bot: ${moveName}. n=${n}, k=${k}`);
    const c1=unk.length>0?pick(unk):pick(aa);

    setTimeout(()=>{
      setFlipped(f=>new Set([...f,c1]));const v1=cRef.current[c1];
      mem.observe(c1,v1);setBotMemState(mem.clone());
      const mp=mem.findMatch(v1,c1);

      if(mp!==null&&al.has(mp)){
        log("Flip matches a memory!");
        setTimeout(()=>{setFlipped(f=>new Set([...f,mp]));mem.observe(mp,cRef.current[mp]);setBotMemState(mem.clone());
          if(cRef.current[mp]===v1){
            setTimeout(()=>{const nm=new Set(mRef.current);nm.add(c1);nm.add(mp);mRef.current=nm;setMatched(nm);setScores(s=>[s[0],s[1]+1]);
              mem.forgetPos(c1);mem.forgetPos(mp);setBotMemState(mem.clone());
              setHl(new Set([c1,mp]));setTimeout(()=>setHl(new Set()),500);
              if(nm.size===cRef.current.length){setGameOver(true);setBotBusy(false);log("Game over!")}
              else setBotBusy(false);
            },DELAY.match);
          } else setTimeout(()=>{setFlipped(f=>{const nf=new Set(f);nf.delete(c1);nf.delete(mp);return nf});setBotBusy(false);setTurn(0);log("Your turn!")},DELAY.noMatch);
        },DELAY.flip);return}

      if(move===1){
        const idle=kn.filter(p=>p!==c1);
        const c2=idle.length>0?pick(idle):aa.filter(p=>p!==c1).length>0?pick(aa.filter(p=>p!==c1)):null;
        if(c2!==null){
          setTimeout(()=>{setFlipped(f=>new Set([...f,c2]));mem.observe(c2,cRef.current[c2]);setBotMemState(mem.clone());
            log(idle.includes(c2)?"Idle flip (known card).":"Second flip.");
            setTimeout(()=>{setFlipped(f=>{const nf=new Set(f);nf.delete(c1);nf.delete(c2);return nf});setBotBusy(false);setTurn(0);log("Your turn!")},DELAY.noMatch);
          },DELAY.flip);
        } else {setBotBusy(false);setTurn(0);log("Your turn!")}
      } else {
        const u2=unk.filter(p=>p!==c1);const c2=u2.length>0?pick(u2):pick(aa.filter(p=>p!==c1));
        setTimeout(()=>{setFlipped(f=>new Set([...f,c2]));const v2=cRef.current[c2];
          mem.observe(c2,v2);setBotMemState(mem.clone());
          if(v1===v2){log("Lucky match!");
            setTimeout(()=>{const nm=new Set(mRef.current);nm.add(c1);nm.add(c2);mRef.current=nm;setMatched(nm);setScores(s=>[s[0],s[1]+1]);
              mem.forgetPos(c1);mem.forgetPos(c2);setBotMemState(mem.clone());
              setHl(new Set([c1,c2]));setTimeout(()=>setHl(new Set()),500);
              if(nm.size===cRef.current.length){setGameOver(true);setBotBusy(false);log("Game over!")}
              else setBotBusy(false);
            },DELAY.match);
          } else {log("Two unknowns, no match.");
            setTimeout(()=>{setFlipped(f=>{const nf=new Set(f);nf.delete(c1);nf.delete(c2);return nf});setBotBusy(false);setTurn(0);log("Your turn!")},DELAY.noMatch);
          }
        },DELAY.flip);
      }
    },DELAY.think);
  },[turn,gameOver,botBusy,mode,aliveSet,log]);

  // Simulation
  const runSim=useCallback(()=>{
    setSimRunning(true);setSimRes(null);
    setTimeout(()=>{
      const t1=computeMoveTable(nPairs,b1s==="zwick"?null:b1sm);
      const t2=computeMoveTable(nPairs,b2s==="zwick"?null:b2sm);
      let p1w=0,p2w=0,dr=0,ts=[0,0];
      for(let i=0;i<simN;i++){const s=playHeadlessGame(nPairs,b1m,b2m,t1,t2);ts[0]+=s[0];ts[1]+=s[1];if(s[0]>s[1])p1w++;else if(s[1]>s[0])p2w++;else dr++}
      const dec=p1w+p2w;
      setSimRes({p1w,p2w,dr,g:simN,p1r:(p1w/simN*100).toFixed(1),p2r:(p2w/simN*100).toFixed(1),
        dR:(dr/simN*100).toFixed(1),p2c:dec>0?(p2w/dec*100).toFixed(1):"50.0",
        av:[(ts[0]/simN).toFixed(1),(ts[1]/simN).toFixed(1)]});
      setSimRunning(false);
    },50);
  },[nPairs,b1s,b1sm,b1m,b2s,b2sm,b2m,simN]);

  const cols=nPairs<=8?4:nPairs<=12?6:8;
  const memE=botMemState?botMemState.getEntries().filter(([p])=>!matched.has(p)):[];

  return(
    <div className="min-h-screen bg-stone-900 text-stone-100 p-3" style={{fontFamily:"'Courier New',monospace"}}>
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-2">
          <h1 className="text-xl font-bold tracking-wider text-amber-400">MEMORY</h1>
          <p className="text-stone-500 text-xs">Play against the optimal strategy — or pit two bots against each other</p>
        </div>

        <div className="flex flex-wrap gap-2 justify-center mb-2 text-xs">
          <Sel label="Mode:" value={mode} onChange={setMode} options={[["play","Play vs Bot"],["sim","Bot vs Bot"]]}/>
          <Sel label="Pairs:" value={nPairs} onChange={v=>setNPairs(+v)}
            options={[[6,"6 (12)"],[8,"8 (16)"],[10,"10 (20)"],[12,"12 (24)"],[16,"16 (32)"]]}/>
        </div>

        {mode==="play"&&(<>
          <div className="flex flex-wrap gap-1.5 justify-center mb-2 text-xs">
            <Sel label="Start:" value={humanStarts?"h":"b"} onChange={v=>setHumanStarts(v==="h")} options={[["h","You first"],["b","Bot first"]]}/>
            <Sel label="Strategy:" value={botStrat} onChange={setBotStrat} options={[["bounded","Bounded"],["zwick","Zwick (∞)"]]}/>
            {botStrat==="bounded"&&<label className="flex items-center gap-1 bg-stone-800 px-2 py-1 rounded text-xs">
              <span className="text-stone-400">Table M={botStratM}</span>
              <input type="range" min={3} max={20} value={botStratM} onChange={e=>setBotStratM(+e.target.value)} className="w-14 accent-amber-500"/>
            </label>}
            <label className="flex items-center gap-1 bg-stone-800 px-2 py-1 rounded text-xs">
              <span className="text-stone-400">Memory={botMem}</span>
              <input type="range" min={3} max={Math.max(20,nPairs*2)} value={botMem} onChange={e=>setBotMem(+e.target.value)} className="w-14 accent-rose-500"/>
            </label>
            <label className="flex items-center gap-1.5 bg-stone-800 px-2 py-1 rounded text-xs cursor-pointer">
              <input type="checkbox" checked={showHints} onChange={e=>setShowHints(e.target.checked)} className="accent-amber-500"/>
              <span className="text-stone-400">Show memory</span>
            </label>
            <button onClick={initGame} className="bg-amber-600 hover:bg-amber-500 text-stone-900 font-bold px-3 py-1 rounded text-xs">New Game</button>
          </div>

          <div className="flex justify-center gap-6 mb-3">
            <div className={`text-center px-3 py-1.5 rounded ${turn===0&&!gameOver?'bg-emerald-900 ring-1 ring-emerald-500':'bg-stone-800'}`}>
              <div className="text-xs text-stone-400">YOU</div>
              <div className="text-xl font-bold text-emerald-400">{scores[0]}</div>
            </div>
            <div className="flex items-center text-stone-500 text-xs">
              {gameOver?(scores[0]>scores[1]?"You win! 🎉":scores[1]>scores[0]?"Bot wins 🤖":"Draw!"):(turn===0?"Your turn":"Bot thinking...")}
            </div>
            <div className={`text-center px-3 py-1.5 rounded ${turn===1&&!gameOver?'bg-rose-900 ring-1 ring-rose-500':'bg-stone-800'}`}>
              <div className="text-xs text-stone-400">BOT</div>
              <div className="text-xl font-bold text-rose-400">{scores[1]}</div>
            </div>
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <div className="grid gap-1.5" style={{gridTemplateColumns:`repeat(${cols},1fr)`}}>
                {cards.map((val,idx)=>{
                  const isF=flipped.has(idx),isM=matched.has(idx),isH=hl.has(idx);
                  const bk=showHints&&botMemState&&botMemState.store.has(idx)&&!isM;
                  return(<button key={idx} onClick={()=>handleClick(idx)} disabled={isM||botBusy||turn!==0}
                    className={`aspect-square rounded-lg flex items-center justify-center transition-all duration-300 relative
                      ${isM?'opacity-15 scale-90':''} ${isH?'ring-2 ring-yellow-400 scale-105':''}
                      ${isF?'bg-stone-100 text-stone-900 scale-105':'bg-stone-700 hover:bg-stone-600 cursor-pointer'}
                      ${bk&&!isF?'ring-1 ring-rose-800':''}`}
                    style={{fontSize:nPairs>12?'1rem':'1.4rem'}}>
                    {(isF||isM)?val:"?"}
                    {bk&&!isF&&<div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-rose-500 rounded-full"/>}
                  </button>);
                })}
              </div>
            </div>
            {showHints&&<div className="w-36 flex-shrink-0 space-y-2">
              <div className="bg-stone-800 rounded p-2">
                <div className="text-xs text-stone-400 mb-1 flex justify-between"><span>Bot Memory</span><span className="text-rose-400">{memE.length}/{botMem}</span></div>
                <div className="space-y-0.5">
                  {Array.from({length:botMem},(_,i)=>{const e=memE[memE.length-1-i];
                    return(<div key={i} className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-xs ${e?'bg-stone-700':'border border-stone-700 border-dashed'}`}>
                      {e?(<><span className="text-sm">{cRef.current[e[0]]}</span><span className="text-stone-500">#{e[0]}</span>
                        {i===botMem-1&&memE.length>=botMem&&<span className="text-rose-600 ml-auto">⏳</span>}</>)
                        :(<span className="text-stone-600">—</span>)}
                    </div>);})}
                </div>
              </div>
              <div className="bg-stone-800 rounded p-2">
                <div className="text-xs text-stone-400 mb-1">Log</div>
                <div className="space-y-0 max-h-28 overflow-y-auto">
                  {gameLog.map((m,i)=><div key={i} className={`text-xs leading-tight ${i===0?'text-stone-300':'text-stone-600'}`}>{m}</div>)}
                </div>
              </div>
            </div>}
          </div>
        </>)}

        {mode==="sim"&&(<>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-3">
            <BotCfg label="🟢 Bot 1 (P1)" strat={b1s} setStrat={setB1s} stratM={b1sm} setStratM={setB1sm} mem={b1m} setMem={setB1m} nPairs={nPairs} color="border-emerald-600"/>
            <BotCfg label="🔴 Bot 2 (P2)" strat={b2s} setStrat={setB2s} stratM={b2sm} setStratM={setB2sm} mem={b2m} setMem={setB2m} nPairs={nPairs} color="border-rose-600"/>
          </div>
          <div className="flex items-center justify-center gap-3 mb-3">
            <Sel label="Games:" value={simN} onChange={v=>setSimN(+v)} options={[[100,"100"],[500,"500"],[1000,"1k"],[5000,"5k"],[10000,"10k"]]}/>
            <button onClick={runSim} disabled={simRunning}
              className="bg-amber-600 hover:bg-amber-500 disabled:bg-stone-600 text-stone-900 font-bold px-4 py-1.5 rounded text-xs">
              {simRunning?"Running...":"Run Simulation"}
            </button>
          </div>
          {simRes&&<div className="bg-stone-800 rounded-lg p-4 max-w-md mx-auto">
            <h3 className="text-sm font-bold text-amber-400 mb-3 text-center">Results ({simRes.g.toLocaleString()} games, n={nPairs})</h3>
            <div className="grid grid-cols-3 gap-2 text-center mb-3">
              <div className="bg-stone-700 rounded p-2"><div className="text-xs text-stone-400">Bot 1</div><div className="text-lg font-bold text-emerald-400">{simRes.p1r}%</div></div>
              <div className="bg-stone-700 rounded p-2"><div className="text-xs text-stone-400">Draw</div><div className="text-lg font-bold text-stone-400">{simRes.dR}%</div></div>
              <div className="bg-stone-700 rounded p-2"><div className="text-xs text-stone-400">Bot 2</div><div className="text-lg font-bold text-rose-400">{simRes.p2r}%</div></div>
            </div>
            <div className="text-center space-y-1 text-xs text-stone-400">
              <div>P(Bot 2 wins | decisive): <span className="text-stone-200 font-bold">{simRes.p2c}%</span></div>
              <div>Avg pairs: Bot 1={simRes.av[0]}, Bot 2={simRes.av[1]}</div>
            </div>
          </div>}
        </>)}
      </div>
    </div>
  );
}
