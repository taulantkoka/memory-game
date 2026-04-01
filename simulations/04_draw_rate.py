#!/usr/bin/env python3
"""
Draw rate vs memory capacity M under optimal play (both players
use the bounded-memory optimal strategy for their M).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fractions import Fraction
from joblib import Parallel, delayed
from collections import OrderedDict
import os, time

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
def savefig(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight')

# ═══════════════════════════════════════════════════════════════
# COMPUTE STRATEGY TABLES
# ═══════════════════════════════════════════════════════════════
def compute_bounded_values(N_max, M):
    if M is None: M = 2*N_max+10
    e = {(0,0): Fraction(0)}; opt = {}
    for n in range(1, N_max+1):
        if n <= M: e[(n,n)] = Fraction(n); opt[(n,n)] = 1
        start_k = min(n-1, M) if n <= M else M
        for k in range(start_k, -1, -1):
            p_den = 2*n-k
            if p_den == 0: continue
            p = Fraction(k, p_den); q = Fraction(2*(n-k), p_den)
            if k >= 1:
                if k < M: v1 = p*(1+e[(n-1,min(k-1,M))]) - q*e[(n,min(k+1,M))]
                else: v1 = p*(1+e[(n-1,M-1)])/(1+q) if (1+q)!=0 else Fraction(0)
            else: v1 = None
            d2 = 2*n-k-1
            if d2 > 0:
                k_prime=min(k+1,M); k_lucky=k if k<M else M-1
                k_auto=k if k<M else M-1; k_nm2=min(k+2,M)
                fl=Fraction(1,d2); fa=Fraction(k_prime-1,d2) if k_prime>=1 else Fraction(0)
                nk1=n-k-1; fn=Fraction(2*nk1,d2) if nk1>0 else Fraction(0)
                first=p*(1+e[(n-1,min(k-1,M))]) if k>=1 else Fraction(0)
                ik=fl*(1+e[(n-1,min(k_lucky,M))]); ia=fa*(1+e[(n-1,min(k_auto,M))])
                if k_nm2!=k and (n,k_nm2) in e:
                    v2=first+q*(ik-ia-fn*e[(n,k_nm2)])
                elif k<M: v2=first+q*(ik-ia-fn*e.get((n,k_nm2),Fraction(0)))
                else:
                    rhs=first+q*(ik-ia); denom2=1+q*fn
                    v2=rhs/denom2 if denom2!=0 else Fraction(0)
            elif d2==0:
                v2=Fraction(1)+(e.get((n-1,min(k-1 if k>=1 else 0,M)),Fraction(0)) if k>=1 else Fraction(0))
            else: v2=Fraction(0)
            v1v=v1 if v1 is not None else Fraction(-99999)
            if k==0: e[(n,k)]=v2; opt[(n,k)]=2
            elif k==1:
                if v1v>=v2: e[(n,k)]=v1v; opt[(n,k)]=1
                else: e[(n,k)]=v2; opt[(n,k)]=2
            else:
                if v1v>0 and v1v>=v2: e[(n,k)]=v1v; opt[(n,k)]=1
                elif v2>=0 and v2>=v1v: e[(n,k)]=v2; opt[(n,k)]=2
                elif v1v<=0 and v2<=0: e[(n,k)]=Fraction(0); opt[(n,k)]=0
                else: e[(n,k)]=max(v1v,v2); opt[(n,k)]=1 if v1v>v2 else 2
    return e, opt

N = 40
M_values = list(range(3, 21)) + [25, 30, 40]

print("Computing strategy tables...")
TABLES = {}
for M in M_values:
    _, opt = compute_bounded_values(N, M)
    TABLES[M] = opt
print(f"  {len(M_values)} tables computed")

# ═══════════════════════════════════════════════════════════════
# GAME ENGINE
# ═══════════════════════════════════════════════════════════════
class PlayerMemory:
    def __init__(self, cap, rng):
        self.cap=cap; self.store=OrderedDict(); self.rng=rng
    def observe(self, pos, value):
        if pos in self.store: self.store.move_to_end(pos); self.store[pos]=value; return
        while len(self.store)>=self.cap: self.store.popitem(last=False)
        self.store[pos]=value
    def find_value(self, value, exclude=None):
        for pos,val in self.store.items():
            if val==value and pos!=exclude: self.store.move_to_end(pos); return pos
        return None
    def known_alive(self, alive): return sum(1 for p in self.store if p in alive)
    def forget_pos(self, pos):
        if pos in self.store: del self.store[pos]

def play_game(n_pairs, M, table, seed):
    rng=np.random.default_rng(seed)
    cards=list(range(n_pairs))*2; rng.shuffle(cards)
    board=np.array(cards); alive=set(range(2*n_pairs))
    mem=[PlayerMemory(M, rng), PlayerMemory(M, rng)]
    scores=[0,0]

    def flip(pos):
        v=board[pos]; mem[0].observe(pos,v); mem[1].observe(pos,v); return v
    def remove(p1,p2,player):
        alive.discard(p1); alive.discard(p2)
        mem[0].forget_pos(p1); mem[0].forget_pos(p2)
        mem[1].forget_pos(p1); mem[1].forget_pos(p2)
        scores[player]+=1
    def pick_new(player, exclude=None):
        al=[p for p in alive if p!=exclude]
        if not al: return None
        unk=[p for p in al if p not in mem[player].store]
        return rng.choice(unk) if unk else rng.choice(al)
    def pick_known(player, exclude=None):
        al=[p for p in alive if p!=exclude]
        if not al: return None
        kn=[p for p in al if p in mem[player].store]
        return rng.choice(kn) if kn else rng.choice(al)
    def try_match(player, pos1, val1):
        mp=mem[player].find_value(val1, exclude=pos1)
        if mp is not None and mp in alive:
            flip(mp)
            if board[mp]==val1: remove(pos1,mp,player); return True
        return False
    def auto_take(player, val, pos):
        opp=1-player; om=mem[opp].find_value(val, exclude=pos)
        if om is not None and om in alive:
            flip(om)
            if board[om]==val: remove(pos,om,opp)

    cur=0; passes=0
    for _ in range(50000):
        n=len(alive)//2
        if n==0: break
        k=min(mem[cur].known_alive(alive), M)
        move=table.get((n,k), 2)
        if move==0:
            passes+=1
            if passes>=4: break
            cur=1-cur; continue
        passes=0
        if move==1:
            pos1=pick_new(cur)
            if pos1 is None: break
            val1=flip(pos1)
            if try_match(cur,pos1,val1): continue
            idle=pick_known(cur,pos1)
            if idle is not None: flip(idle)
            cur=1-cur
        elif move==2:
            pos1=pick_new(cur)
            if pos1 is None: break
            val1=flip(pos1)
            if try_match(cur,pos1,val1): continue
            pos2=pick_new(cur,pos1)
            if pos2 is None: cur=1-cur; continue
            val2=flip(pos2)
            if val2==val1: remove(pos1,pos2,cur); continue
            auto_take(cur,val2,pos2)
            cur=1-cur

    s0,s1=scores; w=0 if s0>s1 else (1 if s1>s0 else -1)
    return w, s0, s1

def measure(n, M, table, ng, seed):
    seeds=[np.random.SeedSequence(seed).spawn(ng)]
    ints=[s.generate_state(1)[0] for s in seeds[0]]
    res=Parallel(n_jobs=1, backend='loky')(
        delayed(play_game)(n, M, table, s) for s in ints)
    p1w,p2w,dr=0,0,0
    for w,s0,s1 in res:
        if w==0: p1w+=1
        elif w==1: p2w+=1
        else: dr+=1
    dec=p1w+p2w
    return {
        'p1_wr':p1w/ng, 'p2_wr':p2w/ng, 'draws':dr/ng,
        'p2_cond':p2w/dec if dec>0 else .5,
    }

def run_point(n, M, table, ng, seed, label):
    return (label, n, M, measure(n, M, table, ng, seed))

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT: Draw rate vs M for each board size
# ═══════════════════════════════════════════════════════════════
ng = 100000
board_sizes = [8, 12, 16, 24, 36]

jobs = []
for n in board_sizes:
    for M in M_values:
        seed = 42000*n + 100*M
        jobs.append((n, M, TABLES[M], ng, seed, f"n={n} M={M}"))

print(f"\nRunning {len(jobs)} points × {ng} games...")
t0 = time.time()
results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(run_point)(*j) for j in jobs)
print(f"Done in {time.time()-t0:.1f}s")

# Organize
data = {}
for label, n, M, r in results:
    data.setdefault(n, {'Ms':[], 'draws':[], 'p2c':[]})
    data[n]['Ms'].append(M)
    data[n]['draws'].append(r['draws'])
    data[n]['p2c'].append(r['p2_cond'])

for n in data:
    order = np.argsort(data[n]['Ms'])
    for field in data[n]:
        data[n][field] = np.array(data[n][field])[order]

# Print
for n in board_sizes:
    print(f"\nn={n}:")
    for i, M in enumerate(data[n]['Ms']):
        print(f"  M={M:3.0f}: draws={data[n]['draws'][i]:.3f}  P2|dec={data[n]['p2c'][i]:.3f}")

# ═══════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(board_sizes)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f'Game Properties vs Memory Capacity (optimal play, {ng} games/point)',
             fontsize=14, fontweight='bold')

for idx, n in enumerate(board_sizes):
    d = data[n]
    ax1.plot(d['Ms'], d['draws'], '-o', color=colors[idx], ms=5, lw=2,
             label=f'n={n} ({2*n} cards)')
    ax2.plot(d['Ms'], d['p2c'], '-o', color=colors[idx], ms=5, lw=2,
             label=f'n={n} ({2*n} cards)')

ax1.axvspan(5, 9, alpha=0.08, color='red', label="Miller's 7±2")
ax1.set_xlabel('Memory capacity M (card positions)', fontsize=12)
ax1.set_ylabel('Draw rate', fontsize=12)
ax1.set_title('How Often Does the Game Tie?', fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.2)
ax1.set_xscale('log')
ax1.set_xticks([3,5,7,10,15,20,30,40])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax2.axhline(0.5, color='black', ls='--', alpha=0.5)
ax2.axvspan(5, 9, alpha=0.08, color='red', label="Miller's 7±2")
ax2.set_xlabel('Memory capacity M (card positions)', fontsize=12)
ax2.set_ylabel('P(P2 wins | decisive)', fontsize=12)
ax2.set_title('Who Wins (Under Optimal Play)?', fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)
ax2.set_ylim(0.46, 0.54)
ax2.set_xscale('log')
ax2.set_xticks([3,5,7,10,15,20,30,40])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.tight_layout(rect=[0,0,1,0.93])
savefig('draw_rate_vs_capacity.png')
print("\nSaved draw_rate_vs_capacity.png")

print("\n" + "="*70)
print("ALL DONE")
print("="*70)