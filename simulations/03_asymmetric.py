#!/usr/bin/env python3
"""
Asymmetric Memory: What if players have different capacities?
Each player uses their own Bounded-optimal strategy (computed for their M).
Each player has their own memory (capacity M1 vs M2).

NOTE: With asymmetric memory, players no longer have identical memory.
This means the shared-memory proof doesn't apply. However, both players
still observe all flips — the difference is only in what they retain.
We use each player's own optimal table as a reasonable (not provably optimal)
strategy.
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
# COMPUTE STRATEGY TABLES FOR EACH M
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
            d2 = (2*n-M) if k == M else (2*n-k-1)
            if d2 > 0:
                k_prime=min(k+1,M); k_lucky=k if k<M else M-1
                k_auto=k if k<M else M-1; k_nm2=min(k+2,M)
                fl=Fraction(1,d2); fa=Fraction(k_prime-1,d2) if k_prime>=1 else Fraction(0)
                nk1=(n-M) if k == M else (n-k-1); fn=Fraction(2*nk1,d2) if nk1>0 else Fraction(0)
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
M_values = list(range(3, 21))  # M=3 to M=20

print("Computing strategy tables for each M...")
TABLES = {}
VALUES = {}
for M in M_values:
    e, opt = compute_bounded_values(N, M)
    TABLES[M] = opt
    VALUES[M] = e
    print(f"  M={M:2d}: done")

# ═══════════════════════════════════════════════════════════════
# GAME ENGINE WITH ASYMMETRIC MEMORY
# ═══════════════════════════════════════════════════════════════
class PlayerMemory:
    def __init__(self, cap, rng):
        self.cap=cap; self.rng=rng; self.store=OrderedDict()
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

def play_game(n_pairs, M1, M2, table1, table2, seed):
    rng=np.random.default_rng(seed)
    cards=list(range(n_pairs))*2; rng.shuffle(cards)
    board=np.array(cards); alive=set(range(2*n_pairs))
    mem=[PlayerMemory(M1, rng), PlayerMemory(M2, rng)]
    scores=[0,0]; tables=[table1,table2]

    def flip(pos):
        v=board[pos]
        mem[0].observe(pos,v); mem[1].observe(pos,v)
        return v
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
        k=min(mem[cur].known_alive(alive), mem[cur].cap)
        move=tables[cur].get((n,k), 2)

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

def measure(n, M1, M2, t1, t2, ng, seed):
    seeds=[np.random.SeedSequence(seed).spawn(ng)]
    ints=[s.generate_state(1)[0] for s in seeds[0]]
    res=Parallel(n_jobs=1,backend='loky')(
        delayed(play_game)(n,M1,M2,t1,t2,s) for s in ints)
    p1w,p2w,dr=0,0,0; ts=[0,0]
    for w,s0,s1 in res:
        ts[0]+=s0; ts[1]+=s1
        if w==0: p1w+=1
        elif w==1: p2w+=1
        else: dr+=1
    dec=p1w+p2w
    return {
        'p1_wr':p1w/ng, 'p2_wr':p2w/ng, 'draws':dr/ng,
        'p2_cond':p2w/dec if dec>0 else .5,
        'gain':(ts[0]-ts[1])/ng, 'avg':[ts[0]/ng, ts[1]/ng],
    }

def run_point(n, M1, M2, t1, t2, ng, seed, label):
    return (label, n, M1, M2, measure(n, M1, M2, t1, t2, ng, seed))

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Full M1 × M2 matrix at fixed board sizes
# ═══════════════════════════════════════════════════════════════
ng = 100000
test_Ms = [3, 5, 7, 9, 12, 20]
board_sizes = [8, 12, 16, 24]

print(f"\nEXP 1: Asymmetric memory matchup matrices")
print(f"  {len(test_Ms)}×{len(test_Ms)} capacity pairs × {len(board_sizes)} boards × {ng} games")

jobs1 = []
for n in board_sizes:
    for M1 in test_Ms:
        for M2 in test_Ms:
            seed = 10000*n + 100*M1 + M2
            jobs1.append((n, M1, M2, TABLES[M1], TABLES[M2], ng, seed,
                         f"n={n} M1={M1} M2={M2}"))

t0 = time.time()
res1 = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(run_point)(*j) for j in jobs1)
print(f"Done in {time.time()-t0:.1f}s")

# Organize into matrices
matrices = {}
for label, n, M1, M2, r in res1:
    matrices.setdefault(n, {})[(M1, M2)] = r

# Print
for n in board_sizes:
    print(f"\nn={n} ({2*n} cards): P1 expected gain")
    print(f"  {'P1\\P2':>8s}", end="")
    for M2 in test_Ms: print(f"  M2={M2:2d}", end="")
    print()
    for M1 in test_Ms:
        print(f"  M1={M1:2d}  ", end="")
        for M2 in test_Ms:
            r = matrices[n][(M1, M2)]
            print(f"  {r['gain']:+6.2f}", end="")
        print()

    print(f"\n  P(P2 wins | decisive)")
    print(f"  {'P1\\P2':>8s}", end="")
    for M2 in test_Ms: print(f"  M2={M2:2d}", end="")
    print()
    for M1 in test_Ms:
        print(f"  M1={M1:2d}  ", end="")
        for M2 in test_Ms:
            r = matrices[n][(M1, M2)]
            print(f"  {r['p2_cond']:6.3f}", end="")
        print()

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: How much is one extra slot worth?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXP 2: Value of one extra memory slot")
print(f"{'='*70}")

fine_Ms = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
jobs2 = []
for n in [8, 12, 16, 24]:
    for M_base in fine_Ms:
        if M_base + 1 > 20: continue
        M_better = M_base + 1
        if M_better not in TABLES: continue
        # Better memory as P1
        jobs2.append((n, M_better, M_base, TABLES[M_better],
                      TABLES[M_base], ng, 20000*n+100*M_base,
                      f"n={n} P1:M={M_better} vs P2:M={M_base}"))
        # Better memory as P2
        jobs2.append((n, M_base, M_better, TABLES[M_base],
                      TABLES[M_better], ng, 30000*n+100*M_base,
                      f"n={n} P1:M={M_base} vs P2:M={M_better}"))

t0 = time.time()
res2 = Parallel(n_jobs=-1, backend='loky', verbose=5)(
    delayed(run_point)(*j) for j in jobs2)
print(f"Done in {time.time()-t0:.1f}s")

for label, n, M1, M2, r in res2:
    better = "P1" if M1 > M2 else "P2"
    print(f"  {label:<40s} gain={r['gain']:+.3f} P2|dec={r['p2_cond']:.3f} "
          f"→ {better} (M={max(M1,M2)}) advantage")

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

# ── Fig 1: Heatmap matrices ──
fig1, axes1 = plt.subplots(1, len(board_sizes), figsize=(6*len(board_sizes), 5))
fig1.suptitle(f'Asymmetric Memory: P1 Expected Gain\n'
              f'Each player uses their own Bounded-optimal strategy | {ng} games',
              fontsize=14, fontweight='bold')

for idx, n in enumerate(board_sizes):
    ax = axes1[idx]
    mat = np.zeros((len(test_Ms), len(test_Ms)))
    for i, M1 in enumerate(test_Ms):
        for j, M2 in enumerate(test_Ms):
            mat[i, j] = matrices[n][(M1, M2)]['gain']

    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_xticks(range(len(test_Ms)))
    ax.set_yticks(range(len(test_Ms)))
    ax.set_xticklabels([str(m) for m in test_Ms], fontsize=9)
    ax.set_yticklabels([str(m) for m in test_Ms], fontsize=9)
    ax.set_xlabel('P2 memory M₂')
    ax.set_ylabel('P1 memory M₁')
    ax.set_title(f'n={n} ({2*n} cards)', fontweight='bold')

    for i in range(len(test_Ms)):
        for j in range(len(test_Ms)):
            val = mat[i, j]
            color = 'white' if abs(val) > vmax * 0.4 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                   fontsize=8, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='P1 gain (pairs)')

plt.tight_layout(rect=[0, 0, 1, 0.92])
savefig('asymmetric_gain_heatmap.png')
print("\nSaved asymmetric_gain_heatmap.png")

# ── Fig 2: Conditional P2 win rate heatmaps ──
fig2, axes2 = plt.subplots(1, len(board_sizes), figsize=(6*len(board_sizes), 5))
fig2.suptitle(f'Asymmetric Memory: P(P2 Wins | Decisive)\n'
              f'{ng} games per cell',
              fontsize=14, fontweight='bold')

for idx, n in enumerate(board_sizes):
    ax = axes2[idx]
    mat = np.zeros((len(test_Ms), len(test_Ms)))
    for i, M1 in enumerate(test_Ms):
        for j, M2 in enumerate(test_Ms):
            mat[i, j] = matrices[n][(M1, M2)]['p2_cond']

    im = ax.imshow(mat, cmap='RdBu', vmin=0.3, vmax=0.7, aspect='equal')
    ax.set_xticks(range(len(test_Ms)))
    ax.set_yticks(range(len(test_Ms)))
    ax.set_xticklabels([str(m) for m in test_Ms], fontsize=9)
    ax.set_yticklabels([str(m) for m in test_Ms], fontsize=9)
    ax.set_xlabel('P2 memory M₂')
    ax.set_ylabel('P1 memory M₁')
    ax.set_title(f'n={n} ({2*n} cards)', fontweight='bold')

    for i in range(len(test_Ms)):
        for j in range(len(test_Ms)):
            val = mat[i, j]
            color = 'white' if abs(val - 0.5) > 0.1 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   fontsize=7, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='P(P2 wins | decisive)')

plt.tight_layout(rect=[0, 0, 1, 0.92])
savefig('asymmetric_p2cond_heatmap.png')
print("Saved asymmetric_p2cond_heatmap.png")

# ── Fig 3: Memory advantage curve ──
fig3, ax3 = plt.subplots(figsize=(12, 7))

# For each board size, plot P1 win rate when P1 has M+Δ vs P2 has M=7
for idx, n in enumerate(board_sizes):
    Ms_p1 = []
    p1_gains = []
    for M1 in test_Ms:
        r = matrices[n].get((M1, 7))
        if r:
            Ms_p1.append(M1)
            p1_gains.append(r['gain'])

    ax3.plot(Ms_p1, p1_gains, '-o', ms=7, lw=2,
             label=f'n={n} ({2*n} cards)')

ax3.axhline(0, color='black', ls='--', alpha=0.5)
ax3.axvline(7, color='red', ls=':', alpha=0.3, lw=2, label="P2 has M=7")
ax3.set_xlabel('P1 memory capacity M₁ (P2 fixed at M₂=7)', fontsize=12)
ax3.set_ylabel('P1 expected gain (pairs)', fontsize=12)
ax3.set_title('How Much Does Better Memory Help?\n'
              'P1 varies, P2 fixed at M=7, both play Bounded-optimal',
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=11); ax3.grid(True, alpha=0.2)
plt.tight_layout()
savefig('asymmetric_memory_advantage.png')
print("Saved asymmetric_memory_advantage.png")

print("\n" + "="*70)
print("ALL DONE")
print("="*70)