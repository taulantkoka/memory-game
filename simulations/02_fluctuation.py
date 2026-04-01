#!/usr/bin/env python3
"""
Miller-Optimal vs Zwick with fluctuating memory capacity.
Strategy tables: fixed (Miller@M=7, Zwick@M=∞).
Actual capacity each turn: M_base + Uniform(-σ, +σ), clamped to [2, M_base+σ].
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
# MOVE TABLES
# ═══════════════════════════════════════════════════════════════
def compute_miller_values(N_max, M):
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
                k_prime = min(k+1,M)
                k_lucky = k if k < M else M-1
                k_auto = k if k < M else M-1
                k_nm2 = min(k+2,M)
                fl = Fraction(1,d2)
                fa = Fraction(k_prime-1,d2) if k_prime>=1 else Fraction(0)
                nk1 = n-k-1
                fn = Fraction(2*nk1,d2) if nk1>0 else Fraction(0)
                first = p*(1+e[(n-1,min(k-1,M))]) if k>=1 else Fraction(0)
                ik = fl*(1+e[(n-1,min(k_lucky,M))])
                ia = fa*(1+e[(n-1,min(k_auto,M))])
                if k_nm2 != k and (n,k_nm2) in e:
                    v2 = first + q*(ik-ia-fn*e[(n,k_nm2)])
                elif k < M:
                    v2 = first + q*(ik-ia-fn*e.get((n,k_nm2),Fraction(0)))
                else:
                    rhs = first+q*(ik-ia); denom2=1+q*fn
                    v2 = rhs/denom2 if denom2!=0 else Fraction(0)
            elif d2==0:
                v2 = Fraction(1)+(e.get((n-1,min(k-1 if k>=1 else 0,M)),Fraction(0)) if k>=1 else Fraction(0))
            else: v2 = Fraction(0)
            v1v = v1 if v1 is not None else Fraction(-99999)
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

print("Computing move tables...")
_, MILLER_OPT = compute_miller_values(40, 7)
_, ZWICK_OPT = compute_miller_values(40, None)

# ═══════════════════════════════════════════════════════════════
# GAME ENGINE WITH FLUCTUATING CAPACITY
# ═══════════════════════════════════════════════════════════════
class FluctuatingMemory:
    def __init__(self, base, sigma, rng):
        self.base=base; self.sigma=sigma; self.rng=rng
        self.store=OrderedDict(); self.cap=base
    def fluctuate(self):
        if self.sigma > 0:
            self.cap = max(2, self.base + self.rng.integers(-self.sigma, self.sigma+1))
        while len(self.store) > self.cap:
            self.store.popitem(last=False)
    def observe(self, pos, value):
        if pos in self.store: self.store.move_to_end(pos); self.store[pos]=value; return
        while len(self.store) >= self.cap: self.store.popitem(last=False)
        self.store[pos] = value
    def find_value(self, value, exclude=None):
        for pos, val in self.store.items():
            if val==value and pos!=exclude: self.store.move_to_end(pos); return pos
        return None
    def known_alive(self, alive): return sum(1 for p in self.store if p in alive)
    def forget_pos(self, pos):
        if pos in self.store: del self.store[pos]

def play_game(n_pairs, base_cap, sigma, t1, t2, seed):
    rng = np.random.default_rng(seed)
    cards = list(range(n_pairs))*2; rng.shuffle(cards)
    board = np.array(cards); alive = set(range(2*n_pairs))
    mem = [FluctuatingMemory(base_cap, sigma, rng),
           FluctuatingMemory(base_cap, sigma, rng)]
    scores=[0,0]; last_matcher=-1; tables=[t1,t2]

    def flip(pos):
        v=board[pos]; mem[0].observe(pos,v); mem[1].observe(pos,v); return v
    def remove(p1,p2,player):
        nonlocal last_matcher
        alive.discard(p1); alive.discard(p2)
        mem[0].forget_pos(p1); mem[0].forget_pos(p2)
        mem[1].forget_pos(p1); mem[1].forget_pos(p2)
        scores[player]+=1; last_matcher=player
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
        mem[cur].fluctuate()
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
    return w, last_matcher, s0, s1

def measure(n, base, sigma, t1, t2, ng, seed):
    seeds=[np.random.SeedSequence(seed).spawn(ng)]
    ints=[s.generate_state(1)[0] for s in seeds[0]]
    res=Parallel(n_jobs=1,backend='loky')(
        delayed(play_game)(n,base,sigma,t1,t2,s) for s in ints)
    p1w,p2w,dr=0,0,0; ts=[0,0]
    for w,lm,s0,s1 in res:
        ts[0]+=s0; ts[1]+=s1
        if w==0: p1w+=1
        elif w==1: p2w+=1
        else: dr+=1
    dec=p1w+p2w
    return {
        'p1_wr':p1w/ng, 'p2_wr':p2w/ng, 'draws':dr/ng,
        'p2_cond':p2w/dec if dec>0 else .5,
        'p2_cond_err':1.96*np.sqrt(max((p2w/dec)*(1-p2w/dec),1e-4)/dec) if dec>10 else .1,
        'gain':(ts[0]-ts[1])/ng,
    }

def run_point(n, base, sigma, t1, t2, ng, seed, label):
    return (label, n, sigma, measure(n, base, sigma, t1, t2, ng, seed))

# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════
ng = 10000
base = 7
board_sizes = [8, 12, 16, 24, 36]
sigmas = [0, 1, 2, 3, 4, 5]

matchups = [
    ("Miller v Miller", MILLER_OPT, MILLER_OPT),
    ("Zwick v Zwick", ZWICK_OPT, ZWICK_OPT),
    ("P1:Zwick v P2:Miller", ZWICK_OPT, MILLER_OPT),
    ("P1:Miller v P2:Zwick", MILLER_OPT, ZWICK_OPT),
]

all_jobs = []
for n in board_sizes:
    for sigma in sigmas:
        for label, t1, t2 in matchups:
            seed = 50000*n + 1000*sigma + hash(label)%10000
            all_jobs.append((n, base, sigma, t1, t2, ng, seed,
                            f"n={n} σ={sigma} {label}"))

print(f"Running {len(all_jobs)} matchups × {ng} games...")
t0 = time.time()
all_res = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(run_point)(*j) for j in all_jobs)
print(f"Done in {time.time()-t0:.1f}s")

# Organize: data[(n, matchup)] = {sigmas: [...], p2c: [...], ...}
data = {}
for label, n, sigma, r in all_res:
    parts = label.split(" ", 2)
    mname = parts[2]
    key = (n, mname)
    data.setdefault(key, {'sigmas':[], 'p2c':[], 'p2ce':[], 'gain':[]})
    data[key]['sigmas'].append(sigma)
    data[key]['p2c'].append(r['p2_cond'])
    data[key]['p2ce'].append(r['p2_cond_err'])
    data[key]['gain'].append(r['gain'])

# Sort
for key in data:
    order = np.argsort(data[key]['sigmas'])
    for field in data[key]:
        data[key][field] = np.array(data[key][field])[order]

# Print summary
for n in board_sizes:
    print(f"\nn={n}:")
    for sigma in sigmas:
        print(f"  σ={sigma}:")
        for mname, _, _ in matchups:
            key = (n, mname)
            idx = list(data[key]['sigmas']).index(sigma)
            r_p2c = data[key]['p2c'][idx]
            r_gain = data[key]['gain'][idx]
            print(f"    {mname:<30s} P2|dec={r_p2c:.3f}  gain={r_gain:+.3f}")

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
colors_m = {'Miller v Miller':'#e7298a', 'Zwick v Zwick':'#2166ac',
            'P1:Zwick v P2:Miller':'#d6604d', 'P1:Miller v P2:Zwick':'#4dac26'}

# ── Fig 1: P2 conditional win rate vs σ, per board size ──
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle('Effect of Memory Fluctuation on Miller vs Zwick\n'
              f'Base capacity M₀=7, capacity each turn = 7 ± σ | {ng} games/point',
              fontsize=14, fontweight='bold')

for idx, n in enumerate(board_sizes):
    ax = axes1[idx//3][idx%3]
    for mname in colors_m:
        key = (n, mname)
        if key not in data: continue
        d = data[key]
        ax.plot(d['sigmas'], d['p2c'], '-o', color=colors_m[mname],
                ms=6, lw=2, label=mname if idx==0 else None)
    ax.axhline(0.5, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Fluctuation σ')
    ax.set_ylabel('P(P2 wins | decisive)')
    ax.set_title(f'n={n} ({2*n} cards)', fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.40, 0.60)

# Legend in last panel
ax_leg = axes1[1][2]
handles = [plt.Line2D([0],[0],color=colors_m[m],lw=2,label=m) for m in colors_m]
ax_leg.legend(handles=handles, fontsize=10, loc='center')
ax_leg.axis('off')

plt.tight_layout(rect=[0,0,1,0.93])
savefig('fluctuation_miller_vs_zwick.png')
print("\nSaved fluctuation_miller_vs_zwick.png")

# ── Fig 2: Miller advantage (Miller_v_Miller - Zwick_v_Zwick) vs σ ──
fig2, ax2 = plt.subplots(figsize=(12, 7))
board_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(board_sizes)))

for idx, n in enumerate(board_sizes):
    key_mm = (n, 'Miller v Miller')
    key_zz = (n, 'Zwick v Zwick')
    if key_mm not in data or key_zz not in data: continue
    diff = data[key_mm]['p2c'] - data[key_zz]['p2c']
    ax2.plot(data[key_mm]['sigmas'], diff, '-o', color=board_colors[idx],
             ms=7, lw=2, label=f'n={n} ({2*n} cards)')

ax2.axhline(0, color='black', ls='--', lw=1.5, alpha=0.5,
            label='Equal (Miller = Zwick)')
ax2.set_xlabel('Memory fluctuation σ (capacity = 7 ± σ)', fontsize=12)
ax2.set_ylabel('P2 cond. win rate: Miller minus Zwick', fontsize=12)
ax2.set_title('Does Fluctuation Change Which Strategy Is Better?\n'
              '> 0 means Miller gives P2 more advantage',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2)
plt.tight_layout()
savefig('fluctuation_advantage_diff.png')
print("Saved fluctuation_advantage_diff.png")

# ── Fig 3: Cross-strategy gain vs σ ──
fig3, ax3 = plt.subplots(figsize=(12, 7))

for idx, n in enumerate(board_sizes):
    # How much does Miller player gain over Zwick player?
    key_mz = (n, 'P1:Miller v P2:Zwick')
    key_zm = (n, 'P1:Zwick v P2:Miller')
    if key_mz not in data: continue
    # Miller advantage = gain when playing Miller as P1 against Zwick P2
    ax3.plot(data[key_mz]['sigmas'], data[key_mz]['gain'],
             '-o', color=board_colors[idx], ms=6, lw=2,
             label=f'n={n}: Miller P1 gain vs Zwick P2')
    ax3.plot(data[key_zm]['sigmas'], [-g for g in data[key_zm]['gain']],
             '--s', color=board_colors[idx], ms=5, lw=1.5, alpha=0.7,
             label=f'n={n}: Miller P2 gain vs Zwick P1')

ax3.axhline(0, color='black', ls='--', alpha=0.5)
ax3.set_xlabel('Fluctuation σ', fontsize=12)
ax3.set_ylabel('Miller player expected gain over Zwick player', fontsize=12)
ax3.set_title('How Much Does Miller Beat Zwick? (by board size and fluctuation)',
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=8, ncol=2); ax3.grid(True, alpha=0.2)
plt.tight_layout()
savefig('fluctuation_cross_gain.png')
print("Saved fluctuation_cross_gain.png")

print("\n" + "="*70)
print("ALL DONE")
print("="*70)