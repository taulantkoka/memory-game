#!/usr/bin/env python3
"""
Head-to-head: Miller-optimal strategy vs Zwick strategy
Both players have Miller memory M=7, but one uses the optimal
moves for M=7, the other uses Zwick's moves (designed for M=∞).

The difference: at k=3,5,7 Zwick says "flip two", Miller says "flip one".
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
# PRECOMPUTE BOTH MOVE TABLES
# ═══════════════════════════════════════════════════════════════

def compute_miller_values(N_max, M):
    if M is None: M = 2 * N_max + 10
    e = {(0,0): Fraction(0)}; opt = {}
    for n in range(1, N_max+1):
        if n <= M: e[(n,n)] = Fraction(n); opt[(n,n)] = 1
        start_k = min(n-1, M) if n <= M else M
        for k in range(start_k, -1, -1):
            p_den = 2*n-k
            if p_den == 0: continue
            p = Fraction(k, p_den); q = Fraction(2*(n-k), p_den)
            # e^1
            if k >= 1:
                k_nm1 = min(k+1, M)
                if k < M:
                    v1 = p*(1+e[(n-1, min(k-1,M))]) - q*e[(n, k_nm1)]
                else:
                    denom1 = 1 + q
                    v1 = p*(1+e[(n-1, M-1)])/denom1 if denom1 != 0 else Fraction(0)
            else: v1 = None
            # e^2
            d2 = 2*n-k-1
            if d2 > 0:
                k_prime = min(k+1, M)
                if k < M: k_lucky = k
                else: k_lucky = M-1
                if k < M: k_auto = k
                else: k_auto = M-1
                k_nm2 = min(k+2, M)
                fl = Fraction(1, d2)
                fa = Fraction(k_prime-1, d2) if k_prime >= 1 else Fraction(0)
                nk1 = n-k-1
                fn = Fraction(2*nk1, d2) if nk1 > 0 else Fraction(0)
                first = p*(1+e[(n-1, min(k-1,M))]) if k >= 1 else Fraction(0)
                ik = fl*(1+e[(n-1, min(k_lucky,M))])
                ia = fa*(1+e[(n-1, min(k_auto,M))])
                if k_nm2 <= M and (n, k_nm2) in e and k_nm2 != k:
                    v2 = first + q*(ik - ia - fn*e[(n, k_nm2)])
                elif k < M:
                    v2 = first + q*(ik - ia - fn*e.get((n, k_nm2), Fraction(0)))
                else:
                    rhs = first + q*(ik - ia)
                    denom2 = 1 + q*fn
                    v2 = rhs/denom2 if denom2 != 0 else Fraction(0)
            elif d2 == 0:
                v2 = Fraction(1) + e.get((n-1, min(k-1 if k>=1 else 0, M)), Fraction(0)) if k >= 1 else Fraction(1)
            else: v2 = Fraction(0)
            # Choose
            v1v = v1 if v1 is not None else Fraction(-99999)
            if k == 0: e[(n,k)] = v2; opt[(n,k)] = 2
            elif k == 1:
                if v1v >= v2: e[(n,k)] = v1v; opt[(n,k)] = 1
                else: e[(n,k)] = v2; opt[(n,k)] = 2
            else:
                if v1v > 0 and v1v >= v2: e[(n,k)] = v1v; opt[(n,k)] = 1
                elif v2 >= 0 and v2 >= v1v: e[(n,k)] = v2; opt[(n,k)] = 2
                elif v1v <= 0 and v2 <= 0: e[(n,k)] = Fraction(0); opt[(n,k)] = 0
                else: e[(n,k)] = max(v1v, v2); opt[(n,k)] = 1 if v1v > v2 else 2
    return e, opt

N = 25
print("Computing move tables...")
_, MILLER_OPT = compute_miller_values(N, 7)
_, ZWICK_OPT = compute_miller_values(N, None)

# Print comparison for n=12
print("\nn=12 moves comparison:")
print(f"  {'k':>3s}  Miller  Zwick")
for k in range(13):
    m = MILLER_OPT.get((12, k), '-')
    z = ZWICK_OPT.get((12, k), '-')
    diff = " ← DIFFERENT" if m != z and m != '-' and z != '-' else ""
    mk = min(k, 7)
    print(f"  k={k:2d}:   {m}       {z}{diff}")

# ═══════════════════════════════════════════════════════════════
# GAME ENGINE
# ═══════════════════════════════════════════════════════════════

class MillerMemory:
    def __init__(self, cap, rng):
        self.cap = cap; self.rng = rng; self.store = OrderedDict()
    def observe(self, pos, value):
        if pos in self.store: self.store.move_to_end(pos); self.store[pos] = value; return
        while len(self.store) >= self.cap: self.store.popitem(last=False)
        self.store[pos] = value
    def find_value(self, value, exclude=None):
        for pos, val in self.store.items():
            if val == value and pos != exclude: self.store.move_to_end(pos); return pos
        return None
    def known_alive(self, alive): return sum(1 for p in self.store if p in alive)
    def forget_pos(self, pos):
        if pos in self.store: del self.store[pos]

def play_game(n_pairs, M, move_table_p1, move_table_p2, seed):
    rng = np.random.default_rng(seed)
    cards = list(range(n_pairs)) * 2; rng.shuffle(cards)
    board = np.array(cards)
    alive = set(range(2*n_pairs))
    mem = [MillerMemory(M, rng), MillerMemory(M, rng)]
    scores = [0, 0]; last_matcher = -1
    tables = [move_table_p1, move_table_p2]

    def flip(pos):
        v = board[pos]
        mem[0].observe(pos, v); mem[1].observe(pos, v)
        return v

    def remove(p1, p2, player):
        nonlocal last_matcher
        alive.discard(p1); alive.discard(p2)
        mem[0].forget_pos(p1); mem[0].forget_pos(p2)
        mem[1].forget_pos(p1); mem[1].forget_pos(p2)
        scores[player] += 1; last_matcher = player

    def pick_new(player, exclude=None):
        al = [p for p in alive if p != exclude]
        if not al: return None
        unk = [p for p in al if p not in mem[player].store]
        return rng.choice(unk) if unk else rng.choice(al)

    def pick_known(player, exclude=None):
        al = [p for p in alive if p != exclude]
        if not al: return None
        kn = [p for p in al if p in mem[player].store]
        return rng.choice(kn) if kn else rng.choice(al)

    def try_match(player, pos1, val1):
        mp = mem[player].find_value(val1, exclude=pos1)
        if mp is not None and mp in alive:
            flip(mp)
            if board[mp] == val1: remove(pos1, mp, player); return True
        return False

    def auto_take_opp(player, val, pos):
        opp = 1 - player
        om = mem[opp].find_value(val, exclude=pos)
        if om is not None and om in alive:
            flip(om)
            if board[om] == val: remove(pos, om, opp)

    cur = 0; passes = 0
    for _ in range(50000):
        n = len(alive) // 2
        if n == 0: break
        k = min(mem[cur].known_alive(alive), M)
        # Clamp k for table lookup
        move = tables[cur].get((n, k))
        if move is None:
            # Fallback: if state not in table, use 2-move
            move = 2

        if move == 0:
            passes += 1
            if passes >= 4: break
            cur = 1 - cur; continue
        passes = 0

        if move == 1:
            pos1 = pick_new(cur)
            if pos1 is None: break
            val1 = flip(pos1)
            if try_match(cur, pos1, val1): continue  # matched, go again
            idle = pick_known(cur, pos1)
            if idle is not None: flip(idle)
            cur = 1 - cur

        elif move == 2:
            pos1 = pick_new(cur)
            if pos1 is None: break
            val1 = flip(pos1)
            if try_match(cur, pos1, val1): continue
            pos2 = pick_new(cur, pos1)
            if pos2 is None: cur = 1 - cur; continue
            val2 = flip(pos2)
            if val2 == val1: remove(pos1, pos2, cur); continue
            auto_take_opp(cur, val2, pos2)
            cur = 1 - cur

    s0, s1 = scores
    w = 0 if s0 > s1 else (1 if s1 > s0 else -1)
    return w, last_matcher, s0, s1

def measure(n, M, t1, t2, ng, seed):
    seeds = [np.random.SeedSequence(seed).spawn(ng)]
    ints = [s.generate_state(1)[0] for s in seeds[0]]
    res = Parallel(n_jobs=1, backend='loky')(
        delayed(play_game)(n, M, t1, t2, s) for s in ints)
    p1w, p2w, dr = 0, 0, 0; ts = [0,0]; p2l = 0; lw = 0; nd = 0
    for w, lm, s0, s1 in res:
        ts[0]+=s0; ts[1]+=s1
        if w==0: p1w+=1
        elif w==1: p2w+=1
        else: dr+=1
        if lm==1: p2l+=1
        if w>=0: nd+=1; lw+=(w==lm)
    dec = p1w+p2w
    return {
        'p1_wr': p1w/ng, 'p2_wr': p2w/ng, 'draws': dr/ng,
        'p2_cond': p2w/dec if dec>0 else .5,
        'p2_cond_err': 1.96*np.sqrt(max((p2w/dec)*(1-p2w/dec),1e-4)/dec) if dec>10 else .1,
        'gain': (ts[0]-ts[1])/ng, 'avg': [ts[0]/ng, ts[1]/ng],
    }

def run_matchup(n, M, t1, t2, ng, seed, label):
    return (label, measure(n, M, t1, t2, ng, seed))

# ═══════════════════════════════════════════════════════════════
# MATCHUPS
# ═══════════════════════════════════════════════════════════════
M = 7
ng = 10000

matchups_12 = [
    ("Miller v Miller", MILLER_OPT, MILLER_OPT),
    ("Zwick v Zwick", ZWICK_OPT, ZWICK_OPT),
    ("P1:Miller v P2:Zwick", MILLER_OPT, ZWICK_OPT),
    ("P1:Zwick v P2:Miller", ZWICK_OPT, MILLER_OPT),
]

board_sizes = [8, 10, 12, 16, 20]

# All matchups across board sizes
all_jobs = []
for n in board_sizes:
    for label, t1, t2 in matchups_12:
        all_jobs.append((n, M, t1, t2, ng, 99999*n + hash(label)%10000, f"n={n} {label}"))

print(f"\nRunning {len(all_jobs)} matchups × {ng} games...")
t0 = time.time()
all_res = Parallel(n_jobs=-1, backend='loky', verbose=5)(
    delayed(run_matchup)(*j) for j in all_jobs)
print(f"Done in {time.time()-t0:.1f}s")

# Print results
print(f"\n{'Matchup':<35s} {'P1':>6s} {'P2':>6s} {'Draw':>6s} {'P2|dec':>7s} {'Gain':>7s}")
print("─"*72)
for label, r in all_res:
    print(f"{label:<35s} {r['p1_wr']:6.3f} {r['p2_wr']:6.3f} {r['draws']:6.3f} "
          f"{r['p2_cond']:7.3f} {r['gain']:+7.4f}")

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

# Organize by board size and matchup
data = {}
for job, (label, r) in zip(all_jobs, all_res):
    n = job[0]
    # Extract matchup name (remove "n=X " prefix)
    mname = label.split(" ", 1)[1]
    data.setdefault(mname, {'ns': [], 'p2c': [], 'p2ce': [], 'gain': []})
    data[mname]['ns'].append(n)
    data[mname]['p2c'].append(r['p2_cond'])
    data[mname]['p2ce'].append(r['p2_cond_err'])
    data[mname]['gain'].append(r['gain'])

colors = {'Miller v Miller': '#e7298a', 'Zwick v Zwick': '#2166ac',
          'P1:Miller v P2:Zwick': '#4dac26', 'P1:Zwick v P2:Miller': '#d6604d'}

# Fig 1: Conditional P2 win rate
fig1, ax1 = plt.subplots(figsize=(12, 7))
for mname in data:
    d = data[mname]
    order = np.argsort(d['ns'])
    ns = np.array(d['ns'])[order]
    p2c = np.array(d['p2c'])[order]
    p2ce = np.array(d['p2ce'])[order]
    ax1.fill_between(ns, p2c-p2ce, p2c+p2ce, alpha=0.1, color=colors[mname])
    ax1.plot(ns, p2c, '-o', color=colors[mname], ms=7, lw=2, label=mname)

ax1.axhline(0.5, color='black', ls='--', lw=1.5, alpha=0.5)
ax1.set_xlabel('Board size n (pairs)', fontsize=12)
ax1.set_ylabel('P(P2 wins | someone wins)', fontsize=12)
ax1.set_title(f'Miller-Optimal vs Zwick Strategy (M={M}, {ng} games/point)\n'
              f'Who wins under limited memory?',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.2)
ax1.set_ylim(0.44, 0.56)
plt.tight_layout(); savefig('miller_vs_zwick_conditional.png')
print("\nSaved miller_vs_zwick_conditional.png")

# Fig 2: Expected gain
fig2, ax2 = plt.subplots(figsize=(12, 7))
for mname in data:
    d = data[mname]
    order = np.argsort(d['ns'])
    ns = np.array(d['ns'])[order]
    gains = np.array(d['gain'])[order]
    ax2.plot(ns, gains, '-o', color=colors[mname], ms=7, lw=2, label=mname)

ax2.axhline(0, color='black', ls='--', lw=1.5, alpha=0.5)
ax2.set_xlabel('Board size n (pairs)', fontsize=12)
ax2.set_ylabel('P1 expected gain (< 0 = P2 advantage)', fontsize=12)
ax2.set_title(f'Expected Gain: Miller-Optimal vs Zwick (M={M})',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2)
plt.tight_layout(); savefig('miller_vs_zwick_gain.png')
print("Saved miller_vs_zwick_gain.png")

# Fig 3: Strategy dominance matrix at n=12
fig3, ax3 = plt.subplots(figsize=(8, 6))
strats = ['Miller', 'Zwick']
matrix = np.zeros((2, 2))
for label, r in all_res:
    if 'n=12' not in label: continue
    mname = label.split(" ", 1)[1]
    if mname == 'Miller v Miller': matrix[0,0] = r['p2_cond']
    elif mname == 'Zwick v Zwick': matrix[1,1] = r['p2_cond']
    elif mname == 'P1:Miller v P2:Zwick': matrix[0,1] = r['p2_cond']
    elif mname == 'P1:Zwick v P2:Miller': matrix[1,0] = r['p2_cond']

im = ax3.imshow(matrix, cmap='RdBu', vmin=0.45, vmax=0.55)
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(strats, fontsize=13); ax3.set_yticklabels(strats, fontsize=13)
ax3.set_xlabel('P2 Strategy', fontsize=12); ax3.set_ylabel('P1 Strategy', fontsize=12)
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax3, label='P(P2 wins | decisive)', shrink=0.8)
ax3.set_title(f'Strategy Matrix at n=12, M={M}\nBlue=P2 wins more, Red=P1 wins more',
              fontsize=13, fontweight='bold')
plt.tight_layout(); savefig('miller_vs_zwick_matrix.png')
print("Saved miller_vs_zwick_matrix.png")

print("\n" + "="*70)
print("DONE")
print("="*70)#!/usr/bin/env python3
"""
Head-to-head: Miller-optimal strategy vs Zwick strategy
Both players have Miller memory M=7, but one uses the optimal
moves for M=7, the other uses Zwick's moves (designed for M=∞).

The difference: at k=3,5,7 Zwick says "flip two", Miller says "flip one".
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
# PRECOMPUTE BOTH MOVE TABLES
# ═══════════════════════════════════════════════════════════════

def compute_miller_values(N_max, M):
    if M is None: M = 2 * N_max + 10
    e = {(0,0): Fraction(0)}; opt = {}
    for n in range(1, N_max+1):
        if n <= M: e[(n,n)] = Fraction(n); opt[(n,n)] = 1
        start_k = min(n-1, M) if n <= M else M
        for k in range(start_k, -1, -1):
            p_den = 2*n-k
            if p_den == 0: continue
            p = Fraction(k, p_den); q = Fraction(2*(n-k), p_den)
            # e^1
            if k >= 1:
                k_nm1 = min(k+1, M)
                if k < M:
                    v1 = p*(1+e[(n-1, min(k-1,M))]) - q*e[(n, k_nm1)]
                else:
                    denom1 = 1 + q
                    v1 = p*(1+e[(n-1, M-1)])/denom1 if denom1 != 0 else Fraction(0)
            else: v1 = None
            # e^2
            d2 = 2*n-k-1
            if d2 > 0:
                k_prime = min(k+1, M)
                if k < M: k_lucky = k
                else: k_lucky = M-1
                if k < M: k_auto = k
                else: k_auto = M-1
                k_nm2 = min(k+2, M)
                fl = Fraction(1, d2)
                fa = Fraction(k_prime-1, d2) if k_prime >= 1 else Fraction(0)
                nk1 = n-k-1
                fn = Fraction(2*nk1, d2) if nk1 > 0 else Fraction(0)
                first = p*(1+e[(n-1, min(k-1,M))]) if k >= 1 else Fraction(0)
                ik = fl*(1+e[(n-1, min(k_lucky,M))])
                ia = fa*(1+e[(n-1, min(k_auto,M))])
                if k_nm2 <= M and (n, k_nm2) in e and k_nm2 != k:
                    v2 = first + q*(ik - ia - fn*e[(n, k_nm2)])
                elif k < M:
                    v2 = first + q*(ik - ia - fn*e.get((n, k_nm2), Fraction(0)))
                else:
                    rhs = first + q*(ik - ia)
                    denom2 = 1 + q*fn
                    v2 = rhs/denom2 if denom2 != 0 else Fraction(0)
            elif d2 == 0:
                v2 = Fraction(1) + e.get((n-1, min(k-1 if k>=1 else 0, M)), Fraction(0)) if k >= 1 else Fraction(1)
            else: v2 = Fraction(0)
            # Choose
            v1v = v1 if v1 is not None else Fraction(-99999)
            if k == 0: e[(n,k)] = v2; opt[(n,k)] = 2
            elif k == 1:
                if v1v >= v2: e[(n,k)] = v1v; opt[(n,k)] = 1
                else: e[(n,k)] = v2; opt[(n,k)] = 2
            else:
                if v1v > 0 and v1v >= v2: e[(n,k)] = v1v; opt[(n,k)] = 1
                elif v2 >= 0 and v2 >= v1v: e[(n,k)] = v2; opt[(n,k)] = 2
                elif v1v <= 0 and v2 <= 0: e[(n,k)] = Fraction(0); opt[(n,k)] = 0
                else: e[(n,k)] = max(v1v, v2); opt[(n,k)] = 1 if v1v > v2 else 2
    return e, opt

N = 40
print("Computing move tables...")
_, MILLER_OPT = compute_miller_values(N, 7)
_, ZWICK_OPT = compute_miller_values(N, None)

# Print comparison for key board sizes
for n_show in [12, 16, 24, 36]:
    print(f"\nn={n_show} ({2*n_show} cards) moves comparison:")
    print(f"  {'k':>3s}  Miller  Zwick")
    k_max = min(n_show, M)
    for k in range(k_max + 1):
        m = MILLER_OPT.get((n_show, k), '-')
        z = ZWICK_OPT.get((n_show, k), '-')
        diff = " ← DIFFERENT" if m != z else ""
        print(f"  k={k:2d}:   {m}       {z}{diff}")
    # Show what Zwick would do beyond M
    if n_show > M:
        print(f"  --- Zwick continues (unreachable with M={M}) ---")
        for k in range(M+1, min(n_show+1, M+6)):
            z = ZWICK_OPT.get((n_show, k), '-')
            print(f"  k={k:2d}:   -       {z}  (unreachable)")

# ═══════════════════════════════════════════════════════════════
# GAME ENGINE
# ═══════════════════════════════════════════════════════════════

class MillerMemory:
    def __init__(self, cap, rng):
        self.cap = cap; self.rng = rng; self.store = OrderedDict()
    def observe(self, pos, value):
        if pos in self.store: self.store.move_to_end(pos); self.store[pos] = value; return
        while len(self.store) >= self.cap: self.store.popitem(last=False)
        self.store[pos] = value
    def find_value(self, value, exclude=None):
        for pos, val in self.store.items():
            if val == value and pos != exclude: self.store.move_to_end(pos); return pos
        return None
    def known_alive(self, alive): return sum(1 for p in self.store if p in alive)
    def forget_pos(self, pos):
        if pos in self.store: del self.store[pos]

def play_game(n_pairs, M, move_table_p1, move_table_p2, seed):
    rng = np.random.default_rng(seed)
    cards = list(range(n_pairs)) * 2; rng.shuffle(cards)
    board = np.array(cards)
    alive = set(range(2*n_pairs))
    mem = [MillerMemory(M, rng), MillerMemory(M, rng)]
    scores = [0, 0]; last_matcher = -1
    tables = [move_table_p1, move_table_p2]

    def flip(pos):
        v = board[pos]
        mem[0].observe(pos, v); mem[1].observe(pos, v)
        return v

    def remove(p1, p2, player):
        nonlocal last_matcher
        alive.discard(p1); alive.discard(p2)
        mem[0].forget_pos(p1); mem[0].forget_pos(p2)
        mem[1].forget_pos(p1); mem[1].forget_pos(p2)
        scores[player] += 1; last_matcher = player

    def pick_new(player, exclude=None):
        al = [p for p in alive if p != exclude]
        if not al: return None
        unk = [p for p in al if p not in mem[player].store]
        return rng.choice(unk) if unk else rng.choice(al)

    def pick_known(player, exclude=None):
        al = [p for p in alive if p != exclude]
        if not al: return None
        kn = [p for p in al if p in mem[player].store]
        return rng.choice(kn) if kn else rng.choice(al)

    def try_match(player, pos1, val1):
        mp = mem[player].find_value(val1, exclude=pos1)
        if mp is not None and mp in alive:
            flip(mp)
            if board[mp] == val1: remove(pos1, mp, player); return True
        return False

    def auto_take_opp(player, val, pos):
        opp = 1 - player
        om = mem[opp].find_value(val, exclude=pos)
        if om is not None and om in alive:
            flip(om)
            if board[om] == val: remove(pos, om, opp)

    cur = 0; passes = 0
    for _ in range(50000):
        n = len(alive) // 2
        if n == 0: break
        k = min(mem[cur].known_alive(alive), M)
        # Clamp k for table lookup
        move = tables[cur].get((n, k))
        if move is None:
            # Fallback: if state not in table, use 2-move
            move = 2

        if move == 0:
            passes += 1
            if passes >= 4: break
            cur = 1 - cur; continue
        passes = 0

        if move == 1:
            pos1 = pick_new(cur)
            if pos1 is None: break
            val1 = flip(pos1)
            if try_match(cur, pos1, val1): continue  # matched, go again
            idle = pick_known(cur, pos1)
            if idle is not None: flip(idle)
            cur = 1 - cur

        elif move == 2:
            pos1 = pick_new(cur)
            if pos1 is None: break
            val1 = flip(pos1)
            if try_match(cur, pos1, val1): continue
            pos2 = pick_new(cur, pos1)
            if pos2 is None: cur = 1 - cur; continue
            val2 = flip(pos2)
            if val2 == val1: remove(pos1, pos2, cur); continue
            auto_take_opp(cur, val2, pos2)
            cur = 1 - cur

    s0, s1 = scores
    w = 0 if s0 > s1 else (1 if s1 > s0 else -1)
    return w, last_matcher, s0, s1

def measure(n, M, t1, t2, ng, seed):
    seeds = [np.random.SeedSequence(seed).spawn(ng)]
    ints = [s.generate_state(1)[0] for s in seeds[0]]
    res = Parallel(n_jobs=1, backend='loky')(
        delayed(play_game)(n, M, t1, t2, s) for s in ints)
    p1w, p2w, dr = 0, 0, 0; ts = [0,0]; p2l = 0; lw = 0; nd = 0
    for w, lm, s0, s1 in res:
        ts[0]+=s0; ts[1]+=s1
        if w==0: p1w+=1
        elif w==1: p2w+=1
        else: dr+=1
        if lm==1: p2l+=1
        if w>=0: nd+=1; lw+=(w==lm)
    dec = p1w+p2w
    return {
        'p1_wr': p1w/ng, 'p2_wr': p2w/ng, 'draws': dr/ng,
        'p2_cond': p2w/dec if dec>0 else .5,
        'p2_cond_err': 1.96*np.sqrt(max((p2w/dec)*(1-p2w/dec),1e-4)/dec) if dec>10 else .1,
        'gain': (ts[0]-ts[1])/ng, 'avg': [ts[0]/ng, ts[1]/ng],
    }

def run_matchup(n, M, t1, t2, ng, seed, label):
    return (label, measure(n, M, t1, t2, ng, seed))

# ═══════════════════════════════════════════════════════════════
# MATCHUPS
# ═══════════════════════════════════════════════════════════════
M = 7
ng = 10000

matchups_12 = [
    ("Miller v Miller", MILLER_OPT, MILLER_OPT),
    ("Zwick v Zwick", ZWICK_OPT, ZWICK_OPT),
    ("P1:Miller v P2:Zwick", MILLER_OPT, ZWICK_OPT),
    ("P1:Zwick v P2:Miller", ZWICK_OPT, MILLER_OPT),
]

board_sizes = [8, 12, 16, 24, 36]

# All matchups across board sizes
all_jobs = []
for n in board_sizes:
    for label, t1, t2 in matchups_12:
        all_jobs.append((n, M, t1, t2, ng, 99999*n + hash(label)%10000, f"n={n} {label}"))

print(f"\nRunning {len(all_jobs)} matchups × {ng} games...")
t0 = time.time()
all_res = Parallel(n_jobs=-1, backend='loky', verbose=5)(
    delayed(run_matchup)(*j) for j in all_jobs)
print(f"Done in {time.time()-t0:.1f}s")

# Print results
print(f"\n{'Matchup':<35s} {'P1':>6s} {'P2':>6s} {'Draw':>6s} {'P2|dec':>7s} {'Gain':>7s}")
print("─"*72)
for label, r in all_res:
    print(f"{label:<35s} {r['p1_wr']:6.3f} {r['p2_wr']:6.3f} {r['draws']:6.3f} "
          f"{r['p2_cond']:7.3f} {r['gain']:+7.4f}")

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

# Organize by board size and matchup
data = {}
for job, (label, r) in zip(all_jobs, all_res):
    n = job[0]
    # Extract matchup name (remove "n=X " prefix)
    mname = label.split(" ", 1)[1]
    data.setdefault(mname, {'ns': [], 'p2c': [], 'p2ce': [], 'gain': []})
    data[mname]['ns'].append(n)
    data[mname]['p2c'].append(r['p2_cond'])
    data[mname]['p2ce'].append(r['p2_cond_err'])
    data[mname]['gain'].append(r['gain'])

colors = {'Miller v Miller': '#e7298a', 'Zwick v Zwick': '#2166ac',
          'P1:Miller v P2:Zwick': '#4dac26', 'P1:Zwick v P2:Miller': '#d6604d'}

# Fig 1: Conditional P2 win rate
fig1, ax1 = plt.subplots(figsize=(12, 7))
for mname in data:
    d = data[mname]
    order = np.argsort(d['ns'])
    ns = np.array(d['ns'])[order]
    p2c = np.array(d['p2c'])[order]
    p2ce = np.array(d['p2ce'])[order]
    ax1.fill_between(ns, p2c-p2ce, p2c+p2ce, alpha=0.1, color=colors[mname])
    ax1.plot(ns, p2c, '-o', color=colors[mname], ms=7, lw=2, label=mname)

ax1.axhline(0.5, color='black', ls='--', lw=1.5, alpha=0.5)
ax1.set_xlabel('Board size n (pairs)', fontsize=12)
ax1.set_ylabel('P(P2 wins | someone wins)', fontsize=12)
ax1.set_title(f'Miller-Optimal vs Zwick Strategy (M={M}, {ng} games/point)\n'
              f'Who wins under limited memory?',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.2)
ax1.set_ylim(0.44, 0.56)
plt.tight_layout(); savefig('miller_vs_zwick_conditional.png')
print("\nSaved miller_vs_zwick_conditional.png")

# Fig 2: Expected gain
fig2, ax2 = plt.subplots(figsize=(12, 7))
for mname in data:
    d = data[mname]
    order = np.argsort(d['ns'])
    ns = np.array(d['ns'])[order]
    gains = np.array(d['gain'])[order]
    ax2.plot(ns, gains, '-o', color=colors[mname], ms=7, lw=2, label=mname)

ax2.axhline(0, color='black', ls='--', lw=1.5, alpha=0.5)
ax2.set_xlabel('Board size n (pairs)', fontsize=12)
ax2.set_ylabel('P1 expected gain (< 0 = P2 advantage)', fontsize=12)
ax2.set_title(f'Expected Gain: Miller-Optimal vs Zwick (M={M})',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2)
plt.tight_layout(); savefig('miller_vs_zwick_gain.png')
print("Saved miller_vs_zwick_gain.png")

# Fig 3: Strategy dominance matrix at n=12
fig3, ax3 = plt.subplots(figsize=(8, 6))
strats = ['Miller', 'Zwick']
matrix = np.zeros((2, 2))
n_matrix = 16  # Standard 32-card game
for label, r in all_res:
    if f'n={n_matrix}' not in label: continue
    mname = label.split(" ", 1)[1]
    if mname == 'Miller v Miller': matrix[0,0] = r['p2_cond']
    elif mname == 'Zwick v Zwick': matrix[1,1] = r['p2_cond']
    elif mname == 'P1:Miller v P2:Zwick': matrix[0,1] = r['p2_cond']
    elif mname == 'P1:Zwick v P2:Miller': matrix[1,0] = r['p2_cond']

im = ax3.imshow(matrix, cmap='RdBu', vmin=0.45, vmax=0.55)
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(strats, fontsize=13); ax3.set_yticklabels(strats, fontsize=13)
ax3.set_xlabel('P2 Strategy', fontsize=12); ax3.set_ylabel('P1 Strategy', fontsize=12)
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax3, label='P(P2 wins | decisive)', shrink=0.8)
ax3.set_title(f'Strategy Matrix at n={n_matrix} (32 cards), M={M}\nBlue=P2 wins more',
              fontsize=13, fontweight='bold')
plt.tight_layout(); savefig('miller_vs_zwick_matrix.png')
print("Saved miller_vs_zwick_matrix.png")

print("\n" + "="*70)
print("DONE")
print("="*70)