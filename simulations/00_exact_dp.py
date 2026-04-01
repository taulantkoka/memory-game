#!/usr/bin/env python3
"""
Exact Optimal Strategy for Memory Game with Capacity M

Extends Zwick-Paterson analysis. State: (n, k) where
  n = pairs remaining
  k = cards in shared memory (both players identical), k ≤ M

For k < M: transitions identical to Zwick (no evictions)
For k = M: non-matching flips cause LRU eviction, k stays at M
           This creates solvable self-referential equations.

Key result: produces Table 1 (values) and Table 2 (optimal moves)
for any capacity M, including M=∞ (recovers Zwick exactly).
"""

from fractions import Fraction
import sys

def compute_miller_values(N_max, M):
    """
    Compute exact game values for Memory with capacity M.
    
    Returns e[(n,k)], opt[(n,k)] for all valid states.
    M = None means infinite capacity (standard Zwick).
    """
    if M is None:
        M = 2 * N_max + 10  # effectively infinite
    
    e = {(0, 0): Fraction(0)}
    e1 = {}
    e2 = {}
    opt = {}
    
    for n in range(1, N_max + 1):
        # Terminal: k = n means all alive cards known → take them all
        k_max = min(n, M)
        # But if n ≤ M, we can know all cards: e_{n,n} = n
        if n <= M:
            e[(n, n)] = Fraction(n)
            opt[(n, n)] = 1
        
        # Compute backwards from k = min(n, M) down to k = 0
        # If n > M, start from k = M
        start_k = min(n, M)
        if n <= M:
            start_k = n - 1  # n,n already done
        
        for k in range(start_k, -1, -1):
            p_num = k                # numerator of k/(2n-k)
            p_den = 2 * n - k        # denominator
            q_num = 2 * (n - k)      # numerator of 2(n-k)/(2n-k)
            
            if p_den == 0:
                continue
            
            p = Fraction(p_num, p_den)
            q = Fraction(q_num, p_den)
            
            # ── e^1: 1-move ──
            # Flip one new card:
            # - prob p: matches known → pair → (n-1, k-1), play again
            # - prob q: no match → idle ply → opponent plays from (n, k')
            #   where k' = min(k+1, M)
            
            if k >= 1:
                k_after_match_1 = min(k - 1, M)
                k_after_nomatch_1 = min(k + 1, M)
                
                if k < M:
                    # Standard Zwick: no eviction on non-match
                    val1 = p * (1 + e[(n - 1, k_after_match_1)]) - q * e[(n, k_after_nomatch_1)]
                else:
                    # k = M: non-match loops back to (n, M)
                    # e^1 = p*(1+e_{n-1,M-1}) - q*e_{n,M}
                    # If e_{n,M} = e^1_{n,M}:
                    #   e^1(1+q) = p*(1+e_{n-1,M-1})
                    #   e^1 = p*(1+e_{n-1,M-1})/(1+q)
                    # But we don't yet know if 1-move is optimal at k=M.
                    # Compute the EQUATION value: e^1 + q*e_{n,M} = p*(1+e_{n-1,M-1})
                    # We'll solve below after computing e^2 too.
                    val1 = 'boundary'  # placeholder
                
                e1[(n, k)] = val1
            else:
                e1[(n, k)] = None  # 1-move not legal at k=0
            
            # ── e^2: 2-move ──
            # First card:
            # - prob p: matches known → pair → (n-1, k-1), play again
            # - prob q: no match → flip second card
            #   After first non-match: memory has k' = min(k+1, M) entries
            #   Second card from (2n-k-1) unseen:
            #     prob 1/(2n-k-1): matches card1 → pair → state depends on k
            #     prob (k'-1)/(2n-k-1) = (min(k+1,M)-1)/(2n-k-1):
            #       matches known (not card1) → opponent auto-takes
            #     prob rest: no match → opponent plays from (n, min(k+2, M))
            
            d2 = 2 * n - k - 1
            
            if d2 > 0:
                k_prime = min(k + 1, M)  # memory after first non-matching flip
                
                # Lucky match (card2 = card1): state (n-1, ?)
                # Both cards removed. Memory had k_prime, minus the first card = k_prime - 1
                # (second card matched first, both removed, second never entered memory)
                # Actually: first card IS in memory (was observed). Match removes it.
                # If k < M: first card was added (k→k+1), then pair removed (k+1→k+1-1=k). Wait, both removed: k+1-2=k-1.
                # Hmm: in Zwick, lucky match → (n-1, k). Let me re-derive.
                # Before: k known alive cards. Card1 flipped (new): k+1 known. 
                # Card2 flipped, matches card1: both removed from board AND memory.
                # Memory: k+1-2 = k-1? Or k+1-1=k (only card1 was added, card2 matched immediately)?
                #
                # In Zwick's formula, the lucky match leads to e_{n-1,k}. So the state is (n-1, k).
                # This means: card1 was effectively "inspected" (k→k+1), but then the pair
                # (card1+card2) is removed, reducing k by 1: net k. But card2 was also inspected...
                # Actually in Zwick, card2 "inspected but immediately removed" doesn't count.
                # The convention: after removing a pair, inspected count decreases appropriately.
                #
                # For Miller at k < M: same as Zwick, state is (n-1, k)
                # For Miller at k = M: card1 causes eviction (k stays M), then pair removed.
                #   Memory had M (after eviction from card1), card2 matches card1.
                #   Remove card1 from memory: M-1. Card2 was never in memory.
                #   State: (n-1, M-1)
                
                if k < M:
                    k_lucky = k  # same as Zwick
                else:
                    k_lucky = M - 1
                k_lucky = min(k_lucky, M)
                
                # Auto-take (card2 matches known, not card1): opponent takes pair
                # Memory: card1 is in memory, card2 observed → match with X in memory
                # X and card2 removed. Card1 stays.
                # Zwick: state (n-1, k). In Miller at k=M: 
                #   After card1 (eviction): M entries. Card2 matches X, both removed: M-1.
                #   But wait, card2 also briefly observed. In our convention: match detected
                #   before card2 enters memory. So memory: M-1 (X removed). Card2 never stored.
                #   State: (n-1, M-1)
                # For k < M: card1 added (k→k+1), card2 matches X (X removed, card2 not stored):
                #   k+1-1 = k. State: (n-1, k). Same as Zwick.
                
                if k < M:
                    k_auto = k  # same as Zwick
                else:
                    k_auto = M - 1
                k_auto = min(k_auto, M)
                
                # No match at all: opponent plays from (n, min(k+2, M))
                k_nomatch2 = min(k + 2, M)
                
                # Probabilities for second card
                frac_lucky = Fraction(1, d2)
                frac_auto = Fraction(k_prime - 1, d2) if k_prime >= 1 else Fraction(0)
                nk1 = n - k - 1
                frac_new2 = Fraction(2 * nk1, d2) if nk1 > 0 else Fraction(0)
                
                k_match1 = min(k - 1, M) if k >= 1 else 0
                
                first_term = p * (1 + e[(n - 1, k_match1)]) if k >= 1 else Fraction(0)
                
                # Inner bracket (second card outcomes):
                inner_known = frac_lucky * (1 + e[(n - 1, k_lucky)])
                inner_auto = frac_auto * (1 + e[(n - 1, k_auto)])
                
                if k < M and k + 2 <= M:
                    # No boundary issues
                    inner_new = frac_new2 * e.get((n, k_nomatch2), Fraction(0))
                    val2 = first_term + q * (inner_known - inner_auto - inner_new)
                    e2[(n, k)] = val2
                elif k_nomatch2 <= M and (n, k_nomatch2) in e:
                    inner_new = frac_new2 * e[(n, k_nomatch2)]
                    val2 = first_term + q * (inner_known - inner_auto - inner_new)
                    e2[(n, k)] = val2
                else:
                    # Self-referential: k_nomatch2 = M and we're computing e_{n,M}
                    # val2 = first_term + q*(inner_known - inner_auto) - q*frac_new2*e_{n,M}
                    e2[(n, k)] = 'boundary'
            else:
                # d2 = 0: n=1,k=1 or similar. Only 1 unseen card, must match.
                if k >= 1:
                    e2[(n, k)] = Fraction(1) + e.get((n-1, min(k-1, M)), Fraction(0))
                else:
                    e2[(n, k)] = Fraction(1)
            
            # ── Solve boundary cases ──
            if k == M and (e1.get((n, k)) == 'boundary' or e2.get((n, k)) == 'boundary'):
                # At k = M, both e^1 and e^2 may reference e_{n,M}
                # Solve for each assuming it's the optimal move, then pick best.
                
                k_match1 = M - 1
                first_term = p * (1 + e[(n - 1, k_match1)]) if k >= 1 else Fraction(0)
                
                # e^1 at k=M: p*(1+e_{n-1,M-1}) - q*e_{n,M}
                # If e_{n,M} = e^1: e^1 = p*(1+e_{n-1,M-1})/(1+q)
                denom1 = 1 + q
                if denom1 != 0:
                    v1_solved = first_term / denom1
                else:
                    v1_solved = Fraction(0)
                e1[(n, k)] = v1_solved
                
                # e^2 at k=M:
                if d2 > 0:
                    k_prime = M  # min(k+1, M) = M
                    frac_lucky = Fraction(1, d2)
                    frac_auto = Fraction(M - 1, d2)
                    nk1 = n - k - 1
                    frac_new2 = Fraction(2 * nk1, d2) if nk1 > 0 else Fraction(0)
                    
                    inner_known = frac_lucky * (1 + e[(n - 1, min(M - 1, M))])  # k_lucky = M-1
                    inner_auto = frac_auto * (1 + e[(n - 1, min(M - 1, M))])    # k_auto = M-1
                    
                    # val2 = first_term + q*(inner_known - inner_auto) - q*frac_new2*e_{n,M}
                    # If e_{n,M} = val2:
                    # val2*(1 + q*frac_new2) = first_term + q*(inner_known - inner_auto)
                    
                    rhs = first_term + q * (inner_known - inner_auto)
                    denom2 = 1 + q * frac_new2
                    if denom2 != 0:
                        v2_solved = rhs / denom2
                    else:
                        v2_solved = Fraction(0)
                    e2[(n, k)] = v2_solved
                else:
                    if e2[(n, k)] == 'boundary':
                        e2[(n, k)] = Fraction(1)
            
            # ── Determine optimal move and value ──
            v1 = e1[(n, k)] if e1[(n, k)] is not None else Fraction(-99999)
            v2 = e2[(n, k)] if e2[(n, k)] is not None else Fraction(-99999)
            
            if k == 0:
                e[(n, k)] = v2
                opt[(n, k)] = 2
            elif k == 1:
                if v1 >= v2:
                    e[(n, k)] = v1; opt[(n, k)] = 1
                else:
                    e[(n, k)] = v2; opt[(n, k)] = 2
            else:
                if v1 > 0 and v1 >= v2:
                    e[(n, k)] = v1; opt[(n, k)] = 1
                elif v2 >= 0 and v2 >= v1:
                    e[(n, k)] = v2; opt[(n, k)] = 2
                elif v1 <= 0 and v2 <= 0:
                    e[(n, k)] = Fraction(0); opt[(n, k)] = 0
                else:
                    e[(n, k)] = max(v1, v2)
                    opt[(n, k)] = 1 if v1 > v2 else 2
    
    return e, opt


# ═══════════════════════════════════════════════════════════════
# VERIFICATION: M=∞ should match Zwick exactly
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("VERIFICATION: M=∞ vs Zwick Tables")
print("=" * 70)

e_inf, opt_inf = compute_miller_values(15, None)

print("\nOptimal moves (should match Zwick Table 2):")
expected_moves = {
    1: "21", 2: "221", 3: "2121", 4: "22121", 5: "212101",
    6: "2112101", 7: "21212101", 8: "221212101", 9: "2121212101",
    10: "22121212101", 11: "212121210101", 12: "2212121210101",
    13: "21212121210101", 14: "221212121210101", 15: "2121212121210101",
}

all_match = True
for n in range(1, 16):
    computed = "".join(str(opt_inf[(n, k)]) for k in range(n + 1))
    expected = expected_moves[n]
    match = "✓" if computed == expected else "✗"
    if computed != expected: all_match = False
    print(f"  n={n:2d}: {computed:20s} {'':3s} {match}")

print(f"\n  {'ALL MATCH' if all_match else 'MISMATCH'}")


# ═══════════════════════════════════════════════════════════════
# MAIN: Compute for various M values
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MILLER VALUES: Optimal strategy for finite M")
print("=" * 70)

N_max = 15

for M in [3, 5, 7, 9, 12, 20, None]:
    M_label = f"M={M}" if M is not None else "M=∞"
    print(f"\n{'─'*70}")
    print(f"  {M_label}")
    print(f"{'─'*70}")
    
    e, opt = compute_miller_values(N_max, M)
    
    # Optimal moves
    print(f"\n  Optimal moves:")
    for n in range(1, min(N_max + 1, 16)):
        k_max = min(n, M if M is not None else n)
        moves = "".join(str(opt.get((n, k), '?')) for k in range(k_max + 1))
        print(f"    n={n:2d}: {moves}")
    
    # Values at k=0 (who has the advantage going first?)
    print(f"\n  Position values e_{{n,0}} (>0 = P1 advantage, <0 = P2 advantage):")
    for n in range(1, min(N_max + 1, 21)):
        val = e.get((n, 0), None)
        if val is not None:
            who = "P1" if val > 0 else ("P2" if val < 0 else "fair")
            print(f"    n={n:2d}: e = {float(val):+.6f}  → {who}")
    
    # Compare with Zwick
    if M is not None:
        print(f"\n  Advantage comparison (Miller M={M} vs Zwick M=∞):")
        for n in [6, 8, 10, 12]:
            v_miller = float(e.get((n, 0), 0))
            v_zwick = float(e_inf.get((n, 0), 0))
            ratio = v_miller / v_zwick if v_zwick != 0 else float('inf')
            print(f"    n={n:2d}: Miller={v_miller:+.6f}  Zwick={v_zwick:+.6f}  "
                  f"ratio={ratio:.2f}x")


# ═══════════════════════════════════════════════════════════════
# KEY RESULT: How does M change the advantage?
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("KEY RESULT: e_{n,0} as function of M for various n")
print("=" * 70)

M_values = list(range(2, 25)) + [30, 40, 50, None]
n_values = [6, 8, 10, 12, 16, 20]

print(f"\n{'M':>5s}", end="")
for n in n_values:
    print(f"{'n='+str(n):>12s}", end="")
print()
print("─" * (5 + 12 * len(n_values)))

for M in M_values:
    M_label = f"{M:5d}" if M is not None else "  inf"
    N = max(n_values)
    e, _ = compute_miller_values(N, M)
    print(f"{M_label}", end="")
    for n in n_values:
        val = e.get((n, 0), None)
        if val is not None:
            print(f"{float(val):+12.6f}", end="")
        else:
            print(f"{'N/A':>12s}", end="")
    print()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
