# Optimal Strategy for the Memory Card Game Under Bounded Working Memory

An extension of [Zwick & Paterson (1993)](https://doi.org/10.1016/0304-3975(93)90355-W) to the case where players have finite working memory, modelled as an LRU cache of capacity $M$.

**Blog post:** [taulantkoka.com/projects/memory](https://taulantkoka.com/projects/memory) (includes an interactive game where you can play against the optimal strategy)

## What this is about

The card game Memory (Concentration) has a known optimal strategy under perfect recall, computed by Zwick and Paterson via backward induction on the state space $(n, k)$ where $n$ is the number of remaining pairs and $k$ is the number of inspected cards. Their main finding: Player 2 has a slight advantage, and both players should sometimes deliberately pass (flip two known non-matching cards) to control parity.

This project asks: what happens when players can only remember $M$ card positions at a time?

The key modelling choice is deterministic LRU eviction, which keeps both players' memories identical at all times. This preserves the perfect-information structure of the game and allows the same backward induction approach, with a modified boundary condition at $k = M$.

## Main results

1. **Greedy matching is strictly optimal** under shared bounded memory. If you know where a pair is, take it immediately. (Unlike Zwick's model, where holding pairs in reserve can be strategic.)

2. **The optimal strategy is conservative:** flip two unknown cards only when $k \leq 1$; otherwise flip one unknown and waste the second flip on a known card. Never pass. This is the "defensive" strategy that [Kilian (2025)](https://samuelkilian.de/about.html) found empirically.

3. **The bounded-memory strategy strictly dominates Zwick's** when memory is finite. The advantage grows with board size: at $n = 36$ (72 cards), the bounded-memory player gains ~3 pairs over a Zwick opponent.

4. **Memory capacity dwarfs positional advantage.** A single extra memory slot is worth 50-100x more than the P1/P2 positional difference.

## Project structure

```
memory-game/
├── run_analysis.py                  # Master runner (saves all figures as PDF + SVG)
├── simulations/
│   ├── 00_exact_dp.py               # Exact DP: values, move tables, Zwick verification
│   ├── 01_bounded_vs_zwick.py       # Head-to-head: bounded-optimal vs Zwick (100k games)
│   ├── 02_fluctuation.py            # Robustness to capacity noise σ (100k games)
│   ├── 03_asymmetric.py             # Different capacities M₁ ≠ M₂ (100k games)
│   └── 04_draw_rate.py              # Draw rate + P2 advantage vs M (100k games)
├── figures/                         # Auto-generated PDF + SVG (after running)
└── report/
    ├── report.md                    # Standalone report (references figures/)
    ├── memory_game.html             # Interactive game (standalone, CDN dependencies)
    └── memory_game.jsx              # Same game as React component
```

## Running

Requirements: Python 3.9+, numpy, matplotlib, joblib.

```bash
pip install numpy matplotlib joblib

# Run everything (takes ~30 min at 100k games/point)
python run_analysis.py

# Run a single analysis
python run_analysis.py --only 01

# List available analyses
python run_analysis.py --list
```

Figures are saved to `figures/` as both PDF and SVG. The report references the SVG versions.

## The exact DP

The core computation (`00_exact_dp.py`) uses rational arithmetic (`fractions.Fraction`) for exact results. For $M = 7$, $n = 12$:

| | $k{=}0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ |
|---|---|---|---|---|---|---|---|---|
| **Zwick** | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| **Bounded** | 2 | 2 | 1 | **1** | 1 | **1** | 1 | **1** |

The strategies diverge at $k = 3, 5, 7$: Zwick flips two unknown cards, bounded-optimal flips one. The second unknown card churns memory when capacity is the bottleneck.

The Zwick verification reproduces Table 2 of the original paper exactly for all $n \leq 15$.

## Interactive game

`report/memory_game.html` is a self-contained HTML file (React + Tailwind via CDN, no build step) that lets you:

- Play against a bot using either the bounded-optimal or Zwick strategy
- Decouple strategy from memory capacity (e.g., test Zwick with $M = 7$ actual memory)
- Toggle visibility of the bot's memory state
- Choose who goes first
- Run bot-vs-bot simulations with independent strategy/memory settings for each player

To embed in a Jekyll site: drop the file in `/assets/` and use an iframe.

## References

- Zwick, U. & Paterson, M. S. (1993). The memory game. *Theoretical Computer Science*, 110(1), 169-196.
- Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81-97.