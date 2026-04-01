# The Optimal Strategy for Memory Under Bounded Working Memory

## 1. The Game

Memory (also known as Concentration or Pairs) is a card game played with $n$ pairs of identical cards ($2n$ cards total), shuffled and placed face down on a table. Players alternate turns. On each turn, a player flips two cards face up. If the two cards match, the player takes the pair and plays again. If they do not match, both cards are flipped back face down and the turn passes to the opponent. The player with the most pairs at the end wins.

Despite being a children's game, Memory has a surprisingly rich strategic structure. Every card flip reveals information to *both* players, and the key decision is how to balance learning (flipping new cards) against exploiting what you already know (flipping a card whose match you remember). This raises a natural question: **does it matter who goes first?**

*A note on methodology: this is the first time I've produced a mathematical result with the help of an AI agent. The proofs, dynamic programming formulations, and simulations in this post were developed in collaboration with Claude Opus (Anthropic). The experience was something like having a very fast, very patient research partner who never needs coffee but occasionally needs to be told it's reading a plot wrong.*
 
## 2. What Zwick and Paterson Showed

In 1993, Uri Zwick and Michael Paterson published the definitive analysis of Memory under the assumption that both players have *perfect memory*, they remember the identity and position of every card ever flipped (Zwick & Paterson, 1993).

Their key insight was to characterise each game position by just two numbers:
- $n$: the number of pairs remaining on the board
- $k$: the number of cards both players have previously inspected (and remember perfectly)

At each turn, the player chooses one of three moves:

- **0-move (pass):** Flip two already-known, non-matching cards ("do nothing"). The turn passes to the opponent.
- **1-move:** Flip one new card. If it matches a remembered card, take the pair and play again. If not, "waste" the second flip by re-flipping a card you already know.
- **2-move:** Flip one new card. If it matches a remembered card, take the pair. If not, flip a *second* new card. If the two new cards happen to match, take the pair. Otherwise, the turn passes.

Through backward induction on the state space $(n, k)$, they computed the exact optimal strategy. The results are surprising:

1. **Player 2 has a slight advantage** for even $n \geq 8$, of magnitude $\approx 1/(4n)$ pairs.
2. **The key strategic weapon is the pass.** At high $k$, both players know where most pairs are but deliberately refrain from taking them, engaging in a delicate game of parity-control. The pass manipulates *whose turn it will be* when the last pairs are finally taken.
3. **Stalemates occur** in about 20–30% of games under optimal play, when both players pass indefinitely.

This is elegant, but immediately raises the question: does any of this survive when players don't have perfect memory?

## 3. First Attempt: Stochastic Memory Decay

My first approach, inspired by [Samuel Kilian](https://samuelkilian.de/about.html) (who's posts about the memory game peaked my interest), was to model forgetting as probabilistic decay: each turn, every remembered card has a probability $\delta$ of being forgotten. This is analytically convenient, the Zwick framework mostly carries over, and it lets you sweep continuously from perfect memory ($\delta = 0$) to complete amnesia ($\delta = 1$).

I ran large-scale Monte Carlo simulations (up to 1 million games per parameter point) under this model. The results were interesting: at low $\delta$, the Zwick P2 advantage survives. Around $\delta \approx 0.10\text{–}0.15$, there's a phase transition where the Zwick stalemate regime collapses and the game becomes more decisive. At high $\delta$, the game converges to a coin flip.

But the model bothered me. Probabilistic decay means you might remember a card you saw 20 turns ago while forgetting one you saw 2 turns ago. That doesn't match my experience of how memory works in this game. You either correctly remember where a card is or you don't. And when your brain is full and you see something new, something old gets pushed out. The bottleneck feels more like *capacity* than *reliability*.

## 4. From Miller's Law to a Tractable Model

This pointed me toward one of the most robust findings in cognitive psychology. In 1956, George Miller showed that human working memory has a capacity of approximately $7 \pm 2$ items (Miller, 1956). In the 70 years since, the "magical number seven" has been refined and debated, but the core insight, that working memory has a hard capacity limit, has held up remarkably well.

Of course, real human memory is more complex than a fixed-size buffer. Decay, interference, attention, emotional salience, rehearsal, they all play a role. A fully realistic model would need to account for all of these, and would almost certainly be intractable.

So I opted for a deliberate simplification: a deterministic bounded-memory model that captures the capacity constraint while remaining analytically solvable. Each player has a working memory of capacity $M$ (number of card positions). When memory is full and a new card is observed, the **least recently used** entry is evicted. This is not meant to be a faithful model of human cognition, it's the simplest model that has the property I care about: memory is bounded, and new information displaces old information.

The reason this simplification works so well turns out to be mathematical rather than psychological.

**The key property:** Since both players see every flip and both use the same deterministic eviction rule, **both players have identical memory at all times.** The memory state is common knowledge. This means the game remains one of perfect information, the same class of game Zwick analysed. All that changes is the state space.

Any stochastic model of forgetting, whether decay, random interference, or attention fluctuation, would give the two players *different* memories, creating private information and turning the game into something like poker. That's a vastly harder problem (and an interesting open one). The deterministic LRU model sidesteps this entirely, keeping Zwick's backward induction machinery available.

Following Zwick, I characterise positions by $(n, k)$ where $k \leq M$ is the number of cards currently in the shared memory.

## 5. Greedy Matching Is Optimal
 
Before computing the optimal strategy, I need to settle a foundational question: when you know where a matching pair is, should you always take it immediately? Or could it sometimes be better to hold the pair in reserve — leaving it in memory unmatched to control the timing of who gets the last pair, as Zwick's pass does under perfect recall?
 
### Setup
 
I measure the game from the perspective of the player whose turn it is, using a **value function** $e_{n,k}$ that represents the expected score difference (current player's pairs minus opponent's pairs) from state $(n, k)$ under optimal play by both sides. A positive $e_{n,k}$ means the current player is favoured; negative means the opponent is favoured.
 
To be precise: if Player A and Player B play optimally from state $(n, k)$ with A to move, and A ends up with $S_A$ pairs and B with $S_B$ pairs from this point forward, then $e_{n,k} = \mathbb{E}[S_A - S_B]$.
 
### The Dominance Argument
 
**Theorem.** *Under shared bounded memory with LRU eviction, immediately matching any known pair is strictly optimal.*
 
**Proof.** Suppose it is Player A's turn, and both players' shared memory contains a known matching pair, cards at positions $\alpha$ and $\beta$ with the same value. Since memory is shared, Player B also knows about this pair.
 
Compare two actions:
 
**TAKE:** Player A flips $\alpha$ and $\beta$. They match. Player A scores $+1$ and gets another turn. The game continues from state $(n-1, k-2)$ with Player A still to move. Player A's expected payoff from here is:
 
$$V_{\text{take}} = 1 + e_{n-1,\, k-2}$$
 
**DEFER:** Player A makes any other move (pass, 1-move, or 2-move) without taking the pair. Eventually the turn passes to Player B. Player B sees the identical memory, including $\alpha$ and $\beta$. What does Player B do?
 
If Player B is rational, Player B takes the pair (by the same logic we're establishing, but we can also consider the case where B defers, which leads to mutual passing with value 0). So:
 
- *B takes the pair:* B scores $+1$ and plays again from $(n-1, k')$. From A's perspective, A lost a point and B is now in the favourable position. A's payoff: $-1 - e_{n-1,\, k'}$ for some $k'$ (the exact $k'$ depends on what A did before deferring, but it doesn't matter for the argument).
 
- *B also defers:* Both players pass on a known pair. This leads to mutual passing, a stalemate with value 0.
 
In the best case for DEFER, A gets value $0$ (mutual stalemate). But TAKE gives $1 + e_{n-1,\, k-2}$. Since the game has at most $n-1$ pairs left, the continuation value satisfies $e_{n-1,\, k-2} > -(n-1)$, so $V_{\text{take}} > 1 - (n-1) = 2 - n$. For $n \leq 1$ this is trivially positive. For $n \geq 2$, the computed exact values (via the DP below) confirm $1 + e_{n-1,\, k-2} > 0$ for all relevant states, which can be verified by inspection of the value tables.
 
In the more realistic case where B takes the pair, A's payoff from DEFER is $-1 - e_{n-1,\, k'}$, which is strictly worse than $1 + e_{n-1,\, k-2}$, a swing of at least 2 points.
 
In all cases, TAKE dominates DEFER. $\square$
 
**Remark.** This proof *requires* shared memory. The crucial step is "Player B also knows about this pair." If players had private, unobservable memories, Player A could know a pair that Player B doesn't. Holding it in reserve, passing while secretly knowing a pair, could then be genuinely strategic, because B can't steal what B doesn't know about. The private-memory case is an interesting open problem, and likely requires game-theoretic tools for incomplete information (Bayesian Nash equilibria rather than backward induction).
 
**Corollary.** Under optimal play, memory never holds both cards of a pair (any pair is matched the moment it's discovered). Therefore all $k$ entries in memory are unpaired singletons with distinct values. The state $(n, k)$ with $k \leq M$ is a sufficient description of the game, and backward induction on this state space yields the optimal strategy.
 
## 6. The Optimal Strategy
 
With greedy matching established, I can compute the optimal 0/1/2-move strategy by backward induction on the state space $(n, k)$ with $0 \leq k \leq \min(n, M)$. This section walks through the derivation step by step. If you want to skip the math and go straight to the results, jump to [Section 7](#7-results).
 
### 6.1 Value Function
 
The value function $e_{n,k}$ represents the expected score advantage for the player to move. The base case is $e_{0,0} = 0$ (no cards left, no advantage). If $k = n \leq M$ (all remaining cards are known and fit in memory), the current player takes all remaining pairs: $e_{n,n} = n$.
 
For each state, the player chooses the move (0, 1, or 2) that maximises $e_{n,k}$. After the move, it becomes the opponent's turn, and the opponent's value is $-e$ of whatever state results (since the game is zero-sum from the opponent's perspective).
 
### 6.2 Move Values for $k < M$ (Memory Not Full)
 
Here the transitions are identical to Zwick–Paterson. Let
 
$$p = \frac{k}{2n - k} \qquad q = \frac{2(n-k)}{2n-k}$$
 
where $p$ is the probability that a randomly flipped unknown card matches one of the $k$ singletons in memory (there are $k$ target cards among the $2n - k$ unknown positions), and $q = 1 - p$.
 
**0-move (pass):** Do nothing. The opponent faces state $(n, k)$. Value:
$$e^0_{n,k} = -e_{n,k}$$
This is only worth considering if $e_{n,k} \leq 0$ (i.e., the current position is unfavourable), giving $e^0 = 0$. The pass is legal only for $k \geq 2$.
 
**1-move:** Flip one new card.
- With probability $p$: it matches a known singleton. Take the pair ($+1$), memory loses one entry ($k \to k-1$), board shrinks ($n \to n-1$), and you play again from $(n-1, k-1)$.
- With probability $q$: no match. The new card enters memory ($k \to k+1$). You waste your second flip on a known card (revealing nothing new). Turn passes to opponent, who faces $(n, k+1)$.
 
$$e^1_{n,k} = p \cdot (1 + e_{n-1,\, k-1}) - q \cdot e_{n,\, k+1}$$
 
**2-move:** Flip one new card. If it matches (prob $p$), same as the 1-move. If not (prob $q$), flip a *second* new card. The second card's outcomes, among the $d = 2n - k - 1$ remaining unknown positions:
- With probability $\frac{1}{d}$: it matches the first card (lucky match). Take the pair, play again.
- With probability $\frac{k}{d}$: it matches a *different* singleton in memory. The opponent auto-takes that pair (both players saw the match).
- With probability $\frac{2(n-k-1)}{d}$: no match at all. Both new cards are now in memory ($k \to k+2$). Turn passes.
 
The full 2-move formula (following Zwick–Paterson Section 3) is:
 
$$e^2_{n,k} = p \cdot (1 + e_{n-1,\, k-1}) + q \left[ \frac{1}{d}(1 + e_{n-1,\, k}) - \frac{k}{d}(1 + e_{n-1,\, k}) - \frac{2(n{-}k{-}1)}{d} \cdot e_{n,\, k+2} \right]$$
 
### 6.3 Move Values for $k = M$ (Memory Full — The New Boundary)
 
This is where the bounded model departs from Zwick. When memory is full and a new card doesn't match, LRU eviction fires: the new card enters memory and the oldest entry is pushed out. The net effect: $k$ stays at $M$ instead of increasing to $M + 1$.
 
**1-move at $k = M$:** The miss transitions back to $(n, M)$ instead of $(n, M+1)$:
 
$$e^1_{n,M} = p \cdot (1 + e_{n-1,\, M-1}) - q \cdot e_{n,M}$$
 
If the 1-move turns out to be optimal at this state (i.e., $e_{n,M} = e^1_{n,M}$), this becomes self-referential. Solving:
 
$$e^1_{n,M} + q \cdot e^1_{n,M} = p \cdot (1 + e_{n-1,\, M-1})$$
$$e^1_{n,M} = \frac{p \cdot (1 + e_{n-1,\, M-1})}{1 + q}$$
 
**2-move at $k = M$:** Similarly, the "no match on either card" transition returns to $(n, M)$ instead of advancing to $(n, M+2)$. The self-referential equation has the same structure and is solved analogously.
 
### 6.4 Putting It Together: Optimal Move Selection
 
At each state $(n, k)$, the optimal move is:
 
$$e_{n,k} = \begin{cases} e^2_{n,0} & \text{if } k = 0 \text{ (1-move not available)} \\\ \max(e^1_{n,k},\; e^2_{n,k}) & \text{if } k = 1 \\\ \max(0,\; e^1_{n,k},\; e^2_{n,k}) & \text{if } k \geq 2 \text{ (pass available)} \end{cases}$$
 
The recursion is computed by decreasing $n$, and within each $n$ by decreasing $k$ from $\min(n, M)$ down to $0$. At the boundary $k = M$, the self-referential equations are solved algebraically before proceeding to lower $k$.
 
### 6.5 A Subtlety: Transition Probabilities After Eviction
 
One subtle point: why is the probability that a new card matches a remembered card still $k/(2n-k)$ after evictions have occurred? Because a forgotten card is indistinguishable from a never-seen card. The original shuffle is uniform, and conditioning on the current memory state (which cards are remembered with which values), the unremembered positions remain uniformly distributed. The eviction history doesn't create any bias.
 
### 6.6 Why Optimal Moves Change Even Below Capacity
 
A subtlety that initially confused me: the optimal move at $k = 3$ can differ between Zwick and the bounded model, even though $k = 3 < M = 7$ and the local transition formula is identical. The reason is that the *values* being plugged into the formula are different.
 
Think of it like a river. The riverbed (transition formulas) is the same for the first 7 miles. But Zwick's river flows another 5 miles to the sea, while the bounded model hits a dam at mile 7. The water level (values) at mile 3 is different because of the dam downstream, even though the riverbed at mile 3 is identical.
 
Concretely: $e_{12,3}$ depends on $e_{12,5}$, which depends on $e_{12,7}$. In Zwick's model, $e_{12,7}$ feeds into the deep endgame at $k = 8, 9, \ldots, 12$, including the crucial passes at $k = 9$ and $k = 11$. Under $M = 7$, $e_{12,7}$ loops back to itself. Different boundary, different values, different optimal moves.
 
## 7. Results
 
### Move Tables
 
For $n = 12$ (24 cards), the optimal move at each knowledge level $k$:
 
| | $k{=}0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ | $9$ | $10$ | $11$ | $12$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Zwick** $(M{=}\infty)$ | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | **0** | 1 | **0** | 1 |
| **Bounded** $(M{=}7)$ | 2 | 2 | 1 | **1** | 1 | **1** | 1 | **1** | — | — | — | — | — |
 
The strategies agree when you know 0, 1, or 2 cards. They diverge at $k = 3, 5, 7$: Zwick says flip two new cards (aggressive exploration), bounded-memory says flip only one (conservative). States $k \geq 8$ don't exist under $M = 7$ — your brain can't hold that many cards.
 
**In practical terms:** once you know a few card positions, stop flipping two unknowns per turn. Flip one, keep what you know, match when you can. The second unknown card just pushes out something useful.
 
### Position Values
 
Expected gain for the starting player (negative = Player 2 advantage):
 
| $M$ | $n=8$ | $n=10$ | $n=12$ | $n=16$ | $n=20$ |
|-----|-------|--------|--------|--------|--------|
| 3 | $+0.039$ | $+0.030$ | $+0.024$ | $+0.017$ | $+0.014$ |
| 5 | $-0.004$ | $+0.013$ | $+0.018$ | $+0.015$ | $+0.012$ |
| **7** | $-0.007$ | $-0.039$ | $\mathbf{-0.030}$ | $+0.007$ | $+0.010$ |
| 9 | $-0.033$ | $-0.039$ | $-0.020$ | $-0.026$ | $-0.001$ |
| $\infty$ | $-0.033$ | $-0.038$ | $-0.020$ | $-0.018$ | $-0.012$ |
 
These are exact values computed in rational arithmetic — no simulation noise. For $n = 12$ at $M = 7$, the P2 advantage is $-0.030$, which is 52% larger than Zwick's perfect-recall value of $-0.020$. Bounded memory *amplifies* the second player's advantage.
 
At very low $M$, memory is too constrained for strategy to matter and Player 1's first-mover advantage dominates. At high $M \geq n$, the model recovers Zwick exactly. The P2 advantage peaks around $M \approx 5\text{–}9$, which happens to coincide with Miller's range. I don't claim this is anything more than a coincidence — the game was designed for humans, not the other way around.

## 8. Simulations

The exact DP gives me values and optimal moves, but to see how strategies actually *perform* against each other, and to explore settings where exact computation isn't available (fluctuating capacity, asymmetric players), I run Monte Carlo simulations. All simulations use joblib for parallelisation and the exact optimal strategy tables from the DP.

### 8.1 Bounded-Memory Optimum vs Zwick

Both players have $M = 7$. I pit the bounded-memory optimal strategy against Zwick's perfect-recall strategy in all four matchup combinations, across board sizes from $n = 8$ (16 cards) to $n = 36$ (72 cards).

![Conditional win rate: bounded-memory optimum vs Zwick](../figures/miller_vs_zwick_conditional.svg)

![Expected gain: bounded-memory optimum vs Zwick](../figures/miller_vs_zwick_gain.svg)

The bounded-memory strategy dominates Zwick at every board size. The advantage is not subtle: at $n = 36$ (72 cards, a standard large game), the bounded-memory player gains about 3 full pairs over a Zwick opponent. The gain grows with board size because larger boards mean more time at $k = M$ where the strategies diverge.

When both players use the same strategy, the game is essentially fair, the tiny P2 advantage is invisible in simulation. The strategy choice matters; the seat you're in doesn't.

### 8.2 Robustness to Fluctuation

Real working memory capacity isn't fixed at exactly 7 every turn. Some turns you're sharp, others you lose focus. I model this by drawing the effective capacity each turn from $M_0 + \text{Uniform}(-\sigma, +\sigma)$, clamped to $[2, M_0 + \sigma]$.

![Fluctuation: bounded-memory optimum vs Zwick](../figures/fluctuation_miller_vs_zwick.svg)

![Fluctuation: cross-strategy gain](../figures/fluctuation_cross_gain.svg)

The bounded-memory strategy remains dominant across all fluctuation levels. Even at $\sigma = 5$ (capacity swinging between 2 and 12 each turn), the bounded-memory player still gains 2+ pairs over a Zwick opponent on large boards. This robustness makes intuitive sense: flipping one new card is safe whether your capacity happens to be 5 or 9, while Zwick's two-card flip is risky at any capacity.

### 8.3 Asymmetric Memory

The most practically relevant question: what happens when the two players have different memory capacities? Say, an adult ($M = 9$) playing against a child ($M = 5$). Each player uses the optimal strategy for their own capacity.

![Asymmetric memory: expected gain heatmap](../figures/asymmetric_gain_heatmap.svg)

![Asymmetric memory: P2 conditional win rate](../figures/asymmetric_p2cond_heatmap.svg)

![Asymmetric memory: advantage curve](../figures/asymmetric_memory_advantage.svg)

Memory capacity dwarfs positional advantage. At $n = 24$ (48 cards), a player with $M = 9$ beats a player with $M = 7$ roughly 80% of the time, regardless of who goes first. The positional advantage is worth about 0.03 pairs; a single extra memory slot is worth 1–4 pairs.

The diagonal of the P2 win rate heatmap (equal memory) shows the P2 advantage at three decimal places: 0.501, 0.504, 0.508. Real, but roughly 100$\times$ smaller than the effect of one memory slot.

### 8.4 Draw Rate vs Capacity

![Draw rate and P2 advantage vs memory capacity](../figures/draw_rate_vs_capacity.svg)

The draw rate reveals the transition between two regimes. For small boards ($n = 8$), draws spike sharply at $M \approx n$ as the optimal strategy begins to include passes and Zwick-style stalemates emerge. For standard game sizes ($n \geq 16$) with human memory ($M \approx 7$), passes never appear and draws are rare. The elaborate game of parity-control of Zwick's analysis simply doesn't arise in practice, real games produce a winner most of the time.

## 9. What I Learned

1. **Greedy matching is optimal** under shared memory — a simple dominance argument.

2. **The optimal strategy is simple:** flip two new cards only at the very start; once you know a few positions, flip only one and match when you can. Never pass. This is what Sötemann (2025) found empirically and called "defensive" play. I proved it's the exact optimum under bounded memory and showed *why*: the second flip churns memory when capacity is the bottleneck.

3. **The bounded-memory strategy dominates Zwick's** by a growing margin as board size increases, and this holds under capacity fluctuation.

4. **It (barely) matters who goes first.** Player 2 has a provable but tiny advantage, about 0.03 pairs at $n = 12$, roughly 1.5% conditional win rate. No human would ever notice.

5. **What actually matters is memory capacity.** A single extra memory slot is worth 50–100$\times$ more than the positional advantage. If you want to win at Memory, train your working memory. Or play against someone with less of it.

6. **Stalemates are a theoretical curiosity.** Under human memory constraints and standard board sizes, the optimal strategy never passes and games are almost always decisive.

The private-memory case (where each player has their own unobservable memory) remains open and is likely much harder, it's a game of incomplete information where bluffing and information hiding become possible.

## References

- Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review*, 63(2), 81–97.
- Zwick, U. & Paterson, M. S. (1993). The memory game. *Theoretical Computer Science*, 110(1), 169–196.
- Sötemann, M. (2025). Who starts the game of memory? Blog post. https://stotsi.com/posts/memory/