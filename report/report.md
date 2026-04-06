---
title: "The Memory Game"
permalink: /projects/memory/
layout: single
mathjax: true
sidebar:
  nav: "projects"
toc: true
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    TeX: {
      extensions: ["bm.js", "AMSmath.js", "AMSsymbols.js"]
    },
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    }
  });
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<h2 data-toc-skip>The Optimal Strategy for Memory Under Bounded Working Memory</h2>
**Taulant Koka · March 2026 · [GitHub: memory-game](https://github.com/taulantkoka/memory-game)**

## 1. The Game

Memory (also known as Concentration or Pairs) is a card game played with $n$ pairs of identical cards ($2n$ cards total), shuffled and placed face down on a table. Players alternate turns. On each turn, a player flips two cards face up. If the two cards match, the player takes the pair and plays again. If they do not match, both cards are flipped back face down and the turn passes to the opponent. The player with the most pairs at the end wins.

Despite being a children's game, Memory has a surprisingly rich strategic structure. Every card flip reveals information to *both* players, and the key decision is how to balance learning (flipping new cards) against exploiting what you already know (flipping a card whose match you remember). This raises a natural question: **does it matter who goes first?**

If you want to skip the theory and just play, there's an [interactive game](#9-play-against-the-bot) at the end where you can take on the optimal strategy yourself.

## 2. What Zwick and Paterson Showed

In 1993, Uri Zwick and Michael Paterson published the definitive analysis of Memory under the assumption that both players have *perfect memory*: they remember the identity and position of every card ever flipped.

Their key insight was to characterise each game position by just two numbers:
- $n$: the number of pairs remaining on the board
- $k$: the number of cards both players have previously inspected (and remember perfectly)

At each turn, the player chooses one of three moves:

- **0-move (pass):** Flip two already-known, non-matching cards. The turn passes to the opponent.
- **1-move:** Flip one new card. If it matches a remembered card, take the pair and play again. If not, "waste" the second flip by re-flipping a card you already know.
- **2-move:** Flip one new card. If it matches a remembered card, take the pair. If not, flip a *second* new card. If the two new cards happen to match, take the pair. Otherwise, the turn passes.

Through backward induction on the state space $(n,k)$ (solving the game by working backwards from the end, where the optimal play is obvious, to the beginning), they computed the exact optimal strategy. The results are surprising:

1. **Player 2 has a slight advantage** for even $n \ge 8$, of magnitude about $1/(4n)$ pairs.
2. **The key strategic weapon is the pass.** At high $k$, both players know where most pairs are but deliberately refrain from taking them, engaging in a delicate game of parity control.
3. **Stalemates occur** in a substantial fraction of games under optimal play, when both players pass indefinitely.

This is elegant, but it immediately raises the question: does any of this survive when players do not have perfect memory?

## 3. First Attempt: Stochastic Memory Decay

My first approach, inspired by [Samuel Kilian](https://www.youtube.com/shorts/MPXQDAMsmro), was to model forgetting as probabilistic decay: each turn, every remembered card has a probability $\delta$ of being forgotten. This is analytically convenient, the Zwick framework mostly carries over, and it lets you sweep continuously from perfect memory ($\delta=0$) to complete amnesia ($\delta=1$).

I ran large-scale Monte Carlo simulations under this model. The results were interesting: at low $\delta$, the Zwick Player 2 advantage survives. Around $\delta \approx 0.10\text{--}0.15$, there is a phase transition where the Zwick stalemate regime collapses and the game becomes more decisive. At high $\delta$, the game converges to a coin flip.

But the model bothered me. Probabilistic decay means you might remember a card you saw 20 turns ago while forgetting one you saw 2 turns ago. That does not match my experience of how memory works in this game. You either correctly remember where a card is or you do not. And when your brain is full and you see something new, something old gets pushed out. The bottleneck feels more like *capacity* than *reliability*.

## 4. From Miller's Law to a Tractable Model

This pointed me toward George Miller's famous 1956 paper on the "magical number seven, plus or minus two". Later work has revised the effective capacity downward, with Cowan (2001) arguing for something closer to 4 chunks under stricter definitions, and the exact number depends heavily on chunking and task design. For this project I use $M=7$ as a convenient human-scale benchmark, not as a literal psychological constant. The important point is not the precise number but the qualitative fact: working memory has a hard capacity limit, and it is much smaller than a typical Memory board.

Of course, real human memory is more complex than a fixed-size buffer. Decay, interference, attention, emotional salience, rehearsal: they all play a role. A fully realistic model would need to account for all of these, and would almost certainly be intractable.

So I opted for a deliberate simplification: a deterministic bounded-memory model that captures the capacity constraint while remaining analytically solvable. Each player has a working memory of capacity $M$ (number of card positions). When memory is full and a new card is observed, the **least recently used** entry is evicted.

This is not meant to be a faithful model of human cognition. It is the simplest model that has the property I care about: memory is bounded, and new information displaces old information.

The reason this simplification works so well turns out to be mathematical rather than psychological.

**Key property:** since both players see every flip and both use the same deterministic eviction rule, **both players have identical memory at all times**. The memory state is common knowledge. So the game remains one of perfect information, the same class of game Zwick analysed. All that changes is the state space.

Any stochastic model of forgetting (decay, random interference, attention fluctuation) would give the two players *different* memories, creating private information and turning the game into something closer to poker. That is a much harder problem. The deterministic LRU model sidesteps this and keeps backward induction available.

At the level of the full game, a state consists of:
- the unmatched cards remaining on the board,
- whose turn it is,
- the ordered shared LRU memory list.

The next section shows why, after one key structural theorem, this can be reduced to a much smaller state description.

## 5. Greedy Matching Is Optimal

Before computing the dynamic program, I need to settle a structural question: if the shared memory already contains both positions of some matching pair, is it ever optimal to leave that pair on the board?

The short answer is no. Under shared memory, if you can see a pair, so can your opponent. Deferring it either hands it to your opponent (who takes it) or risks it being forgotten (evicted from memory). Neither outcome is better than just taking it now.

### Theorem

*If both players can see a matching pair in shared memory, taking it immediately is always at least as good as any other move.*

More precisely: in the bounded-memory model with deterministic LRU eviction, suppose the shared memory contains both positions of some matching pair $P$. Then no legal action that leaves $P$ on the board can yield a higher expected payoff than taking $P$ at once. This is weak dominance: if two pairs are visible simultaneously, taking either one first may be equally good, but deferring a known pair is never strictly better.

<details markdown="1" class="notice">
<summary>Proof</summary>

Write $s=(B,\pi,L)$ for a full game state, where $B$ is the set of unmatched cards, $\pi$ is the player to move, and $L$ is the ordered shared LRU memory list. For any legal action $a$, let $Q(s,a)$ be the value to the player to move of taking action $a$ in state $s$ and then playing optimally thereafter, and let

$$
V(s)=\max_a Q(s,a).
$$

Assume $\pi=A$, and that $L$ contains both $\alpha,\beta$ of the matching pair $P=\{\alpha,\beta\}$.

Let $a$ be any legal action by $A$ that does not take $P$ immediately. We compare:

- the **deferred** line, in which $A$ plays $a$ from $s$;
- the **immediate-take** line, in which $A$ first takes $P$, moving to

  $$
  s^+ := T_P(s),
  $$

  and then both players play optimally.

Thus

$$
Q(s,\text{take }P)=1+V(s^+).
$$

The key observation is that $s^+$ is obtained from $s$ by deleting $\alpha,\beta$ from both the board and the shared memory.

**Lemma (LRU monotonicity).** Let $L$ be an LRU memory list, and let $L'$ be obtained from $L$ by deleting some entries. Suppose both lists are then updated by the same sequence of observations

$$
x_1,x_2,\dots,x_t,
$$

none of which is one of the deleted entries. Then after every prefix $x_1,\dots,x_r$, the updated list $L'_r$ is obtained from $L_r$ by deleting some subset of the originally deleted entries. In particular:

1. every non-deleted card remembered in $L_r$ is also remembered in $L'_r$;
2. the relative LRU order of the non-deleted remembered cards is the same in both lists.

*Proof of lemma.* By induction on $r$.

For $r=0$ this is true by construction. Assume it holds at step $r$, and consider the next observation $x_{r+1}$.

- If $x_{r+1}$ is already present in both lists, both move it to the MRU end.
- If $x_{r+1}$ is absent from both lists, both insert it. If no eviction occurs, the relation is preserved. If eviction occurs, $L_r$ evicts its LRU entry; $L'_r$, being obtained from $L_r$ by deleting entries, either evicts the same entry or one already deleted.
- If $x_{r+1}$ is present in $L'_r$, then by the induction hypothesis it is also present in $L_r$, so this reduces to the first case.
- The case where $x_{r+1}$ is present in $L_r$ but absent from $L'_r$ cannot occur, because deleted entries are excluded from the observation sequence.

So the invariant is preserved for all $r$. $\square$

Apply the lemma with the deleted entries equal to $\alpha,\beta$. It follows that for any common sequence of subsequent observations of non-$P$ cards, the immediate-take state $s^+$ is never worse informed about the remaining board than the deferred state.

Now let $\tau$ be the first time in the deferred play at which one of three things happens: $A$ takes $P$, $B$ takes $P$, or one of $\alpha,\beta$ is evicted from memory:

**Case 1: $B$ takes $P$ at time $\tau$.** Then the deferred line has allowed the opponent to score a publicly known pair that $A$ could have taken immediately. Relative to the immediate-take line, $A$ is down one pair and, by the lemma, is not better informed about the remaining board. Hence

$$
Q(s,\text{take }P)\;>\;Q(s,a).
$$

**Case 2: one of $\alpha,\beta$ is evicted before $P$ is taken.** Then the deferred line has failed to bank an immediately available point and has weakly reduced future information. The immediate-take line has already scored the point and is weakly better informed on the remaining board. So again

$$
Q(s,\text{take }P)\;>\;Q(s,a).
$$

**Case 3: $A$ takes $P$ at time $\tau$.** At that moment, both lines have removed the same pair $P$, and the remaining board is identical. Let $s_\tau^{\mathrm{def}}$ and $s_\tau^{\mathrm{imm}}$ be the resulting full states after removal of $P$ in the deferred and immediate-take lines respectively. By the LRU monotonicity lemma,

$$
s_\tau^{\mathrm{imm}}
\quad\text{is weakly better informed than}\quad
s_\tau^{\mathrm{def}}
$$

on the remaining board. Therefore

$$
V\!\left(s_\tau^{\mathrm{imm}}\right)\;\ge\;V\!\left(s_\tau^{\mathrm{def}}\right),
$$

and hence

$$
Q(s,\text{take }P)\;\ge\;Q(s,a).
$$

In all three cases,

$$
Q(s,\text{take }P)\;\ge\;Q(s,a)
\qquad
\text{for every legal }a\text{ that leaves }P\text{ on the board}.
$$

So taking a publicly known pair immediately is weakly dominant. $\square$

**Remark.** This proof uses the fact that memory is shared and publicly observable. With private memory, A could know a pair that B does not know about. In that setting, holding the pair in reserve could be genuinely strategic.

</details>

### Corollary

We may restrict attention to optimal strategies that never leave a publicly known pair unmatched at a decision point. Hence, in reduced states, the shared memory contains only unmatched singletons, one from each of $k$ distinct pairs. The value of a reduced state depends only on $(n,k)$, not on the specific card identities or their LRU order, because all singletons are exchangeable under relabeling. So the game state reduces to the pair $(n,k)$, and the full-state value $V(s)$ becomes the scalar $e_{n,k}$.

<details markdown="1" class="notice">
<summary>Why does value depend only on (n, k)?</summary>

Among reduced states with $n$ remaining pairs and $k$ remembered singleton positions, the value depends only on $(n,k)$.

*Sketch.* Once all remembered cards are unmatched singletons with distinct values, the only quantities that affect the transition law are:

- the number $k$ of remembered singleton positions;
- the number $2n-k$ of unknown positions;
- the number of remembered singleton positions removed or added in each branch.

The specific labels of the remembered cards are exchangeable under relabeling of pair identities. Moreover, when a known card is used as the idle flip in a 1-move or as part of a 0-move, the resulting continuation value depends only on the updated count.

Thus the DP closes on $(n,k)$, and $V(s) = e_{n,k}$ where $(n,k)$ is the reduced state corresponding to $s$.

</details>

## 6. The Optimal Strategy

With greedy matching established, I can compute the optimal strategy by backward induction. The state space is $(n,k)$ where $n$ is the number of pairs left and $k \le \min(n,M)$ is the number of card positions currently in shared memory. At each state, the player chooses between three moves (pass, flip one new card, or flip two new cards) and picks whichever yields the highest expected payoff.

I write $e_{n,k}$ for the value of state $(n,k)$ to the player whose turn it is. Positive means the current player is ahead; negative means the opponent is.

The base cases are straightforward: $e_{0,0}=0$ (no cards left, no points to gain), and $e_{n,n}=n$ whenever $n \le M$ (every remaining pair is known, so the current player takes them all).

**Observation convention.** When does a flipped card enter working memory? Only when its outcome is *unresolved*. If a card completes a pair the moment it is revealed, whether by matching the first card (lucky match) or a singleton the opponent recognises (auto-take), the pair is removed immediately and neither card needs to be stored for future recall. Only unresolved cards occupy a memory slot. This convention matters at the boundary $k=M$: a lucky match after one eviction leaves $M-1$ old singletons in memory, not $M-2$.

### 6.1 The three moves (when memory is not full, $k < M$)

Let $p = k/(2n-k)$ be the probability that a newly flipped card matches one already in memory, and let $q = 1-p$.

- **0-move (pass):** flip two cards you already know don't match. Nothing changes; the opponent inherits the same state. Value: $0$. Only legal when $k \ge 2$.
- **1-move:** flip one new card. If it matches a remembered card (prob $p$), take the pair and play again from $(n{-}1, k{-}1)$. If not (prob $q$), waste your second flip on a card you already know, and the opponent faces $(n, k{+}1)$.
- **2-move:** flip one new card. If it matches a remembered card, take the pair. If not, flip a second new card, which leads to one of three outcomes.

<details markdown="1" class="notice">
<summary>Full formulas for k < M</summary>

**0-move.**

$$
e^0_{n,k}=-e_{n,k}.
$$

Including the pass among the candidate moves is equivalent to clamping the value at zero: if both the 1-move and 2-move are negative, the player passes.

**1-move.**

$$
e^1_{n,k}
=
p\bigl(1+e_{n-1,k-1}\bigr)-q\,e_{n,k+1}.
$$

On a match, you score a point and continue with one fewer pair and one fewer singleton in memory. On a miss, the new card enters memory (raising $k$ by one) and the opponent moves.

**2-move.** Suppose the first card misses. It is now in memory, so there are $k+1$ remembered positions and $d := 2n-k-1$ remaining unknowns. The second card is drawn uniformly from these $d$ unknowns:

- **Lucky match** (prob $\frac{1}{d}$): the second card matches the first. You take the pair and continue from $(n-1, k)$.
- **Auto-take** (prob $\frac{k}{d}$): the second card matches a *different* remembered singleton. The opponent immediately takes that pair, scoring a point and continuing from $(n-1,k)$ in their frame, which is $-(1 + e_{n-1,k})$ in yours.
- **Double miss** (prob $\frac{2(n-k-1)}{d}$): no match at all. Both new cards enter memory and the opponent faces $(n, k+2)$.

A note on the auto-take bookkeeping: the matched singleton and the second new card are both removed from memory ($-2$), but the first new card stays ($+1$). Net change in $k$: zero. Hence the continuation state is $(n-1,k)$, not $(n-1,k-1)$.

Putting it together:

$$
e^2_{n,k}
=
p\bigl(1+e_{n-1,k-1}\bigr)
+
q\left[
\frac{1}{d}\bigl(1+e_{n-1,k}\bigr)
-\frac{k}{d}\bigl(1+e_{n-1,k}\bigr)
-\frac{2(n-k-1)}{d}\,e_{n,k+2}
\right].
$$

</details>

### 6.2 What changes when memory is full ($k = M$)

When memory is already full, flipping a new card triggers LRU eviction: the new card enters and the oldest singleton is pushed out. This changes the recurrence in two ways.

First, a miss no longer increases $k$. Instead of advancing to $(n, M+1)$, the state **loops back to $(n, M)$**. This makes the equations self-referential, but they can still be solved in closed form.

Second, eviction changes the number of unknowns available for the second flip in a 2-move. Below capacity, the first miss turns one unknown into a known card, reducing the unknown count by one (from $2n-k$ to $2n-k-1$). At the boundary, the evicted card re-enters the unknown pool at the same time, so the two effects cancel: the number of unknowns **stays at $d := 2n - M$**.

After the first miss, memory holds $M-1$ old singletons plus the first new card. Only those $M-1$ old singletons can be matched by the second card (the evicted one is no longer remembered). So the auto-take probability drops from $M/d$ to $(M-1)/d$, and both the lucky-match and auto-take branches lead to $(n-1, M-1)$ rather than $(n-1, M)$.

<details markdown="1" class="notice">
<summary>Full boundary formulas at k = M</summary>

**1-move at $k=M$.** A miss evicts the oldest singleton and loops back to $(n,M)$:

$$
e^1_{n,M}
=
p\bigl(1+e_{n-1,M-1}\bigr)-q\,e_{n,M}.
$$

If the 1-move turns out to be optimal at this state, then $e_{n,M}=e^1_{n,M}$ and we can solve directly:

$$
e^1_{n,M}
=
\frac{p\bigl(1+e_{n-1,M-1}\bigr)}{1+q}.
$$

**2-move at $k=M$.** After a first-card miss (with eviction), the second card is drawn from $d = 2n-M$ unknowns:

- **Lucky match** (prob $\frac{1}{d}$): the second card matches the first. Take the pair; $M-1$ old singletons remain. The state is $(n-1, M-1)$, not $(n-1, M-2)$, because the second card is acted on immediately and never enters memory (see observation convention above).
- **Auto-take** (prob $\frac{M-1}{d}$): the second card matches one of the $M-1$ remaining old singletons. The opponent takes that pair. State: $(n-1, M-1)$.
- **Double miss** (prob $\frac{2(n-M)}{d}$): no match. Memory is still full; the state loops back to $(n, M)$.

Sanity check: $1 + (M-1) + 2(n-M) = 2n - M = d$. $\checkmark$

$$
e^2_{n,M}
=
p\bigl(1+e_{n-1,M-1}\bigr)
+
q\left[
\frac{1}{d}\bigl(1+e_{n-1,M-1}\bigr)
-\frac{M-1}{d}\bigl(1+e_{n-1,M-1}\bigr)
-\frac{2(n-M)}{d}\,e_{n,M}
\right].
$$

Collecting $e_{n,M}$ on the left:

$$
e^2_{n,M}
=
\frac{
\left(p+\frac{q(2-M)}{d}\right)\bigl(1+e_{n-1,M-1}\bigr)
}{
1+\frac{2q(n-M)}{d}
}.
$$

Every term on the right involves $n-1$ pairs, which have already been computed. This closed-form boundary equation replaces Zwick's deep-$k$ recursion.

</details>

### 6.3 Optimal move selection

At each state $(n,k)$, the player picks the move with the highest value:

$$
e_{n,k}
=
\begin{cases}
e^2_{n,0}, & k=0,\\[4pt]
\max\bigl\{e^1_{n,k},\,e^2_{n,k}\bigr\}, & k=1,\\[4pt]
\max\bigl\{0,\,e^1_{n,k},\,e^2_{n,k}\bigr\}, & k\ge 2.
\end{cases}
$$

At $k=0$ there is nothing in memory to match, so the 1-move is unavailable. The pass also requires at least two known cards, so only the 2-move remains. At $k=1$ the pass is still unavailable. For $k \ge 2$ the pass (value $0$) enters the candidate set.

The recursion is evaluated by increasing $n$, and for each $n$ by decreasing $k$ from $\min(n,M)$ down to $0$. This ordering ensures that every value on the right-hand side has already been computed, except for the self-referential boundary terms at $k = M$, which are resolved in closed form as shown above.

<details markdown="1" class="notice">
<summary>Two technical subtleties</summary>

**Why eviction does not bias match probabilities.** One might worry that forgetting cards changes the distribution of what remains. It does not: conditional on the current reduced state, a forgotten card is indistinguishable from one that was never seen. The original shuffle is uniform, and the unseen portion of the board remains exchangeable regardless of eviction history. So the match probability is still $k/(2n-k)$.

**Why optimal moves can change even below capacity.** The formulas for $k < M$ are identical in the bounded and Zwick models. Yet the optimal move at, say, $k=3$ can differ between the two. The reason is that the *values* plugged into those formulas are different, because they ultimately depend on what happens at the boundary.

An analogy: the riverbed is the same for the first 7 miles, but Zwick's river flows on to the sea while the bounded model hits a dam at mile 7. The dam changes the water level everywhere upstream.

Concretely, $e_{12,3}$ depends on $e_{12,5}$, which depends on $e_{12,7}$. In Zwick's model, $e_{12,7}$ feeds into $k=8,9,\dots,12$, including the pass equilibrium. Under $M=7$, $e_{12,7}$ loops back into the boundary recursion instead. Same local formula, different downstream values, different optimal move.

</details>

## 7. Results

### Move tables

The tables below show the optimal move at each knowledge level $k$ for several board sizes. Bold entries mark where the bounded-memory strategy differs from Zwick.

**$n = 8$ (16 cards):**

| | $k{=}0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ |
|---|---|---|---|---|---|---|---|---|---|
| **Zwick** $(M{=}\infty)$ | 2 | 2 | 1 | 2 | 1 | 2 | 1 | **0** | 1 |
| **Bounded** $(M{=}7)$ | 2 | 2 | 1 | 2 | 1 | **0** | 1 | **1** | --- |

**$n = 10$ (20 cards):**

| | $k{=}0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ | $9$ | $10$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Zwick** $(M{=}\infty)$ | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | **0** | 1 |
| **Bounded** $(M{=}7)$ | 2 | 2 | 1 | 2 | 1 | **1** | 1 | **1** | --- | --- | --- |

**$n = 12$ (24 cards):**

| | $k{=}0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ | $9$ | $10$ | $11$ | $12$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Zwick** $(M{=}\infty)$ | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | **0** | 1 | **0** | 1 |
| **Bounded** $(M{=}7)$ | 2 | 2 | 1 | **1** | 1 | **1** | 1 | **1** | --- | --- | --- | --- | --- |

**$n = 16$ (32 cards):**

| | $k{=}0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ |
|---|---|---|---|---|---|---|---|---|
| **Zwick** $(M{=}\infty)$ | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| **Bounded** $(M{=}7)$ | 2 | **1** | 1 | **1** | 1 | **1** | 1 | **1** |

**$n = 20$ (40 cards):**

| | $k{=}0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ |
|---|---|---|---|---|---|---|---|---|
| **Zwick** $(M{=}\infty)$ | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| **Bounded** $(M{=}7)$ | 2 | **1** | 1 | **1** | 1 | **1** | 1 | **1** |

For $n=16$ and $n=20$, the Zwick columns are truncated at $k=7$ for comparison; the full tables extend to $k=n$ with the familiar alternating 2/1 pattern and passes near the top.

A clear pattern emerges as the board grows: the bounded-memory strategy converges to "flip two unknowns only at $k=0$; flip one everywhere else." The second unknown card mostly churns memory, and the cost of that churn increases with board size.

**In practical terms:** once you know a few card positions, stop flipping two unknowns per turn. Flip one, keep what you know, and match when you can.

### Position values

Expected gain for the starting player (negative = Player 2 advantage):

| $M$ | $n=8$ | $n=10$ | $n=12$ | $n=16$ | $n=20$ |
|-----|-------|--------|--------|--------|--------|
| 3 | $+0.039$ | $+0.030$ | $+0.024$ | $+0.017$ | $+0.014$ |
| 5 | $-0.005$ | $+0.013$ | $+0.018$ | $+0.015$ | $+0.012$ |
| **7** | $-0.007$ | $-0.039$ | $\mathbf{-0.030}$ | $+0.007$ | $+0.010$ |
| 9 | $-0.033$ | $-0.039$ | $-0.020$ | $-0.026$ | $-0.001$ |
| $\infty$ | $-0.033$ | $-0.038$ | $-0.020$ | $-0.018$ | $-0.012$ |

These are exact values computed in rational arithmetic, with no simulation noise. For $n=12$ at $M=7$, the Player 2 advantage is $-0.030$, larger in magnitude than Zwick's perfect-recall value of $-0.020$.

At very low $M$, memory is too small for strategy to matter and Player 1's first-mover advantage dominates. At high $M \ge n$, the model recovers Zwick exactly.
## 8. Simulations

The exact DP gives values and optimal moves, but simulations show how strategies perform against one another and let me explore settings where exact analysis is unavailable, such as fluctuating capacity and asymmetric players.

### 8.1 Bounded-memory optimum vs Zwick

Both players have $M=7$. I pit the bounded-memory optimal strategy against Zwick's perfect-recall strategy in all four matchup combinations, across board sizes from $n=8$ to $n=36$.

<a href="/figures_memory/bounded_vs_zwick_conditional.svg" class="image-popup">
  <img src="/figures_memory/bounded_vs_zwick_conditional.svg" alt="Conditional win rate: Bounded vs Zwick"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>

<a href="/figures_memory/bounded_vs_zwick_gain.svg" class="image-popup">
  <img src="/figures_memory/bounded_vs_zwick_gain.svg" alt="Expected gain: Bounded vs Zwick"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>

Against opponents using Zwick's policy, the bounded-memory-optimal policy wins across all tested board sizes. At $n=36$ (72 cards), the bounded-memory player gains roughly 3 full pairs over a Zwick opponent. The gain grows with board size because larger boards spend more time near the full-memory boundary where the two strategies differ.

The strategy matrix at $n=16$ shows that bounded-memory optimal dominates Zwick within this two-strategy comparison: no matter which of the two strategies your opponent plays, you are better off playing bounded. (The DP proves optimality within the full class of 0/1/2-move strategies; the simulation confirms it specifically against Zwick.)

<a href="/figures_memory/bounded_vs_zwick_matrix.svg" class="image-popup">
  <img src="/figures_memory/bounded_vs_zwick_matrix.svg" alt="Strategy matrix"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>

### 8.2 Robustness to fluctuation

Real working-memory capacity is not fixed at exactly 7 every turn. I model this by drawing the effective capacity each turn from $M_0+\mathrm{Uniform}(-\sigma,+\sigma)$, clamped to a feasible range.

<a href="/figures_memory/fluctuation_bounded_vs_zwick.svg" class="image-popup">
  <img src="/figures_memory/fluctuation_bounded_vs_zwick.svg" alt="Bounded-optimal vs Zwick under fluctuation"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>

<a href="/figures_memory/fluctuation_cross_gain.svg" class="image-popup">
  <img src="/figures_memory/fluctuation_cross_gain.svg" alt="Cross-strategy gain under fluctuation"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>

The bounded-memory strategy remains dominant across all fluctuation levels. Even when capacity varies widely, flipping one new card is robust, while Zwick's two-card exploration remains fragile because it churns memory.

### 8.3 Asymmetric memory

What happens when the two players have different capacities? Say, an adult ($M=9$) against a child ($M=5$). Each player uses the optimal strategy for their own capacity.

<a href="/figures_memory/asymmetric_gain_heatmap.svg" class="image-popup">
  <img src="/figures_memory/asymmetric_gain_heatmap.svg" alt="Asymmetric memory: expected gain heatmap"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>
<a href="/figures_memory/asymmetric_p2cond_heatmap.svg" class="image-popup">
  <img src="/figures_memory/asymmetric_p2cond_heatmap.svg" alt="Asymmetric memory: P2 conditional win rate"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>
<a href="/figures_memory/asymmetric_memory_advantage.svg" class="image-popup">
  <img src="/figures_memory/asymmetric_memory_advantage.svg" alt="Asymmetric memory: advantage curve"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>

Memory capacity dwarfs positional advantage. A single extra memory slot is worth far more than the tiny first- or second-player effect.

With asymmetric memory, however, the shared-memory theorem no longer applies literally: the two players no longer have the same information state. So in that setting, these strategies should be understood as strong heuristics rather than proven optima.

### 8.4 Draw rate vs capacity

<a href="/figures_memory/draw_rate_vs_capacity.svg" class="image-popup">
  <img src="/figures_memory/draw_rate_vs_capacity.svg" alt="Draw rate and P2 advantage vs memory capacity"
     style="background: #fff; border-radius: 6px; cursor: zoom-in;">
</a>

The draw rate shows a striking non-monotonic pattern. For small boards, passes can create sticky stalemates at certain capacities; once memory becomes large enough to cover the whole board, those stalemates disappear because one player eventually sweeps everything.

For standard game sizes with human-scale memory ($M\approx 7$), $M$ is far below $n$, passes typically disappear from the optimal strategy, and draws are rare.

## 9. Play Against the Bot

Theory is nice, but the best way to feel the difference between strategies is to play. The game below lets you play Memory against a bot that uses either the bounded-memory optimal strategy or Zwick's perfect-recall strategy. You can adjust the bot's memory capacity and see which cards it remembers in real time.

Try this experiment: play a few games against the bounded-memory bot ($M=7$), then switch to Zwick. With the bounded strategy, the bot flips one unknown card and keeps its memories stable. With Zwick, it flips two unknowns and old memories get evicted. The Zwick bot *looks* smarter because it explores faster, but on bounded memory it performs worse.

<iframe src="/assets/memory_game.html" width="100%" height="750" style="border:none; border-radius:8px; overflow:hidden;"></iframe>

## 10. Code & Reproducibility

The reference implementations are all available on GitHub:

**[GitHub: memory-game](https://github.com/taulantkoka/memory-game)**

### Quick Start
```bash
git clone https://github.com/taulantkoka/memory-game.git
cd memory-game
pip install numpy matplotlib joblib

# Run everything (~30 min at 100k games/point)
python run_analysis.py

# Run a single analysis
python run_analysis.py --only 01

# List available analyses
python run_analysis.py --list
```

## 11. What I Learned

1. **Greedy matching is optimal** under shared memory: leaving a publicly known pair on the board is weakly dominated by taking it immediately.
2. **The bounded-memory optimum is simple:** flip two new cards only at the very start; once you know a few positions, flip only one and match when you can.
3. **The bounded-memory strategy dominates Zwick's** on large boards, and the gap grows with size.
4. **Who goes first barely matters.** The positional effect is real but tiny.
5. **What really matters is memory capacity.** A single extra memory slot is worth much more than move order.
6. **Stalemates are mostly a perfect-memory phenomenon.** Under human-scale memory limits and standard board sizes, they largely disappear.

The private-memory version of the game remains open and looks much harder: once players remember different things, the problem becomes one of incomplete information rather than perfect information.

## References

- Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, 24(1), 87-114.
- Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review*, 63(2), 81-97.
- Zwick, U., & Paterson, M. S. (1993). The memory game. *Theoretical Computer Science*, 110(1), 169-196.