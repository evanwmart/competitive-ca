Here's a concrete progression, ordered so each step builds on the last.

---

**Step 1: Instrument the simulation**

Before measuring anything, you need observables. Add output modes to the existing C code that emit per-frame statistics to stderr or a separate file rather than just raw video:

- **Mean domain size** — average number of nodes with the same dominant type in a connected region. Requires a flood fill at each measurement interval.
- **Domain count** — number of distinct connected domains.
- **Boundary density** — fraction of edges that are null (incompatible). This is your primary order parameter. High boundary density = disordered. Low = ordered.
- **Dominant type distribution** — are the three types equally represented or does one take over?
- **Domain size distribution** — the full histogram, not just the mean. This is where power laws show up if the system is critical.

You don't need to measure every frame. Sample every 100 frames or so once the system has run long enough to leave the initial transient.

---

**Step 2: Establish the baseline — find the equilibrium**

For fixed parameters, run long enough that your observables stop trending and fluctuate around a stable mean. That's your equilibrium. You need to know:

- How long the transient lasts as a function of grid size
- Whether the system actually reaches equilibrium or cycles
- Whether equilibrium depends on initial conditions (if yes, you have multiple phases)

This is just careful housekeeping but it's essential. Every subsequent measurement is meaningless without knowing you're in equilibrium.

---

**Step 3: Map the phase diagram**

Vary your two main control parameters independently and measure boundary density at equilibrium:

- **reinforce_min** across its range (2, 3, 4, and fractional equivalents if you generalize it to a continuous probability)
- **mutation rate** from near-zero to high

You're looking for the boundary between an ordered phase (low boundary density, large stable domains) and a disordered phase (high boundary density, small unstable domains). Plot boundary density as a heatmap over this 2D parameter space. The transition region is where things get interesting.

Making reinforce_min continuous is worth doing — instead of "must have exactly K agreeing neighbors," make it a probability that increases with the number of agreeing neighbors. This lets you sweep continuously rather than in integer steps.

---

**Step 4: Characterize the transition**

Once you've located the transition region, zoom in. This is the core physics work.

At a continuous phase transition, several things happen simultaneously:
- The order parameter (boundary density) changes continuously, not abruptly
- **Fluctuations diverge** — variance of boundary density spikes at the critical point
- **Correlation length diverges** — domains become correlated over arbitrarily long distances
- **Domain size distribution becomes a power law** — instead of a characteristic size, you get scale-free domains

Measure all of these as you sweep through the transition. The power law in the domain size distribution is the clearest signal — if you plot domain size vs. frequency on a log-log scale and get a straight line at the critical point, you've found criticality.

The critical exponents (the slopes and scaling relationships between these diverging quantities) are what characterize the universality class. You extract them by finite-size scaling — running the same sweep on grids of different sizes and measuring how the transition sharpens with system size.

---

**Step 5: Compare to known universality classes**

Once you have critical exponents, look them up. The main candidates:

- **Voter model** — the simplest opinion dynamics model, exactly solvable in 2D, known exponents
- **2D Potts model** — the most natural comparison given your 3-type system; the q=3 Potts model has known exact exponents
- **Directed percolation** — relevant if your transition involves an absorbing state (a frozen configuration the system can't escape)
- **Ising** — less likely given 3 types but worth checking

If your exponents match a known class, that tells you something deep: your system, despite its specific rules, belongs to the same universality class as simpler models. The microscopic details don't matter at the transition. If they *don't* match any known class, that's more interesting — you may have a new universality class, which is a significant finding.

---

**Step 6: Vary vocabulary size**

Everything above is for 3 types. Now repeat the phase diagram mapping for larger vocabularies — 4, 6, 8, 16 types. Questions to answer:

- Does the critical point shift as vocabulary grows?
- Do the critical exponents change, or does the system stay in the same universality class?
- Is there a vocabulary size above which the system never orders, because no type can achieve dominance?

The last question in particular connects back to the attractor capacity question and has a clean theoretical prediction you can test.

---

**Tools you'll need**

The C simulation is fine for generating frames but you'll want a separate analysis layer. Python with NumPy is the natural choice — read the raw frame output, compute observables, store time series, plot. You might also add a headless mode to the C code that skips video output entirely and just dumps statistics, which will run significantly faster.

For finite-size scaling and critical exponent extraction, there are standard techniques in the statistical physics literature — the Binder cumulant method is the cleanest for locating the critical point precisely. That's worth reading about before Step 4.

---

**What success looks like**

At the end of this, you'll have a phase diagram with a characterized transition, a set of measured critical exponents, and a classification of the universality class. That's a complete, self-contained result. It either confirms the system is in a known class (interesting, tells you the dynamics are equivalent to something simpler) or reveals something new (more interesting, publishable on its own terms).

The whole thing is tractable as a solo project. Steps 1-3 are engineering. Steps 4-6 are the actual science.
