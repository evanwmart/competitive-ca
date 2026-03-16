# Self-Assembling Knowledge Graph — Test Plan

## Overview

Use the existing CA dynamic graph infrastructure with fixed node states (embeddings) and
adaptive edges to test whether local competition dynamics recover known community structure
without global supervision. The control parameter (similarity threshold) maps onto mutation
rate. The phase transition is the core result to look for.

---

## Phase 1 — Synthetic Baseline (No Embeddings)

**Goal:** Confirm the CA recovers known community structure before introducing real data.

**Setup:**
- Use a stochastic block model (SBM) — a synthetic graph where ground truth communities
  are known by construction
- Node states: one-hot community label vectors (e.g. 5 communities → 5-dimensional vector)
- Edge initialization: random, ignoring ground truth
- Competition rule: cosine similarity between node state vectors
- Formation: if similarity > threshold, maybe_form edge
- Severance: if similarity < threshold, maybe_sever edge

**Tools:**
```python
import networkx as nx
# Generate SBM with 5 communities of 40 nodes each
sizes = [40, 40, 40, 40, 40]
p_in = 0.7   # within-community edge probability
p_out = 0.05 # between-community edge probability
G = nx.stochastic_block_model(sizes, [[p_in if i==j else p_out 
                                        for j in range(5)] 
                                        for i in range(5)])
```

**What to measure:**
- Modularity of final graph vs ground truth
- Normalized mutual information (NMI) between recovered clusters and true labels
- Whether phase transition appears as threshold varies
- Seed variance at the transition point (expect bimodal — first-order signal)

**Success criterion:** NMI > 0.8 across majority of seeds at low threshold. Phase transition
visible in modularity vs threshold curve.

---

## Phase 2 — Word Embeddings on Small Labeled Dataset

**Goal:** Test on real semantic content with known ground truth categories.

**Dataset construction:**
- 200 words across 5 clear domains:
  - Physics: gravity, entropy, photon, quark, relativity, spacetime, boson, fermion...
  - Cooking: sauté, simmer, emulsify, roux, mirepoix, julienne, blanch, braise...
  - Music: harmony, cadence, timbre, arpeggio, counterpoint, diminuendo, vibrato...
  - Sports: trajectory, momentum, stamina, agility, sprint, lateral, pivot, endurance...
  - Biology: mitosis, osmosis, chlorophyll, synaptic, genome, ribosome, cortex...
- 40 words per domain, chosen to minimize cross-domain ambiguity initially

**Embeddings:**
```python
import gensim.downloader as api
model = api.load("word2vec-google-news-300")  # 300-dimensional vectors, pretrained

nodes = {}
for word in word_list:
    if word in model:
        nodes[word] = model[word]  # fixed, never updated
```

**Competition rule:**
```python
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def should_form(node_a, node_b, threshold):
    return cosine_similarity(nodes[node_a], nodes[node_b]) > threshold

def should_sever(node_a, node_b, threshold):
    return cosine_similarity(nodes[node_a], nodes[node_b]) < threshold
```

**What to measure:**
- Same as Phase 1 (modularity, NMI vs ground truth labels)
- Phase transition location — at what threshold does structure emerge?
- Seed variance at transition — does first-order behavior persist with real embeddings?
- Bridge nodes — words like "entropy" (physics + information theory) should appear at
  community boundaries, not deep interior. Check this explicitly.

---

## Phase 3 — Ambiguous / Cross-Domain Words

**Goal:** Test how the CA handles semantic ambiguity — the interesting case sorting can't
handle cleanly.

**Add deliberately ambiguous words to the dataset:**
- "entropy" — physics AND information theory
- "frequency" — physics AND music
- "pitch" — music AND sports (baseball)
- "energy" — physics AND everyday language
- "culture" — biology AND sociology
- "field" — physics AND sports AND agriculture

**What to look for:**
- Do ambiguous nodes become bridge nodes between communities?
- Do they get pulled into one community deterministically or oscillate?
- Does their final assignment depend on initialization (seed sensitivity)?
- At the transition point, are ambiguous nodes the last to resolve?

This is where the CA should do something clustering algorithms can't — reveal the
topology of ambiguity rather than forcing a hard assignment.

---

## Phase 4 — Hysteresis Test on Knowledge Graph

**Goal:** Confirm first-order behavior in semantic domain.

**Setup:**
- Run same dataset from two initializations at threshold in coexistence region:
  1. Random edge initialization
  2. k-NN initialization (each node starts connected to its k most similar nodes)
- Compare final modularity and community structure across 8+ seeds each

**What to look for:**
- Do the two initializations produce different stable topologies at the same threshold?
- Is the k-NN initialized graph more coherent (higher modularity)?
- Does the gap between the two conditions match the fixed-node-state prediction?

---

## Phase 5 — Graph Traversal and Chain Analysis

**Goal:** Demonstrate semantic chain structure in the self-organized graph.

**After the graph has converged:**
```python
import networkx as nx

def semantic_chain(G, source, target):
    path = nx.shortest_path(G, source, target)
    return path

# Example chains to test
chains = [
    ("black_hole", "pasta"),       # should cross multiple domains
    ("harmony", "frequency"),      # music → physics bridge
    ("genome", "entropy"),         # biology → information theory
]
```

**What to look for:**
- Do paths between distant concepts pass through meaningful intermediate nodes?
- Are bridge words (entropy, frequency) on cross-domain paths?
- Does path length correlate with semantic distance in embedding space?

---

## Phase 6 — Sweep and Phase Diagram

**Goal:** Produce equivalent of the CA phase diagram but for knowledge graph coherence.

**Run:**
- Threshold sweep from 0.0 to 1.0 in steps of 0.05
- 8 seeds per threshold
- Measure: mean modularity, variance, mean degree, NMI vs ground truth

**Expected result:**
- Low threshold: disordered (all nodes connected indiscriminately, low modularity)
- High threshold: fragmented (no edges form, isolated nodes)
- Sweet spot: coherent community structure, peak modularity
- Phase transition: sharp jump in modularity, variance spike, bimodal seed distribution

Plot this as the knowledge graph analog of the CA boundary density vs mutation rate curve.

---

## Measurements Summary

| Measurement | Phase | Maps to CA observable |
|---|---|---|
| Modularity | 1-6 | Boundary density (inverse) |
| NMI vs ground truth | 1-3 | Order parameter |
| Seed variance | 1,4 | Variance spike at transition |
| Bridge node stability | 3 | Coexistence region behavior |
| Hysteresis | 4 | Ordered vs random init divergence |
| Semantic chain length | 5 | Domain size / path structure |
| Threshold sweep | 6 | Mutation rate phase diagram |

---

## Implementation Notes

**Start here — minimal working version:**
```bash
pip install gensim networkx scikit-learn numpy
```

**Adapt existing CA infrastructure:**
- Replace RGB state with embedding vector
- Replace dominance rule with cosine similarity threshold
- Keep maybe_form / maybe_sever logic unchanged
- Keep asynchronous random sequential update unchanged
- Add modularity and NMI measurement to analysis pipeline

**Do not:**
- Use the 2D toroidal grid — nodes live on a general graph, not a lattice
- Update node states — embeddings are fixed
- Run more than 200 nodes initially — keep it interpretable

---

## What Success Looks Like

A figure showing modularity vs threshold with a sharp transition, seed variance spike at
the transition point, and bimodal final-state distribution — directly analogous to your
existing CA results but in semantic space. Ambiguous words sitting at community boundaries.
Semantic chains passing through meaningful intermediates.

That's the proof of concept. It doesn't need to beat existing clustering algorithms.
It needs to exhibit the phase transition in semantic space and demonstrate that the
topology reveals structure (bridge nodes, chain geometry) that clustering alone discards.
