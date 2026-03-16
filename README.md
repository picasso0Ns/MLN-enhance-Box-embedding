# MLN-Enhanced Box Embedding for Complex Query Answering


## Overview


| Operator | Description | Paper Section |
|---|---|---|
| **Initial Projection** | Anchor entity + relation → new box | §IV-B |
| **Path Projection** | Input box + relation → new box using MLN weights w₁, w₂ | §IV-B |
| **Intersection** | n input boxes → new box via MLN-weighted combination | §IV-C |
| **Negation / Difference** | b₁ \ {b₂, …, bₙ} using negated MLN formulas | §IV-C |
| **Union** | Handled via DNF transformation (no new box needed) | §IV-C |

The MLN formula weights are computed by the **VEM algorithm** (Algorithm 1 in the paper), which alternates between:
- **E-step**: GCN-based mean-field approximation Q(Vᵤ)
- **M-step**: GCN-based posterior update P(Vᵤ|Vₒ)

---

## Project Structure

```
mln_box_embedding/
│
├── box_embedding.py        # Box primitives, 4 operators, training loss
├── vem_inference.py        # VEM algorithm: E-step, M-step GCNs
├── mln_builder.py          # Horn rule mining + MLN subgraph construction
├── model.py                # Full MLNBoxEmbedding model
├── query_dataset.py        # 12 query types dataset + on-the-fly generation
├── evaluation_cqa.py       # MRR, Hits@1/3/10 evaluation (filtered)
├── train.py                # Two-stage training script
├── __init__.py
│
└── cpp_ext/                # Optional C++ acceleration (pybind11)
    ├── rule_miner.cpp      # Fast Horn rule mining (O(n log n) overlap)
    └── setup.py            # Build script
```

### File Responsibilities

| File | Implements |
|---|---|
| `box_embedding.py` | `Box` class · `entity_to_box_distance` (Eq. d_box) · `ProjectionOperator` (Eq. 3, 5) · `IntersectionOperator` (Eq. 8) · `NegationOperator` · `training_loss` (Eq. 9) |
| `vem_inference.py` | `VEMInference` · `GCNLayer` · E-step loss L_q (Eq. 7) · M-step loss L_p (Eq. 8) · KL-divergence check · pseudo-likelihood (Eq. 6) |
| `mln_builder.py` | `RuleMiner` · 1-hop & 2-hop Horn rule extraction · `MLNBuilder` for projection / intersection / negation subgraphs |
| `model.py` | `MLNBoxEmbedding` · full query execution for all 12 types · MLN weight extraction helpers · `score_entities` inference |
| `query_dataset.py` | `CQADataset` · on-the-fly query generation for 1p/2p/3p/2i/3i/ip/pi/2u/up/2d/3d/dp · query file I/O |
| `evaluation_cqa.py` | `evaluate_query` · `evaluate_model` · `evaluate_link_prediction` |
| `train.py` | Stage 1 (embedding pre-training) · Stage 2 (joint MLN+box training) |
| `cpp_ext/rule_miner.cpp` | C++14 + pybind11 rule miner · `std::set_intersection` overlap · 2-hop chain enumeration |

---

## Requirements

```
Python >= 3.8
torch >= 1.10
numpy
tqdm
networkx       # (optional, used internally)
```

Optional (for C++ acceleration):
```
pybind11
g++ / clang++ with C++14 support
```

Install all Python dependencies:
```bash
pip install torch numpy tqdm networkx pybind11
```

---

## Building the C++ Extension (Optional but Recommended)

The C++ extension (`cpp_ext/rule_miner`) accelerates the Horn rule mining step by **10–50×** compared to the pure Python fallback, especially on large KGs like FB15k-237.

```bash
cd cpp_ext
python setup.py build_ext --inplace
cd ..
```

If the build succeeds, `mln_builder.py` will automatically use it.  If not, it falls back silently to the pure-Python implementation — no code changes needed.

**macOS note** (Apple Clang + OpenMP):
```bash
brew install libomp
cd cpp_ext
CXXFLAGS="-Xpreprocessor -fopenmp" python setup.py build_ext --inplace
```

---

## Data Format

Each dataset should be a directory with:
```
<dataset_name>/
├── train.txt       # tab-separated  head\trelation\ttail
├── test.txt
├── relations.txt   # one relation per line
└── entities.txt    # (optional) one entity per line
```

Supported datasets (as in the paper):
- **FB15k**     – 14,951 entities · 1,345 relations
- **FB15k-237** – 14,505 entities · 237 relations
- **NELL995**   – 63,361 entities · 200 relations

---

## Quick Start

### Training

```bash
# Full two-stage training on FB15k-237
python train.py \
  --dataset FB15k237 \
  --dim 400 \
  --epochs 200 \
  --stage both \
  --batch_size 512 \
  --neg_ratio 128 \
  --lr 1e-3

# Pre-train embeddings only
python train.py --dataset FB15k237 --stage pretrain --epochs 100

# Fine-tune from a pre-trained checkpoint
python train.py --dataset FB15k237 --stage joint --epochs 100 \
  --ckpt checkpoints/FB15k237_pretrain_best.pt
```

### Inference / Evaluation

```python
import torch
from mln_box_embedding import MLNBoxEmbedding, RuleMiner, MLNBuilder
from mln_box_embedding import evaluate_model, CQADataset
from dataset_loader import KnowledgeGraphDataset

# Load dataset
dataset = KnowledgeGraphDataset('FB15k237')

# Mine rules
miner = RuleMiner(min_confidence=0.2, max_rules=200)
rules, weights = miner.mine(dataset.train_triples, dataset.num_relations)

# Build MLN
builder = MLNBuilder(rules, weights,
                     all_triples=dataset.train_triples,
                     train_triples=dataset.train_triples)

# Build model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLNBoxEmbedding(
    num_entities=dataset.num_entities,
    num_relations=dataset.num_relations,
    dim=400, device=device
)
model.set_mln_builder(builder)
model.load_state_dict(torch.load('checkpoints/FB15k237_final.pt'))

# Answer a single query
#   ?x : nationality(Obama, x)
q_box = model.answer_query('1p', {'anchor': 0, 'relations': [5]})

# Score all entities
scores = model.score_entities(q_box)
top10  = scores.topk(10).indices.tolist()
print("Top-10 answer entities:", top10)

# Full evaluation
cqa = CQADataset(dataset.train_triples,
                 dataset.num_entities,
                 dataset.num_relations,
                 max_queries=5000)
results = evaluate_model(model, [cqa[i] for i in range(len(cqa))])
```

---

## Supported Query Types

| Type | Formula | Example |
|---|---|---|
| **1p** | r(a, ?x) | Direct neighbours |
| **2p** | ∃y r₁(a,y) ∧ r₂(y,?x) | 2-hop path |
| **3p** | ∃y₁,y₂ r₁(a,y₁) ∧ r₂(y₁,y₂) ∧ r₃(y₂,?x) | 3-hop path |
| **2i** | r₁(a₁,?x) ∧ r₂(a₂,?x) | 2-way intersection |
| **3i** | r₁(a₁,?x) ∧ r₂(a₂,?x) ∧ r₃(a₃,?x) | 3-way intersection |
| **ip** | (r₁(a₁,?y) ∧ r₂(a₂,?y)) then r₃(?y,?x) | Intersect then project |
| **pi** | (∃y r₁(a,y) ∧ r₂(y,?x)) ∧ r₃(a₂,?x) | Project then intersect |
| **2u** | r₁(a₁,?x) ∨ r₂(a₂,?x) | Union (via DNF) |
| **up** | (r₁(a₁,?y) ∨ r₂(a₂,?y)) ∧ r₃(?y,?x) | Union then project |
| **2d** | r₁(a₁,?x) ∧ ¬r₂(a₂,?x) | Difference |
| **3d** | r₁(a₁,?x) ∧ ¬r₂(a₂,?x) ∧ ¬r₃(a₃,?x) | 3-way difference |
| **dp** | (∃y r₁(a,y) ∧ r₂(y,?x)) ∧ ¬r₃(a₃,?x) | Project then difference |

---

## Key Equations Implemented

### Entity-to-Box Distance (Sec. IV-D)
```
d_box(e; b) = d_outside(e; b) + α · d_inside(e; b)
d_outside   = ‖max(e − b_max, 0) + max(b_min − e, 0)‖₁
d_inside    = ‖Cen(b) − min(b_max, max(b_min, e))‖₁
```

### Initial Projection (Eq. 3)
```
b_new^c = r^c + Σᵢ wᵢ · eᵢ
b_new^o = r^o + Max(Σᵢ wᵢ · d(eᵢ, r^o))
```

### Path Projection (Eq. 5)
```
b_new^c = (b_h^c + (w₁/w₂) · r^c) / 2
b_new^o = Max(r^o, b_h^o)
```

### Intersection / Negation (Eq. 8)
```
b_new^c = α · Σᵢ (w_{rᵢ}/w_r) · rᵢ^c/n + β · Σᵢ pᵢ · eᵢ/m
b_new^o = Max over i of rᵢ^o
```

### Training Loss (Eq. 9)
```
L = −log σ(γ − d(e, b)) − (1/k) Σᵢ log σ(d(e'ᵢ, b) − γ)
```

---

## C++ Extension Details

The C++ extension (`cpp_ext/rule_miner.cpp`) implements:

1. **Sorted set intersection** for 1-hop overlap counting — O(n log n) vs O(n) Python set ops (faster due to cache efficiency).
2. **Chain enumeration** for 2-hop rules via an entity-indexed reverse mapping.
3. **OpenMP parallelisation** of the outer relation loop (Linux/Windows; optional on macOS).

The Python fallback in `mln_builder.py` (`RuleMiner._mine_python`) is functionally equivalent and used automatically when the extension is not compiled.

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `dim` | 400 | Embedding dimension |
| `gamma` | 12.0 | Training loss margin |
| `alpha` | 0.02 | Inside-distance scaling |
| `gcn_hidden` | 64 | VEM GCN hidden units |
| `num_formulas` | 20 | Max MLN formulas tracked |
| `max_vem_iter` | 5 | VEM iteration limit |
| `kl_threshold` | 0.01 | VEM KL convergence threshold |
| `min_confidence` | 0.2 | Rule mining threshold |
| `max_rules` | 200 | Max rules to mine |

---


## Acknowledgements

This implementation builds on ideas from:
- [Query2Box](https://arxiv.org/abs/2002.05969) – box embedding framework
- [BetaE](https://arxiv.org/abs/2010.11465) – negation in KG embeddings
- [GMNN](https://arxiv.org/abs/1905.06214) – mean-field GCN for MLN
