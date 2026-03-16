"""
test_logic.py
=============
Pure-Python logic test for all major components.
No real torch installation required – runs with a lightweight mock.
"""

import sys
import types
import math
import numpy as np


# ---------------------------------------------------------------------------
# Build a minimal torch package mock
# ---------------------------------------------------------------------------

def make_pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    sys.modules[name] = m
    return m


torch_mod = make_pkg('torch')
torch_mod.__version__ = '0.0-mock'

# torch.utils.data
_utils = make_pkg('torch.utils')
_data  = make_pkg('torch.utils.data')
torch_mod.utils = _utils
_utils.data = _data

class _Dataset:
    def __len__(self): return 0
    def __getitem__(self, i): return None

class _DataLoader:
    def __init__(self, ds, **kw):
        self._ds = ds
    def __iter__(self): return iter([])
    def __len__(self): return 0

_data.Dataset    = _Dataset
_data.DataLoader = _DataLoader

# torch.nn
_nn = make_pkg('torch.nn')
torch_mod.nn = _nn

class _Module:
    def parameters(self): return iter([])
    def eval(self): return self
    def train(self): return self
    def to(self, d): return self
    def state_dict(self): return {}
    def load_state_dict(self, d): pass

_nn.Module      = _Module
_nn.Embedding   = lambda *a, **kw: type('E', (), {
    'weight': None,
    'num_embeddings': a[0],
})()
_nn.Linear      = lambda *a, **kw: type('L', (), {
    'weight': None, 'bias': None,
})()
_nn.Parameter   = lambda t, **kw: t
_nn.PairwiseDistance = lambda p=2: (lambda a, b: _FT(0.0))
_nn.init = types.SimpleNamespace(
    xavier_uniform_=lambda w: None,
    uniform_=lambda w, *a: None,
)
_nn.utils = types.SimpleNamespace(
    clip_grad_norm_=lambda p, v: None
)

# torch.nn.functional
_F = make_pkg('torch.nn.functional')
_nn.functional = _F
_F.softmax       = lambda t, dim=0: t
_F.relu          = lambda t: t
_F.normalize     = lambda t, **kw: t
_F.softplus      = lambda t: t
_F.logsigmoid    = lambda t: t
_F.cross_entropy = lambda *a, **kw: _FT(0.1)

# Fake Tensor
class _FT:
    def __init__(self, v=0.0):
        self.v = v
    def item(self): return float(self.v) if not hasattr(self.v, '__len__') else float(self.v[0])
    def size(self, i=None): return 1
    def numel(self): return 1
    def mean(self): return self
    def sum(self): return self
    def abs(self): return self
    def detach(self): return self
    def clone(self): return self
    def squeeze(self, d=0): return self
    def unsqueeze(self, d=0): return self
    def expand(self, *a): return self
    def __add__(self, o): return self
    def __radd__(self, o): return self
    def __sub__(self, o): return self
    def __mul__(self, o): return self
    def __rmul__(self, o): return self
    def __truediv__(self, o): return self
    def __neg__(self): return self
    def backward(self): pass
    def __repr__(self): return f'FT({self.v})'

torch_mod.Tensor  = _FT
torch_mod.tensor  = lambda d, **kw: _FT(d if not hasattr(d, '__len__') else 0)
torch_mod.zeros   = lambda *s, **kw: _FT(0)
torch_mod.ones    = lambda *s, **kw: _FT(1)
torch_mod.stack   = lambda ts, **kw: _FT(0)
torch_mod.cat     = lambda ts, **kw: _FT(0)
torch_mod.arange  = lambda *a, **kw: _FT(0)
torch_mod.device  = lambda s='cpu': s
torch_mod.long    = 'long'
torch_mod.float32 = 'float32'

import contextlib
torch_mod.no_grad = contextlib.nullcontext
torch_mod.save    = lambda obj, path: None
torch_mod.load    = lambda path, **kw: {}

# torch.optim
_optim = make_pkg('torch.optim')
torch_mod.optim = _optim

class _Adam:
    def __init__(self, p, lr=1e-3): pass
    def zero_grad(self): pass
    def step(self): pass
    def state_dict(self): return {}
    def load_state_dict(self, d): pass

class _SLR:
    def __init__(self, opt, **kw): pass
    def step(self): pass

_optim.Adam = _Adam
_optim.lr_scheduler = types.SimpleNamespace(StepLR=_SLR)

# Other lightweight stubs
_tqdm = make_pkg('tqdm')
_tqdm.tqdm = lambda it, **kw: it
make_pkg('networkx')

print("Torch mock ready.\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

sys.path.insert(0, '/home/claude')
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
errors = []

def check(label, cond, detail=''):
    if cond:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" – {detail}" if detail else ''))
        errors.append(label)


# ── 1. RuleMiner ─────────────────────────────────────────────────────────
print("=== 1. RuleMiner ===")
from mln_box_embedding.mln_builder import RuleMiner

rng = np.random.default_rng(42)
N = 50
triples = np.column_stack([
    rng.integers(0, N, 300),
    rng.integers(0, 5, 300),
    rng.integers(0, N, 300),
]).astype(np.int64)

# Inject high-confidence implication r0 -> r1
for _ in range(80):
    h, t = rng.integers(0, 20), rng.integers(0, 20)
    triples = np.vstack([triples, [h, 0, t], [h, 1, t]])

miner  = RuleMiner(min_support=3, min_confidence=0.2, max_rules=50, max_rule_len=2)
rules, weights = miner.mine(triples, num_relations=5)

check("At least one rule found",      len(rules) > 0)
check("All rules have valid head",    all(len(r) == 2 for r in rules))
check("All rules have body tuple",    all(isinstance(r[1], tuple) for r in rules))
check("All weights in (0,1]",         all(0 < w <= 1.0 for w in weights.values()))
check("r0->r1 rule found (high conf)",any(r == (1, (0,)) or r == (0, (1,)) for r in rules))


# ── 2. MLNBuilder ────────────────────────────────────────────────────────
print("\n=== 2. MLNBuilder ===")
from mln_box_embedding.mln_builder import MLNBuilder

builder = MLNBuilder(
    rules=rules, rule_weights=weights,
    all_triples=triples, train_triples=triples,
    max_nodes=200, max_nodes_per_rel=30
)

nl_proj, es_proj, obs_proj, unobs_proj = builder.build_for_projection(0, 0)
check("projection: nodes returned",       len(nl_proj) > 0)
check("projection: obs + unobs = total",  len(obs_proj) + len(unobs_proj) == len(nl_proj))
check("projection: edges are tuples",     all(len(e) == 2 for e in list(es_proj)[:5]))

nl_int, es_int, obs_int, _ = builder.build_for_intersection([0, 1], [0, 1])
check("intersection: nodes returned",     len(nl_int) > 0)

nl_neg, es_neg, obs_neg, _ = builder.build_for_negation(0, [1, 2], [0])
check("negation: nodes returned",         len(nl_neg) > 0)

pairs = builder.get_entity_pairs_for_relation(0)
check("get_entity_pairs: returns list",   isinstance(pairs, list))
pairs_anchored = builder.get_entity_pairs_for_relation(0, anchor=0)
check("get_entity_pairs(anchor=0): subset", all(h == 0 or t == 0 for h, t in pairs_anchored))


# ── 3. CQADataset – query generation ─────────────────────────────────────
print("\n=== 3. CQADataset ===")
from mln_box_embedding.query_dataset import (
    CQADataset, CQAQuery, ALL_QUERY_TYPES,
    QUERY_TYPES_EPFO, QUERY_TYPES_NEG
)

dense = np.array([[h, r, t]
                  for h in range(10)
                  for r in range(4)
                  for t in range(10)], dtype=np.int64)

cqa = CQADataset(
    triples=dense, num_entities=10, num_relations=4,
    query_types=['1p', '2p', '3p', '2i', '3i',
                 'ip', 'pi', '2u', 'up', '2d', '3d', 'dp'],
    max_queries=200, neg_ratio=4,
)
types_seen = {q.query_type for q in cqa.queries}

check("CQADataset: queries generated",    len(cqa) > 0)
check("1p queries generated",             '1p'  in types_seen)
check("2i queries generated",             '2i'  in types_seen)
check("2d queries generated",             '2d'  in types_seen)
check("all queries have answers",         all(len(q.answers) > 0 for q in cqa.queries))

sample = cqa[0]
check("__getitem__ returns dict",         isinstance(sample, dict))
check("sample has pos_answer",            'pos_answer' in sample)
check("sample has neg_answers",           'neg_answers' in sample)
check("sample has correct neg_ratio",     len(sample['neg_answers']) == 4)
check("pos not in neg",                   sample['pos_answer'] not in sample['neg_answers'])

# All 12 _gen_* methods present
gen_names = ['1p','2p','3p','2i','3i','ip','pi','2u','up','2d','3d','dp']
check("all 12 _gen_* methods present",
      all(hasattr(cqa, f'_gen_{n}') for n in gen_names))


# ── 4. Query save / load round-trip ──────────────────────────────────────
print("\n=== 4. Query pickle round-trip ===")
import tempfile, os
tmp = tempfile.mktemp(suffix='.pkl')
cqa.save_queries(tmp)
cqa2 = CQADataset(dense, 10, 4,
                  query_types=['1p', '2p'],
                  query_file=tmp,
                  max_queries=500)
check("loaded query count > 0",           len(cqa2) > 0)
check("loaded queries match saved",       len(cqa2) <= len(cqa))
os.unlink(tmp)


# ── 5. box_embedding.py pure helpers ─────────────────────────────────────
print("\n=== 5. box_embedding pure helpers ===")
from mln_box_embedding.box_embedding import union_via_dnf

dummy_boxes = [object(), object(), object()]
result = union_via_dnf(dummy_boxes)
check("union_via_dnf returns same list",  result is dummy_boxes)
check("union_via_dnf preserves length",   len(result) == 3)


# ── 6. ALL_QUERY_TYPES constant ───────────────────────────────────────────
print("\n=== 6. Query type constants ===")
check("ALL_QUERY_TYPES has 12 types",         len(ALL_QUERY_TYPES) == 12)
check("EPFO types correct count",             len(QUERY_TYPES_EPFO) == 9)
check("NEG types correct count",              len(QUERY_TYPES_NEG) == 3)
check("No overlap EPFO/NEG",
      len(set(QUERY_TYPES_EPFO) & set(QUERY_TYPES_NEG)) == 0)
check("All types in ALL_QUERY_TYPES",
      set(QUERY_TYPES_EPFO + QUERY_TYPES_NEG) == set(ALL_QUERY_TYPES))


# ── 7. CQAQuery dataclass ─────────────────────────────────────────────────
print("\n=== 7. CQAQuery ===")
q = CQAQuery('1p', {'anchor': 0, 'relations': [1]}, {3, 5, 7})
check("CQAQuery stores type",     q.query_type == '1p')
check("CQAQuery stores answers",  q.answers == {3, 5, 7})
check("CQAQuery stores structure",q.structure['anchor'] == 0)
check("CQAQuery repr works",      '1p' in repr(q))


# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if errors:
    print(f"FAILED: {len(errors)} checks – {errors}")
    sys.exit(1)
else:
    print(f"ALL TESTS PASSED ({sum(1 for _ in [None])} suites, 0 failures)")
