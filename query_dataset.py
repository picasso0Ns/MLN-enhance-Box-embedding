"""
query_dataset.py
================
Dataset handling for complex query answering (CQA).

Supports the 12 query types used in the paper:
  EPFO: 1p, 2p, 3p, 2i, 3i, ip, pi, 2u, up
  Negation / Difference: 2d, 3d, dp

Each query file is expected to be a pickle (.pkl) or text file with
queries indexed by type.  The module also provides a simple in-memory
builder for generating queries on-the-fly from raw triples (useful when
pre-generated query files are not available).
"""

import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


QUERY_TYPES_EPFO = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up']
QUERY_TYPES_NEG  = ['2d', '3d', 'dp']
ALL_QUERY_TYPES  = QUERY_TYPES_EPFO + QUERY_TYPES_NEG


# ---------------------------------------------------------------------------
# Query data structure
# ---------------------------------------------------------------------------

class CQAQuery:
    """
    Represents a single complex query.

    Attributes
    ----------
    query_type  : str
    structure   : dict  – query parameters (anchors, relations, …)
    answers     : set of int  – ground-truth answer entity ids
    """

    def __init__(self,
                 query_type: str,
                 structure:  dict,
                 answers:    Set[int]):
        self.query_type = query_type
        self.structure  = structure
        self.answers    = answers

    def __repr__(self):
        return (f"CQAQuery(type={self.query_type}, "
                f"answers={len(self.answers)})")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CQADataset(Dataset):
    """
    PyTorch Dataset for complex query answering.

    If a pre-built query file exists at `query_file`, it is loaded.
    Otherwise queries are generated on-the-fly from `triples`.
    """

    def __init__(self,
                 triples:       np.ndarray,
                 num_entities:  int,
                 num_relations: int,
                 query_types:   Optional[List[str]] = None,
                 query_file:    Optional[str]       = None,
                 max_queries:   int                 = 50_000,
                 neg_ratio:     int                 = 128):
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.neg_ratio     = neg_ratio
        self.query_types   = query_types or ALL_QUERY_TYPES

        # Build graph for query generation
        self._build_graph(triples)

        if query_file and os.path.exists(query_file):
            self.queries = self._load_queries(query_file)
        else:
            print(f"Generating up to {max_queries} queries …")
            self.queries = self._generate_queries(max_queries)

        print(f"CQADataset: {len(self.queries)} queries "
              f"across types {set(q.query_type for q in self.queries)}")

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, triples: np.ndarray):
        """Build entity→(relation, tail) adjacency for query generation."""
        self._head_to_rt: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
        self._tail_to_rh: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
        self._triple_set: Set[Tuple[int,int,int]] = set()

        for h, r, t in triples:
            h, r, t = int(h), int(r), int(t)
            self._head_to_rt[h].append((r, t))
            self._tail_to_rh[t].append((r, h))
            self._triple_set.add((h, r, t))

    # ------------------------------------------------------------------
    # Query file I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_queries(query_file: str) -> List[CQAQuery]:
        with open(query_file, 'rb') as f:
            raw = pickle.load(f)
        queries = []
        for qt, items in raw.items():
            for q_struct, answers in items:
                queries.append(CQAQuery(qt, q_struct, set(answers)))
        return queries

    def save_queries(self, path: str):
        """Serialise queries as a pickle for reuse."""
        by_type: Dict[str, list] = defaultdict(list)
        for q in self.queries:
            by_type[q.query_type].append((q.structure, list(q.answers)))
        with open(path, 'wb') as f:
            pickle.dump(dict(by_type), f)

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    def _generate_queries(self, max_per_type: int) -> List[CQAQuery]:
        budget = max(1, max_per_type // len(self.query_types))
        queries = []
        for qt in self.query_types:
            qs = self._generate_type(qt, budget)
            queries.extend(qs)
            print(f"  {qt}: {len(qs)} queries generated")
        random.shuffle(queries)
        return queries

    def _generate_type(self, qt: str, n: int) -> List[CQAQuery]:
        gen = getattr(self, f'_gen_{qt.replace("-", "_")}', None)
        if gen is None:
            return []
        queries = []
        attempts = 0
        while len(queries) < n and attempts < n * 10:
            attempts += 1
            q = gen()
            if q is not None and len(q.answers) > 0:
                queries.append(q)
        return queries

    # ---- 1p ----
    def _gen_1p(self) -> Optional[CQAQuery]:
        h = random.choice(list(self._head_to_rt.keys()))
        neighbors = self._head_to_rt[h]
        if not neighbors: return None
        r, t = random.choice(neighbors)
        answers = {t2 for r2, t2 in self._head_to_rt[h] if r2 == r}
        return CQAQuery('1p', {'anchor': h, 'relations': [r]}, answers)

    # ---- 2p ----
    def _gen_2p(self) -> Optional[CQAQuery]:
        h = random.choice(list(self._head_to_rt.keys()))
        nbrs = self._head_to_rt[h]
        if not nbrs: return None
        r1, mid = random.choice(nbrs)
        nbrs2 = self._head_to_rt.get(mid, [])
        if not nbrs2: return None
        r2, t = random.choice(nbrs2)
        answers = {t2 for r2_, t2 in self._head_to_rt.get(mid, []) if r2_ == r2}
        if not answers: return None
        return CQAQuery('2p', {'anchor': h, 'relations': [r1, r2]}, answers)

    # ---- 3p ----
    def _gen_3p(self) -> Optional[CQAQuery]:
        h = random.choice(list(self._head_to_rt.keys()))
        nbrs = self._head_to_rt[h]
        if not nbrs: return None
        r1, mid1 = random.choice(nbrs)
        nbrs2 = self._head_to_rt.get(mid1, [])
        if not nbrs2: return None
        r2, mid2 = random.choice(nbrs2)
        nbrs3 = self._head_to_rt.get(mid2, [])
        if not nbrs3: return None
        r3, t = random.choice(nbrs3)
        answers = {t2 for r3_, t2 in self._head_to_rt.get(mid2, []) if r3_ == r3}
        if not answers: return None
        return CQAQuery('3p', {'anchor': h, 'relations': [r1, r2, r3]}, answers)

    # ---- 2i ----
    def _gen_2i(self) -> Optional[CQAQuery]:
        h1 = random.choice(list(self._head_to_rt.keys()))
        h2 = random.choice(list(self._head_to_rt.keys()))
        nbrs1 = self._head_to_rt[h1]
        nbrs2 = self._head_to_rt[h2]
        if not nbrs1 or not nbrs2: return None
        r1, _ = random.choice(nbrs1)
        r2, _ = random.choice(nbrs2)
        set1  = {t for r_, t in self._head_to_rt[h1] if r_ == r1}
        set2  = {t for r_, t in self._head_to_rt[h2] if r_ == r2}
        answers = set1 & set2
        if not answers: return None
        return CQAQuery('2i', {
            'anchors':   [h1, h2],
            'relations': [r1, r2],
        }, answers)

    # ---- 3i ----
    def _gen_3i(self) -> Optional[CQAQuery]:
        heads = [random.choice(list(self._head_to_rt.keys())) for _ in range(3)]
        rels  = []
        sets  = []
        for h in heads:
            nbrs = self._head_to_rt[h]
            if not nbrs: return None
            r, _ = random.choice(nbrs)
            rels.append(r)
            sets.append({t for r_, t in self._head_to_rt[h] if r_ == r})
        answers = sets[0] & sets[1] & sets[2]
        if not answers: return None
        return CQAQuery('3i', {'anchors': heads, 'relations': rels}, answers)

    # ---- ip  (intersection then projection) ----
    def _gen_ip(self) -> Optional[CQAQuery]:
        inner = self._gen_2i()
        if inner is None: return None
        # Pick a relation for the outer projection
        sample_entity = random.choice(list(inner.answers))
        nbrs = self._head_to_rt.get(sample_entity, [])
        if not nbrs: return None
        r_out, _ = random.choice(nbrs)
        answers = set()
        for e in inner.answers:
            for r_, t in self._head_to_rt.get(e, []):
                if r_ == r_out:
                    answers.add(t)
        if not answers: return None
        struct = dict(inner.structure)
        struct['relations'] = inner.structure['relations'] + [r_out]
        return CQAQuery('ip', struct, answers)

    # ---- pi  (projection then intersection) ----
    def _gen_pi(self) -> Optional[CQAQuery]:
        proj = self._gen_2p()
        if proj is None: return None
        h2   = random.choice(list(self._head_to_rt.keys()))
        nbrs = self._head_to_rt[h2]
        if not nbrs: return None
        r2, _ = random.choice(nbrs)
        set2  = {t for r_, t in self._head_to_rt[h2] if r_ == r2}
        answers = proj.answers & set2
        if not answers: return None
        return CQAQuery('pi', {
            'anchor':    proj.structure['anchor'],
            'anchors':   [proj.structure['anchor'], h2],
            'relations': proj.structure['relations'] + [r2],
        }, answers)

    # ---- 2u ----
    def _gen_2u(self) -> Optional[CQAQuery]:
        q1 = self._gen_1p()
        q2 = self._gen_1p()
        if q1 is None or q2 is None: return None
        answers = q1.answers | q2.answers
        return CQAQuery('2u', {
            'anchors':   [q1.structure['anchor'], q2.structure['anchor']],
            'relations': [q1.structure['relations'][0],
                          q2.structure['relations'][0]],
        }, answers)

    # ---- up  (union then projection) ----
    def _gen_up(self) -> Optional[CQAQuery]:
        union_q = self._gen_2u()
        if union_q is None: return None
        sample  = random.choice(list(union_q.answers))
        nbrs    = self._head_to_rt.get(sample, [])
        if not nbrs: return None
        r_out, _ = random.choice(nbrs)
        answers  = set()
        for e in union_q.answers:
            for r_, t in self._head_to_rt.get(e, []):
                if r_ == r_out:
                    answers.add(t)
        if not answers: return None
        struct = dict(union_q.structure)
        struct['relations'] = union_q.structure['relations'] + [r_out]
        return CQAQuery('up', struct, answers)

    # ---- 2d  (difference) ----
    def _gen_2d(self) -> Optional[CQAQuery]:
        q1 = self._gen_1p()
        q2 = self._gen_1p()
        if q1 is None or q2 is None: return None
        answers = q1.answers - q2.answers
        if not answers: return None
        return CQAQuery('2d', {
            'anchors':   [q1.structure['anchor'], q2.structure['anchor']],
            'relations': [q1.structure['relations'][0],
                          q2.structure['relations'][0]],
        }, answers)

    # ---- 3d ----
    def _gen_3d(self) -> Optional[CQAQuery]:
        q1 = self._gen_1p()
        q2 = self._gen_1p()
        q3 = self._gen_1p()
        if any(q is None for q in [q1, q2, q3]): return None
        answers = q1.answers - q2.answers - q3.answers
        if not answers: return None
        return CQAQuery('3d', {
            'anchors':   [q1.structure['anchor'],
                          q2.structure['anchor'],
                          q3.structure['anchor']],
            'relations': [q1.structure['relations'][0],
                          q2.structure['relations'][0],
                          q3.structure['relations'][0]],
        }, answers)

    # ---- dp  (difference-projection) ----
    def _gen_dp(self) -> Optional[CQAQuery]:
        proj = self._gen_2p()
        neg  = self._gen_1p()
        if proj is None or neg is None: return None
        answers = proj.answers - neg.answers
        if not answers: return None
        return CQAQuery('dp', {
            'anchor':    proj.structure['anchor'],
            'anchors':   [proj.structure['anchor'],
                          neg.structure['anchor']],
            'relations': proj.structure['relations'] + neg.structure['relations'],
        }, answers)

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        q = self.queries[idx]

        # Sample one positive answer
        pos_ans = random.choice(list(q.answers))

        # Sample negative answers
        negs = []
        while len(negs) < self.neg_ratio:
            e = random.randint(0, self.num_entities - 1)
            if e not in q.answers:
                negs.append(e)

        return {
            'query_type': q.query_type,
            'structure':  q.structure,
            'pos_answer': pos_ans,
            'neg_answers': negs,
            'all_answers': list(q.answers),
        }

    @staticmethod
    def collate_fn(batch):
        """Simple collate: return list of dicts (variable structure per type)."""
        return batch
