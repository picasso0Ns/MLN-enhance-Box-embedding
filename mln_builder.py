"""
mln_builder.py
==============
MLN structure construction and Horn rule mining.

For each operator context (projection / intersection / negation) the builder:
  1. Mines Horn rules from the KG (1-hop and 2-hop patterns).
  2. Constructs the MLN subgraph G = (V, E) for the given anchor / relation.
  3. Identifies observed nodes V_o and unobserved nodes V_u.
  4. Exposes Markov-blanket queries for the GCN feature construction.

A C-extension (cpp_ext/rule_miner) is used when available for the
computationally heavy 2-hop rule enumeration; the pure-Python fallback is
used otherwise.
"""

import time
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from tqdm import tqdm

# Try to import the optional C++ accelerator
try:
    from cpp_ext import rule_miner as _cpp_miner
    _USE_CPP = True
except ImportError:
    _USE_CPP = False


# ---------------------------------------------------------------------------
# Rule mining
# ---------------------------------------------------------------------------

class RuleMiner:
    """
    Extract Horn rules of the form:  body_rels → head_rel
    from the training facts.

    Supports up to 2-hop rules: (r1) → r0  or  (r1, r2) → r0.
    """

    def __init__(self,
                 min_support:    int   = 3,
                 min_confidence: float = 0.2,
                 max_rules:      int   = 200,
                 max_rule_len:   int   = 2,
                 max_time:       float = 600.0):
        self.min_support    = min_support
        self.min_confidence = min_confidence
        self.max_rules      = max_rules
        self.max_rule_len   = max_rule_len
        self.max_time       = max_time

    def mine(self,
             triples: np.ndarray,
             num_relations: int) -> Tuple[List[tuple], Dict[tuple, float]]:
        """
        Mine Horn rules from an array of (h, r, t) triples.

        Returns
        -------
        rules        : list of  (head_rel, (body_rels, …))
        rule_weights : dict  rule -> confidence  in [0, 1]
        """
        if _USE_CPP:
            return self._mine_cpp(triples, num_relations)
        return self._mine_python(triples, num_relations)

    # ------------------------------------------------------------------
    # Pure-Python backend
    # ------------------------------------------------------------------

    def _mine_python(self,
                     triples:       np.ndarray,
                     num_relations: int) -> Tuple[List[tuple], Dict[tuple, float]]:
        """Python implementation of rule mining."""
        rel_to_pairs: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        for h, r, t in triples:
            rel_to_pairs[int(r)].add((int(h), int(t)))

        rules: List[tuple] = []
        rule_weights: Dict[tuple, float] = {}

        start = time.time()

        # ---- 1-hop rules: (r_body) → r_head ----
        rel_items = sorted(rel_to_pairs.items(),
                           key=lambda x: len(x[1]), reverse=True)
        top_rels  = [r for r, _ in rel_items]

        print("Mining 1-hop rules …")
        for head_rel, head_pairs in tqdm(rel_items, desc="1-hop head"):
            if time.time() - start > self.max_time:
                break
            if len(rules) >= self.max_rules:
                break
            if len(head_pairs) < self.min_support:
                continue

            for body_rel in top_rels:
                if body_rel == head_rel:
                    continue
                body_pairs = rel_to_pairs[body_rel]
                if len(body_pairs) < self.min_support:
                    continue

                overlap    = len(head_pairs & body_pairs)
                confidence = overlap / len(body_pairs)
                if confidence >= self.min_confidence and overlap >= self.min_support:
                    rule = (head_rel, (body_rel,))
                    rules.append(rule)
                    rule_weights[rule] = confidence
                    if len(rules) >= self.max_rules:
                        break

        # ---- 2-hop rules: (r_b1, r_b2) → r_head ----
        if self.max_rule_len >= 2 and len(rules) < self.max_rules:
            # For 2-hop we look for chain patterns: h -r_b1-> m -r_b2-> t
            # Build tail-to-head mapping for chaining
            tail_to_heads: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for h, r, t in triples:
                tail_to_heads[(int(r), int(t))].append(int(h))

            head_to_tails: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for h, r, t in triples:
                head_to_tails[(int(r), int(h))].append(int(t))

            print("Mining 2-hop rules …")
            processed = 0
            for head_rel, head_pairs in tqdm(rel_items[:30], desc="2-hop head"):
                if time.time() - start > self.max_time:
                    break
                if len(rules) >= self.max_rules:
                    break

                # Enumerate chain r_b1 -> r_b2 such that for each (h, t) in
                # head_pairs there exists an m with r_b1(h, m) and r_b2(m, t)
                chain_counts: Dict[Tuple[int,int], int] = defaultdict(int)

                for h, t in list(head_pairs)[:500]:
                    processed += 1
                    # Find all mid-points reachable from h
                    for r1 in top_rels[:20]:
                        mids = head_to_tails.get((r1, h), [])
                        for m in mids[:10]:
                            # Find all r2 that connect m to t
                            for r2 in top_rels[:20]:
                                if t in head_to_tails.get((r2, m), []):
                                    chain_counts[(r1, r2)] += 1

                for (r1, r2), count in chain_counts.items():
                    if r1 == head_rel or r2 == head_rel:
                        continue
                    body_count = min(
                        len(rel_to_pairs.get(r1, set())),
                        len(rel_to_pairs.get(r2, set()))
                    )
                    if body_count == 0:
                        continue
                    confidence = count / body_count
                    if confidence >= self.min_confidence and count >= self.min_support:
                        rule = (head_rel, (r1, r2))
                        if rule not in rule_weights:
                            rules.append(rule)
                            rule_weights[rule] = confidence
                    if len(rules) >= self.max_rules:
                        break
                if len(rules) >= self.max_rules:
                    break

        # Sort by confidence and trim
        rules.sort(key=lambda r: rule_weights.get(r, 0.0), reverse=True)
        rules = rules[:self.max_rules]
        rule_weights = {r: rule_weights[r] for r in rules if r in rule_weights}

        print(f"Mined {len(rules)} rules "
              f"({sum(1 for r in rules if len(r[1]) == 1)} 1-hop, "
              f"{sum(1 for r in rules if len(r[1]) == 2)} 2-hop)")
        return rules, rule_weights

    # ------------------------------------------------------------------
    # C++ accelerated backend (wraps cpp_ext.rule_miner)
    # ------------------------------------------------------------------

    def _mine_cpp(self,
                  triples:       np.ndarray,
                  num_relations: int) -> Tuple[List[tuple], Dict[tuple, float]]:
        """Delegate to the compiled C++ extension."""
        raw_rules, raw_weights = _cpp_miner.mine_rules(
            triples.astype(np.int32),
            num_relations,
            self.min_support,
            self.min_confidence,
            self.max_rules,
            self.max_rule_len,
        )
        rules: List[tuple] = []
        rule_weights: Dict[tuple, float] = {}
        for rr, w in zip(raw_rules, raw_weights):
            rule = (int(rr[0]), tuple(int(x) for x in rr[1:]))
            rules.append(rule)
            rule_weights[rule] = float(w)
        return rules, rule_weights


# ---------------------------------------------------------------------------
# MLN subgraph builder
# ---------------------------------------------------------------------------

class MLNBuilder:
    """
    Given a dataset, a set of Horn rules, and an operator context (anchor
    entity + relation), construct the MLN subgraph:

      G = (V, E)
      V_o ⊆ V  (observed fact-nodes)
      V_u = V minus V_o  (unobserved fact-nodes)

    The MLN graph is used by VEMInference to compute formula weights and
    membership probabilities.
    """

    def __init__(self,
                 rules:              List[tuple],
                 rule_weights:       Dict[tuple, float],
                 all_triples:        np.ndarray,
                 train_triples:      np.ndarray,
                 max_nodes:          int = 2000,
                 max_nodes_per_rel:  int = 200):
        self.rules             = rules
        self.rule_weights      = rule_weights
        self.max_nodes         = max_nodes
        self.max_nodes_per_rel = max_nodes_per_rel

        # Build lookup structures
        self._rel_to_pairs: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
        for h, r, t in all_triples:
            self._rel_to_pairs[int(r)].append((int(h), int(t)))

        self._entity_neighbors: Dict[int, Set[Tuple[int,int]]] = defaultdict(set)
        for h, r, t in train_triples:
            self._entity_neighbors[int(h)].add((int(r), int(t)))

        self._train_set: Set[Tuple[int,int,int]] = {
            (int(h), int(r), int(t)) for h, r, t in train_triples
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_for_projection(self,
                              anchor:   int,
                              relation: int
                             ) -> Tuple[List[tuple], Set[tuple], Set[tuple], Set[tuple]]:
        """
        Build MLN for the Initial Projection operator.

        Formula set F = {r(anchor, x), r_new(x)}
        V_o = { r(anchor, e) | e ∈ O_r(anchor) }

        Returns
        -------
        node_list, edge_set, observed_facts, unobserved_facts
        """
        # Directly reachable entities via `relation`
        tail_entities = [t for _, t in self._rel_to_pairs.get(relation, [])
                         if _ == anchor or True]  # all tails for simplicity
        tail_entities = [t for h, t in self._rel_to_pairs.get(relation, [])
                         if h == anchor]

        observed: Set[tuple] = set()
        for t in tail_entities:
            observed.add((anchor, relation, t))

        # Add body relation groundings
        nodes: Set[tuple] = set(observed)
        for rule in self.rules:
            head_rel, body_rels = rule
            if head_rel != relation:
                continue
            for body_rel in body_rels:
                for h, t in self._rel_to_pairs.get(body_rel, [])[:self.max_nodes_per_rel]:
                    nodes.add((h, body_rel, t))

        return self._finalise(nodes, observed)

    def build_for_intersection(self,
                                input_relations: List[int],
                                anchor_entities: List[int]
                               ) -> Tuple[List[tuple], Set[tuple], Set[tuple], Set[tuple]]:
        """
        Build MLN for the Intersection operator.

        Formula set:  r_i(x, a_i) → r(x, a_x)

        Returns
        -------
        node_list, edge_set, observed_facts, unobserved_facts
        """
        observed: Set[tuple] = set()
        nodes:    Set[tuple] = set()

        for rel, anchor in zip(input_relations, anchor_entities):
            for h, t in self._rel_to_pairs.get(rel, [])[:self.max_nodes_per_rel]:
                if h == anchor or t == anchor:
                    fact = (h, rel, t)
                    nodes.add(fact)
                    observed.add(fact)

        # Add rule-related nodes
        for rule in self.rules:
            head_rel, body_rels = rule
            if head_rel not in input_relations:
                continue
            for body_rel in body_rels:
                for h, t in self._rel_to_pairs.get(body_rel, [])[:self.max_nodes_per_rel // 2]:
                    nodes.add((h, body_rel, t))

        return self._finalise(nodes, observed)

    def build_for_negation(self,
                            positive_relation: int,
                            negative_relations: List[int],
                            anchor_entities:    List[int]
                           ) -> Tuple[List[tuple], Set[tuple], Set[tuple], Set[tuple]]:
        """
        Build MLN for the Negation operator.

        Positive formula:  r_1(x, a_1) → r(x, a_x)
        Negated formulas:  ¬r_i(x, a_i) → r(x, a_x)  for i ≥ 2

        Negated facts are those in the negative relations that are NOT in
        V_o but are in the KG.

        Returns
        -------
        node_list, edge_set, observed_facts, unobserved_facts
        """
        observed: Set[tuple] = set()
        nodes:    Set[tuple] = set()

        # Positive box observed facts
        for h, t in self._rel_to_pairs.get(positive_relation, [])[:self.max_nodes_per_rel]:
            fact = (h, positive_relation, t)
            nodes.add(fact)
            observed.add(fact)

        # Negative boxes: these are in the graph but should NOT be in the answer
        for neg_rel in negative_relations:
            for h, t in self._rel_to_pairs.get(neg_rel, [])[:self.max_nodes_per_rel // 2]:
                nodes.add((h, neg_rel, t))
                # These are "negated" – we mark them as observed (true) so
                # their presence in V_o tells the MLN they need to be excluded.
                observed.add((h, neg_rel, t))

        return self._finalise(nodes, observed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalise(self,
                  nodes:    Set[tuple],
                  observed: Set[tuple]
                 ) -> Tuple[List[tuple], Set[tuple], Set[tuple], Set[tuple]]:
        """
        Trim nodes to max_nodes, build edges, split V_o / V_u.
        """
        if len(nodes) > self.max_nodes:
            observed_list  = list(observed)[:self.max_nodes // 2]
            remaining      = list(nodes - observed)
            remaining      = random.sample(remaining,
                                           min(len(remaining),
                                               self.max_nodes - len(observed_list)))
            nodes = set(observed_list) | set(remaining)
            observed = observed & nodes

        node_list = sorted(nodes)   # deterministic ordering

        # Build edges: connect nodes that share at least one entity
        entity_to_nodes: Dict[int, List[tuple]] = defaultdict(list)
        for node in node_list:
            h, r, t = node
            entity_to_nodes[h].append(node)
            entity_to_nodes[t].append(node)

        edge_set: Set[tuple] = set()
        for entity, nbrs in entity_to_nodes.items():
            if len(nbrs) > 20:
                nbrs = random.sample(nbrs, 20)
            for i, n1 in enumerate(nbrs):
                for n2 in nbrs[i+1:]:
                    edge_set.add((n1, n2))

        # Rule-based edges
        node_index = {n: i for i, n in enumerate(node_list)}
        rel_to_nodes: Dict[int, List[tuple]] = defaultdict(list)
        for node in node_list:
            rel_to_nodes[node[1]].append(node)

        for head_rel, body_rels in self.rules:
            head_nodes = rel_to_nodes.get(head_rel, [])
            for body_rel in body_rels:
                body_nodes = rel_to_nodes.get(body_rel, [])
                for n1 in head_nodes[:50]:
                    for n2 in body_nodes[:50]:
                        edge_set.add((n1, n2))

        unobserved = set(node_list) - observed

        return node_list, edge_set, observed, unobserved

    def get_entity_pairs_for_relation(self,
                                       relation: int,
                                       anchor:   Optional[int] = None
                                      ) -> List[Tuple[int,int]]:
        """
        Return entity pairs (h, t) for the given relation,
        optionally filtered to those involving `anchor`.
        """
        pairs = self._rel_to_pairs.get(relation, [])
        if anchor is not None:
            pairs = [(h, t) for h, t in pairs if h == anchor or t == anchor]
        return pairs
