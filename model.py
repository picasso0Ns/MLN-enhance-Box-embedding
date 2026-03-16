"""
model.py
========
MLN-Enhanced Box Embedding Model for Complex Query Answering.

Ties together:
  - EntityEmbedding, RelationBoxEmbedding  (box_embedding.py)
  - ProjectionOperator, IntersectionOperator, NegationOperator  (box_embedding.py)
  - VEMInference  (vem_inference.py)
  - MLNBuilder    (mln_builder.py)
  - Training loss (box_embedding.py)

Query types supported (following paper's 12-type taxonomy):
  1p, 2p, 3p  – projection chains
  2i, 3i      – intersections
  ip, pi      – mixed projection + intersection
  2u, up      – union queries (handled via DNF)
  2d, 3d, dp  – difference (negation) queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .box_embedding import (
    Box,
    EntityEmbedding,
    RelationBoxEmbedding,
    ProjectionOperator,
    IntersectionOperator,
    NegationOperator,
    entity_to_box_distance,
    training_loss,
    union_via_dnf,
)
from .vem_inference import VEMInference
from .mln_builder import MLNBuilder, RuleMiner


class MLNBoxEmbedding(nn.Module):
    """
    MLN-Enhanced Box Embedding for Complex Query Answering.

    Parameters
    ----------
    num_entities   : int
    num_relations  : int
    dim            : int   embedding dimension
    gamma          : float margin for training loss
    alpha          : float entity-to-box inside distance scaling
    gcn_hidden     : int   hidden units for VEM GCNs
    max_vem_iter   : int   maximum VEM iterations per operator call
    kl_threshold   : float KL threshold for VEM convergence
    num_formulas   : int   maximum number of MLN formulas tracked
    device         : torch.device
    """

    def __init__(self,
                 num_entities:  int,
                 num_relations: int,
                 dim:           int   = 400,
                 gamma:         float = 12.0,
                 alpha:         float = 0.02,
                 gcn_hidden:    int   = 64,
                 max_vem_iter:  int   = 5,
                 kl_threshold:  float = 0.01,
                 num_formulas:  int   = 20,
                 device:        torch.device = None):
        super().__init__()

        if device is None:
            device = torch.device('cpu')
        self.device      = device
        self.dim         = dim
        self.gamma       = gamma
        self.alpha_dist  = alpha

        # ---- Embedding tables ----
        self.entity_embed   = EntityEmbedding(num_entities, dim)
        self.relation_embed = RelationBoxEmbedding(num_relations, dim)

        # ---- Operators ----
        self.proj_op  = ProjectionOperator(dim)
        self.inter_op = IntersectionOperator(dim)
        self.neg_op   = NegationOperator(dim)

        # ---- VEM inference ----
        self.vem = VEMInference(
            gcn_hidden_dim=gcn_hidden,
            max_iter=max_vem_iter,
            kl_threshold=kl_threshold,
            num_formulas=num_formulas,
            device=device,
        )

        # MLNBuilder is set after rule mining (call set_mln_builder)
        self._mln_builder: Optional[MLNBuilder] = None

        self.to(device)

    # ------------------------------------------------------------------
    # External setup
    # ------------------------------------------------------------------

    def set_mln_builder(self, builder: MLNBuilder):
        """Attach the MLN graph builder after rule mining is done."""
        self._mln_builder = builder

    # ------------------------------------------------------------------
    # Query answering forward pass
    # ------------------------------------------------------------------

    def answer_query(self,
                     query_type: str,
                     query:      dict) -> Box:
        """
        Execute a query and return the resulting Box embedding.

        Parameters
        ----------
        query_type : one of  '1p','2p','3p','2i','3i','ip','pi',
                             '2u','up','2d','3d','dp'
        query      : dict with keys depending on query type:
                       'anchor'    – int entity id (or list of ints for 2i etc.)
                       'relations' – list of int relation ids

        Returns
        -------
        Box  – the query answer box
        """
        qt = query_type.lower()

        if qt == '1p':
            return self._query_1p(query)
        elif qt == '2p':
            return self._query_2p(query)
        elif qt == '3p':
            return self._query_3p(query)
        elif qt == '2i':
            return self._query_2i(query)
        elif qt == '3i':
            return self._query_3i(query)
        elif qt == 'ip':
            return self._query_ip(query)
        elif qt == 'pi':
            return self._query_pi(query)
        elif qt == '2u':
            return self._query_2u(query)
        elif qt == 'up':
            return self._query_up(query)
        elif qt == '2d':
            return self._query_2d(query)
        elif qt == '3d':
            return self._query_3d(query)
        elif qt == 'dp':
            return self._query_dp(query)
        else:
            raise ValueError(f"Unknown query type: {qt}")

    # ------------------------------------------------------------------
    # 1p  –  ?x : r(a, x)
    # ------------------------------------------------------------------

    def _query_1p(self, query: dict) -> Box:
        anchor   = int(query['anchor'])
        rel_id   = int(query['relations'][0])

        rel_box  = self.relation_embed(
            torch.tensor([rel_id], device=self.device))

        # Get MLN weights for initial projection
        entity_embeds, entity_weights = self._get_projection_weights(
            anchor, rel_id)

        if entity_embeds is not None and entity_embeds.size(0) > 0:
            return self.proj_op.initial_projection(
                entity_embeds, entity_weights, rel_box)
        else:
            # Fallback: simple translation
            a_embed = self.entity_embed(
                torch.tensor([anchor], device=self.device))
            center  = a_embed + rel_box.center
            return Box(center, rel_box.offset)

    # ------------------------------------------------------------------
    # 2p  –  ?x : ∃y r1(a,y) ∧ r2(y,x)
    # ------------------------------------------------------------------

    def _query_2p(self, query: dict) -> Box:
        anchor = int(query['anchor'])
        r1, r2 = int(query['relations'][0]), int(query['relations'][1])

        # First hop
        b1 = self._query_1p({'anchor': anchor, 'relations': [r1]})

        # Second hop (path projection)
        rel2_box = self.relation_embed(
            torch.tensor([r2], device=self.device))
        w1, w2 = self._get_path_weights(anchor, r1, r2)
        return self.proj_op.path_projection(b1, rel2_box, w1, w2)

    # ------------------------------------------------------------------
    # 3p  –  chained 3-hop projection
    # ------------------------------------------------------------------

    def _query_3p(self, query: dict) -> Box:
        anchor = int(query['anchor'])
        r1, r2, r3 = [int(r) for r in query['relations'][:3]]

        b1 = self._query_1p({'anchor': anchor, 'relations': [r1]})

        rel2_box = self.relation_embed(
            torch.tensor([r2], device=self.device))
        w1, w2 = self._get_path_weights(anchor, r1, r2)
        b2 = self.proj_op.path_projection(b1, rel2_box, w1, w2)

        rel3_box = self.relation_embed(
            torch.tensor([r3], device=self.device))
        w3, w4 = self._get_path_weights(anchor, r2, r3)
        return self.proj_op.path_projection(b2, rel3_box, w3, w4)

    # ------------------------------------------------------------------
    # 2i  –  intersection of two projection results
    # ------------------------------------------------------------------

    def _query_2i(self, query: dict) -> Box:
        anchors   = query['anchors']    # list of 2 ints
        relations = query['relations']  # list of 2 ints

        b1 = self._query_1p({'anchor': anchors[0], 'relations': [relations[0]]})
        b2 = self._query_1p({'anchor': anchors[1], 'relations': [relations[1]]})

        return self._intersect_boxes(
            [b1, b2],
            [int(r) for r in relations],
            [int(a) for a in anchors],
        )

    # ------------------------------------------------------------------
    # 3i  –  intersection of three projection results
    # ------------------------------------------------------------------

    def _query_3i(self, query: dict) -> Box:
        anchors   = query['anchors']    # list of 3 ints
        relations = query['relations']  # list of 3 ints

        boxes = [self._query_1p({'anchor': int(a), 'relations': [int(r)]})
                 for a, r in zip(anchors, relations)]

        return self._intersect_boxes(boxes,
                                     [int(r) for r in relations],
                                     [int(a) for a in anchors])

    # ------------------------------------------------------------------
    # ip  –  intersection then projection
    # ------------------------------------------------------------------

    def _query_ip(self, query: dict) -> Box:
        # intersection of 2i, then one more projection
        inter_box  = self._query_2i({
            'anchors':   query['anchors'],
            'relations': query['relations'][:2],
        })
        rel_box = self.relation_embed(
            torch.tensor([int(query['relations'][2])], device=self.device))
        w1, w2  = 1.0, 1.0
        return self.proj_op.path_projection(inter_box, rel_box, w1, w2)

    # ------------------------------------------------------------------
    # pi  –  projection then intersection
    # ------------------------------------------------------------------

    def _query_pi(self, query: dict) -> Box:
        proj_box = self._query_2p({
            'anchor':    query['anchor'],
            'relations': query['relations'][:2],
        })
        b2 = self._query_1p({
            'anchor':    int(query['anchors'][1]),
            'relations': [int(query['relations'][2])],
        })
        return self._intersect_boxes(
            [proj_box, b2],
            [int(query['relations'][1]), int(query['relations'][2])],
            [int(query['anchor']),       int(query['anchors'][1])],
        )

    # ------------------------------------------------------------------
    # 2u / up  –  union queries  (via DNF)
    # ------------------------------------------------------------------

    def _query_2u(self, query: dict) -> List[Box]:
        """Returns a list of boxes (one per disjunct) for DNF union."""
        b1 = self._query_1p({'anchor': query['anchors'][0],
                              'relations': [query['relations'][0]]})
        b2 = self._query_1p({'anchor': query['anchors'][1],
                              'relations': [query['relations'][1]]})
        return union_via_dnf([b1, b2])

    def _query_up(self, query: dict) -> List[Box]:
        disjunct_boxes = self._query_2u({
            'anchors':   query['anchors'],
            'relations': query['relations'][:2],
        })
        rel_box = self.relation_embed(
            torch.tensor([int(query['relations'][2])], device=self.device))
        result = []
        for b in disjunct_boxes:
            result.append(self.proj_op.path_projection(b, rel_box))
        return result

    # ------------------------------------------------------------------
    # 2d  –  difference:  b_1 \ b_2
    # ------------------------------------------------------------------

    def _query_2d(self, query: dict) -> Box:
        b1 = self._query_1p({'anchor': query['anchors'][0],
                              'relations': [query['relations'][0]]})
        b2 = self._query_1p({'anchor': query['anchors'][1],
                              'relations': [query['relations'][1]]})

        return self._negate_boxes(
            b1, [b2],
            int(query['relations'][0]),
            [int(query['relations'][1])],
            [int(query['anchors'][0])],
        )

    # ------------------------------------------------------------------
    # 3d  –  difference:  b_1 \ (b_2 ∪ b_3)
    # ------------------------------------------------------------------

    def _query_3d(self, query: dict) -> Box:
        b1 = self._query_1p({'anchor': query['anchors'][0],
                              'relations': [query['relations'][0]]})
        b2 = self._query_1p({'anchor': query['anchors'][1],
                              'relations': [query['relations'][1]]})
        b3 = self._query_1p({'anchor': query['anchors'][2],
                              'relations': [query['relations'][2]]})

        return self._negate_boxes(
            b1, [b2, b3],
            int(query['relations'][0]),
            [int(query['relations'][1]), int(query['relations'][2])],
            [int(query['anchors'][0])],
        )

    # ------------------------------------------------------------------
    # dp  –  projection then difference
    # ------------------------------------------------------------------

    def _query_dp(self, query: dict) -> Box:
        proj_box = self._query_2p({
            'anchor':    query['anchor'],
            'relations': query['relations'][:2],
        })
        neg_box = self._query_1p({
            'anchor':    int(query['anchors'][-1]),
            'relations': [int(query['relations'][-1])],
        })
        return self._negate_boxes(
            proj_box, [neg_box],
            int(query['relations'][1]),
            [int(query['relations'][-1])],
            [int(query['anchor'])],
        )

    # ------------------------------------------------------------------
    # Shared intersection helper
    # ------------------------------------------------------------------

    def _intersect_boxes(self,
                          boxes:     List[Box],
                          relations: List[int],
                          anchors:   List[int]) -> Box:
        """
        Run VEM to get formula weights & entity memberships, then call
        IntersectionOperator.
        """
        fw, global_w, entity_embeds, entity_probs = \
            self._get_intersection_weights(relations, anchors)

        return self.inter_op(
            boxes,
            [torch.tensor(w, device=self.device) for w in fw],
            torch.tensor(global_w, device=self.device),
            entity_embeds,
            entity_probs,
        )

    # ------------------------------------------------------------------
    # Shared negation helper
    # ------------------------------------------------------------------

    def _negate_boxes(self,
                       positive_box:      Box,
                       negative_boxes:    List[Box],
                       positive_relation: int,
                       negative_relations: List[int],
                       anchor_entities:   List[int]) -> Box:
        """
        Run VEM to get weights for the negation operator.
        """
        if self._mln_builder is None:
            # Fallback: equal weights
            pos_w    = torch.tensor(1.0, device=self.device)
            neg_ws   = [torch.tensor(1.0, device=self.device)
                        for _ in negative_boxes]
            global_w = torch.tensor(float(1 + len(negative_boxes)),
                                    device=self.device)
            ee = torch.zeros(0, self.dim, device=self.device)
            ep = torch.zeros(0, device=self.device)
            return self.neg_op(positive_box, negative_boxes,
                               pos_w, neg_ws, global_w, ee, ep)

        node_list, edge_set, observed, _ = \
            self._mln_builder.build_for_negation(
                positive_relation, negative_relations, anchor_entities)

        if len(node_list) == 0:
            pos_w    = torch.tensor(1.0, device=self.device)
            neg_ws   = [torch.tensor(1.0, device=self.device)
                        for _ in negative_boxes]
            global_w = torch.tensor(1.0, device=self.device)
            ee = torch.zeros(0, self.dim, device=self.device)
            ep = torch.zeros(0, device=self.device)
        else:
            posterior, fw_tensor = self.vem(
                node_list, edge_set, observed, self._mln_builder.rules)

            num_pos   = sum(1 for n in node_list if n[1] == positive_relation)
            num_neg   = len(node_list) - num_pos
            pos_w     = fw_tensor[:1].mean()
            neg_ws    = [fw_tensor[1:2].mean() for _ in negative_boxes]
            global_w  = fw_tensor.sum()

            entity_embeds, entity_probs = \
                self._posterior_to_entity_embeddings(node_list, posterior)

        return self.neg_op(positive_box, negative_boxes,
                           pos_w, neg_ws, global_w,
                           entity_embeds, entity_probs)

    # ------------------------------------------------------------------
    # MLN weight extraction helpers
    # ------------------------------------------------------------------

    def _get_projection_weights(self,
                                  anchor: int,
                                  rel:    int
                                 ) -> Tuple[Optional[torch.Tensor],
                                            Optional[torch.Tensor]]:
        """
        Run VEM for initial projection.  Returns (entity_embeds, weights).
        """
        if self._mln_builder is None:
            return None, None

        node_list, edge_set, observed, _ = \
            self._mln_builder.build_for_projection(anchor, rel)

        if len(node_list) == 0:
            return None, None

        posterior, _ = self.vem(
            node_list, edge_set, observed, self._mln_builder.rules)

        return self._posterior_to_entity_embeddings(node_list, posterior)

    def _get_path_weights(self,
                           anchor: int,
                           r1:     int,
                           r2:     int) -> Tuple[float, float]:
        """
        Compute path projection weights w1, w2 for the formula
          r1(anchor, x) → r_new(x)   and  r2(x) → r_new(x)
        using discriminative learning via the VEM module.
        """
        if self._mln_builder is None:
            return 1.0, 1.0

        # Use intersection builder with both relations
        node_list, edge_set, observed, _ = \
            self._mln_builder.build_for_intersection([r1, r2], [anchor, anchor])

        if len(node_list) == 0:
            return 1.0, 1.0

        _, fw_tensor = self.vem(
            node_list, edge_set, observed, self._mln_builder.rules)

        w1 = fw_tensor[0].item() if fw_tensor.numel() > 0 else 1.0
        w2 = fw_tensor[1].item() if fw_tensor.numel() > 1 else 1.0
        return max(w1, 1e-6), max(w2, 1e-6)

    def _get_intersection_weights(self,
                                    relations: List[int],
                                    anchors:   List[int]
                                   ) -> Tuple[List[float], float,
                                              torch.Tensor, torch.Tensor]:
        """
        Run VEM for intersection to obtain formula weights and entity probs.
        """
        if self._mln_builder is None:
            n = len(relations)
            ee = torch.zeros(0, self.dim, device=self.device)
            ep = torch.zeros(0, device=self.device)
            return [1.0] * n, float(n), ee, ep

        node_list, edge_set, observed, _ = \
            self._mln_builder.build_for_intersection(relations, anchors)

        if len(node_list) == 0:
            n = len(relations)
            ee = torch.zeros(0, self.dim, device=self.device)
            ep = torch.zeros(0, device=self.device)
            return [1.0] * n, float(n), ee, ep

        posterior, fw_tensor = self.vem(
            node_list, edge_set, observed, self._mln_builder.rules)

        n = len(relations)
        fw_list  = [fw_tensor[i].item() if i < fw_tensor.numel() else 1.0
                    for i in range(n)]
        global_w = sum(fw_list)

        entity_embeds, entity_probs = \
            self._posterior_to_entity_embeddings(node_list, posterior)

        return fw_list, global_w, entity_embeds, entity_probs

    def _posterior_to_entity_embeddings(
            self,
            node_list: List[tuple],
            posterior:  torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map VEM posterior over MLN nodes to entity embeddings and weights.
        """
        entity_ids = []
        probs      = []

        for i, (h, r, t) in enumerate(node_list):
            # Use the tail entity as the "answer entity" for this fact-node
            entity_ids.append(t)
            probs.append(posterior[i].item() if i < len(posterior) else 0.0)

        if not entity_ids:
            return (torch.zeros(0, self.dim, device=self.device),
                    torch.zeros(0, device=self.device))

        # Deduplicate by entity, keeping max probability
        entity_prob: Dict[int, float] = {}
        for eid, p in zip(entity_ids, probs):
            entity_prob[eid] = max(entity_prob.get(eid, 0.0), p)

        unique_ids   = list(entity_prob.keys())
        unique_probs = [entity_prob[e] for e in unique_ids]

        id_tensor    = torch.tensor(unique_ids,   dtype=torch.long,
                                    device=self.device)
        prob_tensor  = torch.tensor(unique_probs, dtype=torch.float32,
                                    device=self.device)
        prob_tensor  = F.softmax(prob_tensor, dim=0)

        embeds = self.entity_embed(id_tensor)    # (m, d)
        return embeds, prob_tensor

    # ------------------------------------------------------------------
    # Scoring for evaluation
    # ------------------------------------------------------------------

    def score_entities(self,
                        query_box:   Box,
                        entity_ids:  Optional[torch.Tensor] = None,
                        all_entities: bool = True) -> torch.Tensor:
        """
        Compute the negative entity-to-box distance for candidate entities.
        Lower distance = higher score = more likely to be an answer.

        Parameters
        ----------
        query_box    : Box
        entity_ids   : (K,) optional candidate entity ids
        all_entities : if True, score all entities

        Returns
        -------
        scores : (K,) tensor  (negative distance)
        """
        if all_entities or entity_ids is None:
            entity_ids = torch.arange(self.entity_embed.embed.num_embeddings,
                                      device=self.device)

        embeds = self.entity_embed(entity_ids)  # (K, d)
        box_c  = query_box.center.expand(embeds.size(0), -1)
        box_o  = query_box.offset.expand(embeds.size(0), -1)
        b      = Box(box_c, box_o)

        dist = entity_to_box_distance(embeds, b, alpha=self.alpha_dist)
        return -dist    # higher is better

    # ------------------------------------------------------------------
    # Training step (single batch)
    # ------------------------------------------------------------------

    def training_step(self,
                       pos_entity:   torch.Tensor,
                       neg_entities: torch.Tensor,
                       query_box:    Box) -> torch.Tensor:
        """
        Compute the training loss for a (positive entity, negative entities,
        query box) triplet.
        """
        pos_embed = self.entity_embed(pos_entity)      # (d,)
        neg_embed = self.entity_embed(neg_entities)    # (k, d)
        return training_loss(pos_embed, query_box, neg_embed,
                             gamma=self.gamma, alpha=self.alpha_dist)
