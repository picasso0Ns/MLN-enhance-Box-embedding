"""
box_embedding.py
================
Core box embedding representation and four logical operators:
  - Projection (initial + path)
  - Intersection
  - Negation
  - Union (via DNF transformation)

Implements the MLN-enhanced box embedding framework described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Box embedding primitive
# ---------------------------------------------------------------------------

class Box:
    """A d-dimensional axis-aligned box defined by center and non-negative offset."""

    def __init__(self, center: torch.Tensor, offset: torch.Tensor):
        # center: (..., d)   offset: (..., d)  -- offset >= 0
        self.center = center
        self.offset = F.softplus(offset)   # enforce non-negativity

    @property
    def min_embed(self) -> torch.Tensor:
        return self.center - self.offset

    @property
    def max_embed(self) -> torch.Tensor:
        return self.center + self.offset

    def __repr__(self):
        return (f"Box(center={self.center.shape}, offset={self.offset.shape})")


# ---------------------------------------------------------------------------
# Entity-to-Box distance  (Sec. IV-D of the paper)
# ---------------------------------------------------------------------------

def entity_to_box_distance(entity: torch.Tensor,
                            box: Box,
                            alpha: float = 0.02) -> torch.Tensor:
    """
    Compute the distance between entity point-embedding(s) and a box.

    d_box(e; b) = d_outside(e; b) + alpha * d_inside(e; b)

    Parameters
    ----------
    entity : Tensor of shape (..., d)
    box    : Box  (center and offset of matching batch shape)
    alpha  : scalar in (0, 1) that down-weights inside distance

    Returns
    -------
    Tensor of shape (...)  – one scalar per (entity, box) pair
    """
    b_min = box.min_embed   # (..., d)
    b_max = box.max_embed   # (..., d)

    # outside distance – L1 norm of the "excess" beyond the box walls
    d_outside = torch.norm(
        F.relu(entity - b_max) + F.relu(b_min - entity),
        p=1, dim=-1
    )

    # inside distance – L1 distance from center to the projection of entity inside box
    projected = torch.min(b_max, torch.max(b_min, entity))   # clamp to box
    d_inside  = torch.norm(box.center - projected, p=1, dim=-1)

    return d_outside + alpha * d_inside


# ---------------------------------------------------------------------------
# Learnable relation box embeddings
# ---------------------------------------------------------------------------

class RelationBoxEmbedding(nn.Module):
    """
    Each relation r is represented as a box r = (r^c, r^o) ∈ R^{2d}.
    """

    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.center = nn.Embedding(num_relations, dim)
        # Raw offset; will be passed through softplus during forward
        self.offset = nn.Embedding(num_relations, dim)

        nn.init.uniform_(self.center.weight, -0.5, 0.5)
        nn.init.uniform_(self.offset.weight,  0.0, 0.1)

    def forward(self, relation_ids: torch.Tensor) -> Box:
        """Return Box objects for the given relation ids."""
        c = self.center(relation_ids)
        o = self.offset(relation_ids)
        return Box(c, o)


# ---------------------------------------------------------------------------
# Projection operator  (Sec. IV-B of the paper)
# ---------------------------------------------------------------------------

class ProjectionOperator(nn.Module):
    """
    Two-mode projection operator:

    (1) Initial projection  – from an anchor entity embedding + relation box.
    (2) Path projection     – from an existing box + relation box, using the
                               MLN-derived dependency weights w1, w2.

    For the full MLN-weighted version the caller should supply the weights
    returned by the VEMInference module.  A fallback to equal weights is
    provided when they are absent.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def initial_projection(self,
                           entity_embeds: torch.Tensor,
                           entity_weights: torch.Tensor,
                           relation_box: Box) -> Box:
        """
        Eq. (3) of the paper:

          b_new^c = r^c + Σ_i  w_i * e_i
          b_new^o = r^o + Max(Σ_i  w_i * d(e_i, r^o))

        Parameters
        ----------
        entity_embeds   : (n, d)  point embeddings of O_r(anchor)
        entity_weights  : (n,)    MLN-derived membership weights  Σ w_i = 1
        relation_box    : Box with center/offset shape (d,) or (1, d)

        Returns
        -------
        Box
        """
        # Weighted combination of entity embeddings
        w = entity_weights.unsqueeze(-1)          # (n, 1)
        weighted_center = torch.sum(w * entity_embeds, dim=0)  # (d,)

        r_c = relation_box.center.squeeze(0)
        r_o = relation_box.offset.squeeze(0)

        new_center = r_c + weighted_center

        # Weighted distances between entity embeddings and relation offset
        dist = torch.norm(entity_embeds - r_o.unsqueeze(0), p=2, dim=-1)  # (n,)
        weighted_dist = torch.sum(entity_weights * dist)
        new_offset = r_o + weighted_dist

        return Box(new_center.unsqueeze(0), new_offset.unsqueeze(0))

    def path_projection(self,
                        input_box: Box,
                        relation_box: Box,
                        w1: float = 1.0,
                        w2: float = 1.0) -> Box:
        """
        Eq. (5) of the paper:

          b_new^c = ( b_h^c + (w1/w2) * r^c ) / 2
          b_new^o = Max(r^o, b_h^o)

        Parameters
        ----------
        input_box    : Box  b_h
        relation_box : Box  r
        w1, w2       : MLN formula weights
        """
        ratio = (w1 / (w2 + 1e-10))
        b_c = input_box.center
        r_c = relation_box.center
        r_o = relation_box.offset

        new_center = (b_c + ratio * r_c) / 2.0
        new_offset = torch.max(relation_box.offset, input_box.offset)

        return Box(new_center, new_offset)


# ---------------------------------------------------------------------------
# Intersection operator  (Sec. IV-C of the paper)
# ---------------------------------------------------------------------------

class IntersectionOperator(nn.Module):
    """
    Eq. (8) of the paper:

      b_new^c = α * Σ_i (w_{r_i} / w_r) * r_i^c / n
              + β * Σ_i p_i * e_i / m
      b_new^o = Max over i of r_i^o

    Parameters
    ----------
    alpha, beta : scalar mixture coefficients (learnable or fixed)
    """

    def __init__(self, dim: int, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.dim   = dim
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))

    def forward(self,
                input_boxes:    List[Box],
                formula_weights: List[torch.Tensor],
                global_w:        torch.Tensor,
                entity_embeds:   torch.Tensor,
                entity_probs:    torch.Tensor) -> Box:
        """
        Parameters
        ----------
        input_boxes     : list of n Box objects  (each (1, d))
        formula_weights : list of n scalar tensors w_{r_i} from MLN
        global_w        : scalar tensor w_r  (normalisation denominator)
        entity_embeds   : (m, d) embeddings of entities in the intersection
        entity_probs    : (m,)   MLN-derived membership probabilities p_i
        """
        n = len(input_boxes)
        assert n == len(formula_weights), "Need one weight per box"

        # Weighted sum of relation centers
        rel_center_sum = torch.zeros(self.dim,
                                     device=input_boxes[0].center.device)
        for box, w_r_i in zip(input_boxes, formula_weights):
            rel_center_sum = rel_center_sum + (w_r_i / (global_w + 1e-10)) * box.center.squeeze(0)

        # Max of relation offsets
        offsets = torch.stack([box.offset.squeeze(0) for box in input_boxes], dim=0)  # (n, d)
        max_offset = torch.max(offsets, dim=0).values   # (d,)

        # Weighted entity contribution
        if entity_embeds.numel() > 0 and entity_probs.numel() > 0:
            m = entity_embeds.size(0)
            p = entity_probs.unsqueeze(-1)            # (m, 1)
            entity_center_sum = torch.sum(p * entity_embeds, dim=0) / m   # (d,)
        else:
            entity_center_sum = torch.zeros_like(rel_center_sum)

        alpha = torch.sigmoid(self.alpha)
        beta  = torch.sigmoid(self.beta)

        new_center = alpha * rel_center_sum / n + beta * entity_center_sum
        new_offset = max_offset

        return Box(new_center.unsqueeze(0), new_offset.unsqueeze(0))


# ---------------------------------------------------------------------------
# Negation operator  (Sec. IV-C of the paper)
# ---------------------------------------------------------------------------

class NegationOperator(nn.Module):
    """
    Negation follows the same VEM-based weight/membership computation as
    intersection, but the logical expression for box b_1 minus {b_2, ..., b_n} uses
    negated predicates for i >= 2:

      r_1(x, a_1) ∩ ¬r_2(x, a_2) ∩ … ∩ ¬r_n(x, a_n)

    The box computation re-uses IntersectionOperator with the appropriate
    weights derived from the negated MLN formulas.

    Concretely: the center/offset are computed the same way (Eq. 8 reused),
    but the formula_weights for the negated boxes are negated first:
      w_neg_i = -w_i   for i >= 2
    and then re-normalised so they remain positive via softplus.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.intersection = IntersectionOperator(dim)

    def forward(self,
                positive_box:    Box,
                negative_boxes:  List[Box],
                pos_formula_w:   torch.Tensor,
                neg_formula_ws:  List[torch.Tensor],
                global_w:        torch.Tensor,
                entity_embeds:   torch.Tensor,
                entity_probs:    torch.Tensor) -> Box:
        """
        Parameters
        ----------
        positive_box   : b_1 (the retained box)
        negative_boxes : [b_2, …, b_n] (boxes to exclude)
        pos_formula_w  : MLN weight for the positive formula
        neg_formula_ws : MLN weights for the negated formulas (one per neg box)
        global_w       : normalisation weight
        entity_embeds  : (m, d)
        entity_probs   : (m,) membership probabilities for the NEW box
        """
        # Negate the formula weights for excluded boxes
        all_boxes   = [positive_box] + negative_boxes
        all_weights = [pos_formula_w] + [F.softplus(-w) for w in neg_formula_ws]

        return self.intersection(
            all_boxes, all_weights, global_w,
            entity_embeds, entity_probs
        )


# ---------------------------------------------------------------------------
# Union operator – via DNF  (Sec. IV-C of the paper)
# ---------------------------------------------------------------------------

def union_via_dnf(query_boxes: List[Box]) -> List[Box]:
    """
    Per Sec. IV-C (following Query2Box), union is handled by converting the
    query into Disjunctive Normal Form (DNF).  Each disjunct produces an
    independent box; the answer entity set is the union of the answer sets of
    all disjuncts.

    In practice this function simply returns the list of boxes unchanged;
    the caller is responsible for taking the union of entity sets at
    inference time (i.e. an entity answers the query if it answers *any*
    of the returned boxes).

    Parameters
    ----------
    query_boxes : list of Box objects, one per disjunct

    Returns
    -------
    Same list of Box objects (caller handles the set-union logic).
    """
    return query_boxes


# ---------------------------------------------------------------------------
# Full entity / query encoder (shared embeddings)
# ---------------------------------------------------------------------------

class EntityEmbedding(nn.Module):
    """Point embedding for entities  e ∈ R^d."""

    def __init__(self, num_entities: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_entities, dim)
        nn.init.uniform_(self.embed.weight, -0.5, 0.5)

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(entity_ids)


# ---------------------------------------------------------------------------
# Training loss  (Sec. IV-E)
# ---------------------------------------------------------------------------

def training_loss(entity_embed: torch.Tensor,
                  query_box:    Box,
                  neg_embeds:   torch.Tensor,
                  gamma:        float = 12.0,
                  alpha:        float = 0.02) -> torch.Tensor:
    """
    Margin-based negative-sampling loss (Eq. 9 of the paper):

      L = -log σ(γ - d(e, b))
        - (1/k) Σ_i log σ(d(e'_i, b) - γ)

    Parameters
    ----------
    entity_embed : (d,)  positive answer entity
    query_box    : Box   current query box
    neg_embeds   : (k, d) negative sample embeddings
    gamma        : fixed margin scalar
    alpha        : inside-distance scaling factor for d_box
    """
    # Positive distance
    d_pos = entity_to_box_distance(entity_embed.unsqueeze(0), query_box, alpha)   # (1,)

    # Negative distances
    # Expand box to match neg_embeds batch
    neg_box = Box(
        query_box.center.expand(neg_embeds.size(0), -1),
        query_box.offset.expand(neg_embeds.size(0), -1),
    )
    d_neg = entity_to_box_distance(neg_embeds, neg_box, alpha)    # (k,)

    pos_loss = -F.logsigmoid(gamma - d_pos)
    neg_loss = -torch.mean(F.logsigmoid(d_neg - gamma))

    return pos_loss.mean() + neg_loss
