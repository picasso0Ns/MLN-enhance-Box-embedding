"""
vem_inference.py
================
Variational EM (VEM) based MLN inference.

Implements Algorithm 1 from the paper:
  - E-step  : mean-field GCN  Q(V_u) = Softmax(A F_q W_q)
  - M-step  : predictive GCN  P(V_u|V_o) = Softmax(A F_p W_p)
  - Gradient: ∇_w E_q[log P(V_u)] = n(o) - n(Q(V_u))
  - Iterate until KL(P || Q) ≤ ϱ or max iterations reached
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Set


# ---------------------------------------------------------------------------
# One-layer Graph Convolutional Network  (used in both E and M steps)
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """
    Single-layer GCN:
        H' = Softmax( Â  F  W )
    where Â = D^{-1/2} (A + I) D^{-1/2}  (symmetric normalised adjacency).
    """

    def __init__(self, in_dim: int, out_dim: int, num_classes: int = 2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.out = nn.Linear(out_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, A_hat: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        A_hat : (N, N) normalised adjacency (sparse or dense)
        F     : (N, in_dim) feature matrix

        Returns
        -------
        (N, num_classes) softmax probabilities
        """
        H = F @ self.W.weight.T + self.W.bias      # (N, out_dim)
        H = torch.relu(H)
        H = A_hat @ H                               # graph convolution
        logits = self.out(H)                        # (N, num_classes)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Helpers: build adjacency matrix and feature matrices
# ---------------------------------------------------------------------------

def build_normalised_adjacency(node_list: List[tuple],
                                edge_set: Set[tuple],
                                device: torch.device) -> torch.Tensor:
    """
    Build the symmetric-normalised adjacency matrix Â for the MLN subgraph.

    Parameters
    ----------
    node_list : list of (h, r, t) triples that are MLN nodes (ordered)
    edge_set  : set of ((h1,r1,t1), (h2,r2,t2)) undirected edges
    device    : torch device

    Returns
    -------
    Â of shape (N, N)
    """
    N = len(node_list)
    node_index = {n: i for i, n in enumerate(node_list)}

    # Sparse adjacency (including self-loops)
    rows, cols = [], []
    for i in range(N):
        rows.append(i); cols.append(i)   # self-loop

    for n1, n2 in edge_set:
        i = node_index.get(n1)
        j = node_index.get(n2)
        if i is not None and j is not None:
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)   # undirected

    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    vals = torch.ones(len(rows), dtype=torch.float32, device=device)

    A = torch.zeros(N, N, dtype=torch.float32, device=device)
    A[rows, cols] = vals

    # D^{-1/2}
    deg = A.sum(dim=1).clamp(min=1.0)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))

    return D_inv_sqrt @ A @ D_inv_sqrt


def build_Fq_matrix(node_list: List[tuple],
                    observed_facts: Set[tuple],
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the feature matrix F_q for the E-step GCN.

    For each node v:
      label = 1  if v ∈ V_o (observed),  0 otherwise
    Feature vector f_q^v = one-hot label of the node (dim = 2 here, binary).

    Returns
    -------
    F_q : (N, 2) feature matrix
    labels : (N,) 0/1 ground-truth labels
    """
    N = len(node_list)
    F_q = torch.zeros(N, 2, dtype=torch.float32, device=device)
    labels = torch.zeros(N, dtype=torch.long, device=device)

    for i, node in enumerate(node_list):
        if node in observed_facts:
            F_q[i, 1] = 1.0
            labels[i] = 1
        else:
            F_q[i, 0] = 1.0

    return F_q, labels


def build_Fp_matrix(node_list: List[tuple],
                    formula_weights: Dict,
                    rules: List[tuple],
                    q_distribution: torch.Tensor,
                    device: torch.device) -> torch.Tensor:
    """
    Build the feature matrix F_p for the M-step GCN.

    For each node v, f_p^v = [w_i * n_v^r ; w_i * n_v^a] concatenated over
    all formulas (see Sec. IV-C, Fig. 4 in the paper).

    n_v^r = count of truth-value relation-nodes connected to formula f_i
    n_v^a = count of attribute-nodes connected to formula f_i
    w_i   = formula weight

    For simplicity we approximate the feature as a length-2m vector
    (m = number of formulas), where each pair (w_i * n_v^r, w_i * n_v^a)
    summarises the contribution of formula i to node v.

    Parameters
    ----------
    node_list       : ordered list of (h, r, t) MLN nodes
    formula_weights : dict mapping rule -> weight scalar
    rules           : list of (head_rel, (body_rels,)) rules
    q_distribution  : (N,) predicted labels from E-step (float in [0,1])
    device          : torch device

    Returns
    -------
    F_p : (N, 2*m) feature matrix   (m = number of formulas)
    """
    N = len(node_list)
    m = max(len(formula_weights), 1)
    F_p = torch.zeros(N, 2 * m, dtype=torch.float32, device=device)

    rule_list = list(formula_weights.keys())

    for fi, rule in enumerate(rule_list):
        head_rel, body_rels = rule
        w_i = formula_weights[rule]
        if isinstance(w_i, torch.Tensor):
            w_i = w_i.item()

        for ni, node in enumerate(node_list):
            h, r, t = node
            # n_v^r : how many relation nodes linked to this formula are "true"
            # approximate: q_distribution gives soft label
            n_vr = q_distribution[ni].item()

            # n_v^a : attributive count – here set to 0 (no attribute nodes)
            n_va = 0.0

            F_p[ni, 2 * fi]     = w_i * n_vr
            F_p[ni, 2 * fi + 1] = w_i * n_va

    return F_p


# ---------------------------------------------------------------------------
# VEM Inference Module
# ---------------------------------------------------------------------------

class VEMInference(nn.Module):
    """
    Variational EM based MLN inference (Algorithm 1 in the paper).

    Given:
      - The MLN graph structure (nodes, edges)
      - Observed facts V_o
      - Initial formula weights w
      - An unobserved fact set V_u

    Computes:
      - Approximate posterior P(V_u | V_o)
      - Updated formula weights w

    These are then used by the Projection / Intersection / Negation operators
    to compute weighted box embeddings.
    """

    def __init__(self,
                 gcn_hidden_dim: int = 64,
                 max_iter: int = 10,
                 kl_threshold: float = 0.01,
                 num_formulas: int = 10,
                 device: torch.device = None):
        super().__init__()

        if device is None:
            device = torch.device('cpu')
        self.device = device

        self.max_iter     = max_iter
        self.kl_threshold = kl_threshold
        self.num_formulas = num_formulas

        # E-step GCN
        self.gcn_q = GCNLayer(in_dim=2, out_dim=gcn_hidden_dim, num_classes=2)

        # M-step GCN  (input dim = 2 * num_formulas)
        self.gcn_p = GCNLayer(
            in_dim=max(2 * num_formulas, 2),
            out_dim=gcn_hidden_dim,
            num_classes=2
        )

        # Formula weights (one per formula / rule)
        self.formula_weights = nn.Parameter(
            torch.ones(num_formulas, dtype=torch.float32) * 0.1
        )

        self.to(device)

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def forward(self,
                node_list:       List[tuple],
                edge_set:        Set[tuple],
                observed_facts:  Set[tuple],
                rules:           List[tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run VEM inference on the MLN subgraph.

        Parameters
        ----------
        node_list      : list of (h, r, t) – all nodes in this MLN subgraph
        edge_set       : set of undirected edges between nodes
        observed_facts : set of (h, r, t) triples that are known true
        rules          : list of (head_rel, (body_rels,)) Horn clauses

        Returns
        -------
        posterior  : (N,) tensor – P(v = 1 | V_o) for each node
        fw_tensor  : (num_formulas,) tensor of updated formula weights
        """
        if len(node_list) == 0:
            return (torch.zeros(0, device=self.device),
                    self.formula_weights.abs())

        N = len(node_list)
        A_hat = build_normalised_adjacency(node_list, edge_set, self.device)

        # Build formula weight dict  (rule_i -> scalar weight_i)
        fw_dict = {}
        for fi, rule in enumerate(rules[:self.num_formulas]):
            fw_dict[rule] = self.formula_weights[fi].abs()

        # --- Initialise M-step ---
        F_q, labels = build_Fq_matrix(node_list, observed_facts, self.device)

        # Initial M-step pass to get P(V_u|V_o)
        F_p = build_Fp_matrix(
            node_list, fw_dict, rules,
            torch.zeros(N, device=self.device), self.device
        )
        # Pad / trim F_p to match gcn_p input dim
        expected_dim = max(2 * self.num_formulas, 2)
        if F_p.size(1) < expected_dim:
            pad = torch.zeros(N, expected_dim - F_p.size(1), device=self.device)
            F_p = torch.cat([F_p, pad], dim=1)
        elif F_p.size(1) > expected_dim:
            F_p = F_p[:, :expected_dim]

        P_dist = self.gcn_p(A_hat, F_p)   # (N, 2)
        P_pos  = P_dist[:, 1]             # P(v=1 | V_o)

        Q_prev = P_pos.detach().clone()

        # --- VEM iterations ---
        for _ in range(self.max_iter):
            # E-step: update Q
            Q_dist = self.gcn_q(A_hat, F_q)   # (N, 2)
            Q_pos  = Q_dist[:, 1]              # Q(v=1)

            # M-step: update P using Q from previous iteration
            F_p = build_Fp_matrix(
                node_list, fw_dict, rules, Q_prev, self.device
            )
            if F_p.size(1) < expected_dim:
                pad = torch.zeros(N, expected_dim - F_p.size(1), device=self.device)
                F_p = torch.cat([F_p, pad], dim=1)
            elif F_p.size(1) > expected_dim:
                F_p = F_p[:, :expected_dim]

            P_dist = self.gcn_p(A_hat, F_p)
            P_pos  = P_dist[:, 1]

            # Update formula weights via gradient  ∇_w E_q[log P(V_u)] = n(o) - n(Q)
            with torch.no_grad():
                n_o   = float(len(observed_facts))
                n_q   = Q_pos.sum().item()
                grad  = n_o - n_q
                # Simple gradient step on formula weights
                self.formula_weights.data += 0.01 * grad

            # Check KL divergence between P and Q
            kl = self._kl_divergence(P_pos, Q_pos)
            if kl.item() <= self.kl_threshold:
                break

            Q_prev = Q_pos.detach()

        return P_pos.detach(), self.formula_weights.abs().detach()

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def e_step_loss(self,
                    A_hat:          torch.Tensor,
                    F_q:            torch.Tensor,
                    labels:         torch.Tensor,
                    Q_prev:         torch.Tensor) -> torch.Tensor:
        """
        L_q = L_u + L_o   (Eq. 7 of the paper)

        L_u = -Σ_{v ∈ V_u} KL(Q'(V_u) || Q(V_u))
        L_o = -Σ_{v ∈ V_o} y_v log p_c(y_v)
        """
        Q_dist = self.gcn_q(A_hat, F_q)    # (N, 2)

        # L_o : supervised cross-entropy on observed nodes
        L_o = F.cross_entropy(Q_dist, labels)

        # L_u : KL divergence for unobserved nodes  (soft pseudo-labels)
        Q_pos   = Q_dist[:, 1]
        # Avoid log(0)
        eps = 1e-8
        L_u = -torch.mean(
            Q_prev * torch.log(Q_pos + eps) +
            (1 - Q_prev) * torch.log(1 - Q_pos + eps)
        )

        return L_o + L_u

    def m_step_loss(self,
                    A_hat:  torch.Tensor,
                    F_p:    torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        L_p = -Σ_{v ∈ V_u ∪ V_o} log p(y_v)   (Eq. 8 of the paper)
        """
        P_dist = self.gcn_p(A_hat, F_p)    # (N, 2)
        return F.cross_entropy(P_dist, labels)

    # ------------------------------------------------------------------
    # KL divergence helper
    # ------------------------------------------------------------------

    @staticmethod
    def _kl_divergence(p: torch.Tensor, q: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
        """
        Element-wise KL(P || Q) for Bernoulli distributions, averaged.
        """
        p = p.clamp(eps, 1 - eps)
        q = q.clamp(eps, 1 - eps)
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        return kl.mean()

    # ------------------------------------------------------------------
    # discriminative_learning (Eq. 6 – pseudo-likelihood)
    # ------------------------------------------------------------------

    def pseudo_likelihood(self,
                          node_list:      List[tuple],
                          edge_set:       Set[tuple],
                          observed_facts: Set[tuple]) -> torch.Tensor:
        """
        Discriminative learning objective (Eq. 6):

          max_w  Σ_{v ∈ V_o ∪ V_u}  log p[ v = 1_{v ∈ V_o} | MB(v) ]

        Uses the M-step GCN to approximate p[v | MB(v)].

        Returns the negative pseudo-log-likelihood (to be minimised).
        """
        N = len(node_list)
        if N == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        A_hat  = build_normalised_adjacency(node_list, edge_set, self.device)
        F_q, labels = build_Fq_matrix(node_list, observed_facts, self.device)

        Q_dist = self.gcn_q(A_hat, F_q)
        loss   = F.cross_entropy(Q_dist, labels)
        return loss
