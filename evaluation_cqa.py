"""
evaluation_cqa.py
=================
Evaluation metrics for Complex Query Answering (CQA).

Metrics (following the paper and standard benchmarks):
  - MRR   (Mean Reciprocal Rank)
  - Hit@1
  - Hit@3
  - Hit@10

Filtering: the "filtered" setting excludes known true answers from the
candidate list (standard practice in KGE evaluation).

Supports both:
  1. Single-box queries (1p, 2p, 3p, 2i, 3i, ip, pi, 2d, 3d, dp)
  2. Multi-box (DNF union) queries (2u, up)  – score = max over boxes
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union

from .box_embedding import Box, entity_to_box_distance
from .model import MLNBoxEmbedding


# ---------------------------------------------------------------------------
# Core per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(model:       MLNBoxEmbedding,
                   query_type:  str,
                   structure:   dict,
                   answers:     Set[int],
                   all_answers: Optional[Set[int]] = None,
                   alpha:       float = 0.02) -> Dict[str, float]:
    """
    Evaluate a single query and return MRR + Hits.

    Parameters
    ----------
    model        : MLNBoxEmbedding  (in eval mode)
    query_type   : str  e.g. '1p'
    structure    : query dict
    answers      : ground-truth answer set (the "hard" answers)
    all_answers  : all known positive answers (for filtering); defaults to answers
    alpha        : inside-distance scaling factor

    Returns
    -------
    dict with keys 'MRR', 'Hits@1', 'Hits@3', 'Hits@10'
    """
    if all_answers is None:
        all_answers = answers

    device = model.device
    N      = model.entity_embed.embed.num_embeddings

    # ---- Get query box(es) ----
    qt     = query_type.lower()
    result = model.answer_query(qt, structure)

    if isinstance(result, list):
        # Union query: result is a list of Box objects
        # Score = max over boxes
        boxes = result
        def score_fn(entity_ids):
            scores = torch.stack([
                -entity_to_box_distance(
                    model.entity_embed(entity_ids),
                    Box(b.center.expand(len(entity_ids), -1),
                        b.offset.expand(len(entity_ids), -1)),
                    alpha=alpha
                ) for b in boxes
            ], dim=0).max(dim=0).values
            return scores
    else:
        # Single box
        box = result
        def score_fn(entity_ids):
            emb = model.entity_embed(entity_ids)
            b   = Box(box.center.expand(len(entity_ids), -1),
                      box.offset.expand(len(entity_ids), -1))
            return -entity_to_box_distance(emb, b, alpha=alpha)

    # ---- Score all entities in batches ----
    batch_size = 512
    all_scores = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end    = min(start + batch_size, N)
            ids    = torch.arange(start, end, device=device)
            scores = score_fn(ids)
            all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores, dim=0).numpy()   # (N,)

    # ---- Compute ranks for each answer entity ----
    metrics = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}
    if not answers:
        return metrics

    for ans in answers:
        ans_score = all_scores[ans]

        # Filtered: remove other known positives before ranking
        filtered_scores = all_scores.copy()
        for other in all_answers:
            if other != ans:
                filtered_scores[other] = -np.inf

        rank = int(np.sum(filtered_scores > ans_score)) + 1

        metrics['MRR']    += 1.0 / rank
        metrics['Hits@1'] += float(rank <= 1)
        metrics['Hits@3'] += float(rank <= 3)
        metrics['Hits@10']+= float(rank <= 10)

    n = len(answers)
    for k in metrics:
        metrics[k] /= n

    return metrics


# ---------------------------------------------------------------------------
# Full dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model:      MLNBoxEmbedding,
                   test_data:  List[dict],
                   verbose:    bool = True,
                   log_every:  int  = 200) -> Dict[str, Dict[str, float]]:
    """
    Run evaluation over an entire test set and return per-type and
    overall metrics.

    Parameters
    ----------
    model     : MLNBoxEmbedding
    test_data : list of dicts as produced by CQADataset.__getitem__
    verbose   : print progress
    log_every : print interval

    Returns
    -------
    dict  {query_type -> {metric -> float},  'overall' -> {metric -> float}}
    """
    model.eval()

    per_type: Dict[str, Dict[str, List[float]]] = {}

    for i, sample in enumerate(test_data):
        qt         = sample['query_type']
        structure  = sample['structure']
        answers    = set(sample['all_answers'])

        if qt not in per_type:
            per_type[qt] = {'MRR': [], 'Hits@1': [], 'Hits@3': [], 'Hits@10': []}

        try:
            m = evaluate_query(model, qt, structure, answers)
            for k, v in m.items():
                per_type[qt][k].append(v)
        except Exception as e:
            if verbose:
                print(f"  Warning: failed on query {i} ({qt}): {e}")

        if verbose and (i + 1) % log_every == 0:
            print(f"  Evaluated {i+1}/{len(test_data)} …")

    # Aggregate
    result: Dict[str, Dict[str, float]] = {}
    all_mrr, all_h1, all_h3, all_h10 = [], [], [], []

    for qt, mdict in per_type.items():
        result[qt] = {k: float(np.mean(v)) for k, v in mdict.items()}
        all_mrr.extend(mdict['MRR'])
        all_h1.extend(mdict['Hits@1'])
        all_h3.extend(mdict['Hits@3'])
        all_h10.extend(mdict['Hits@10'])

    result['overall'] = {
        'MRR':    float(np.mean(all_mrr))  if all_mrr  else 0.0,
        'Hits@1': float(np.mean(all_h1))   if all_h1   else 0.0,
        'Hits@3': float(np.mean(all_h3))   if all_h3   else 0.0,
        'Hits@10':float(np.mean(all_h10))  if all_h10  else 0.0,
    }

    if verbose:
        _print_results(result)

    model.train()
    return result


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def _print_results(results: Dict[str, Dict[str, float]]):
    header = f"{'Query Type':<12} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}"
    sep    = '-' * len(header)
    print(sep)
    print(header)
    print(sep)

    # EPFO types first
    from .query_dataset import QUERY_TYPES_EPFO, QUERY_TYPES_NEG
    order = QUERY_TYPES_EPFO + QUERY_TYPES_NEG + ['overall']

    for qt in order:
        if qt not in results:
            continue
        m = results[qt]
        tag = '(overall)' if qt == 'overall' else ''
        print(f"{qt:<12} "
              f"{m['MRR']:>8.4f} "
              f"{m['Hits@1']:>8.4f} "
              f"{m['Hits@3']:>8.4f} "
              f"{m['Hits@10']:>8.4f}  {tag}")
    print(sep)


# ---------------------------------------------------------------------------
# Link prediction evaluation (for pre-training TransH stage)
# ---------------------------------------------------------------------------

def evaluate_link_prediction(model:    MLNBoxEmbedding,
                              triples:  np.ndarray,
                              num_entities: int,
                              all_triples_set: Set[Tuple[int,int,int]],
                              batch_size: int = 256,
                              verbose: bool = True
                              ) -> Dict[str, float]:
    """
    Standard filtered link prediction evaluation (head/tail corruption).

    Used during the first training stage where entity + relation embeddings
    are pre-trained with a simpler 1p (link prediction) objective.
    """
    model.eval()
    ranks = []

    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            batch = triples[i : i + batch_size]
            for h, r, t in batch:
                h, r, t = int(h), int(r), int(t)

                # Score all candidate tails
                all_ids = torch.arange(num_entities, device=model.device)
                q_box   = model.answer_query('1p',
                                             {'anchor': h, 'relations': [r]})
                embs    = model.entity_embed(all_ids)
                b       = Box(
                    q_box.center.expand(num_entities, -1),
                    q_box.offset.expand(num_entities, -1),
                )
                scores  = -entity_to_box_distance(embs, b).cpu().numpy()

                # Filter other known positives
                filtered = scores.copy()
                for e in range(num_entities):
                    if e != t and (h, r, e) in all_triples_set:
                        filtered[e] = -np.inf

                rank = int(np.sum(filtered > filtered[t])) + 1
                ranks.append(rank)

            if verbose and (i // batch_size) % 20 == 0:
                print(f"  LP eval: {i}/{len(triples)}")

    ranks = np.array(ranks)
    result = {
        'MRR':    float(np.mean(1.0 / ranks)),
        'Hits@1': float(np.mean(ranks <= 1)),
        'Hits@3': float(np.mean(ranks <= 3)),
        'Hits@10':float(np.mean(ranks <= 10)),
        'MR':     float(np.mean(ranks)),
    }

    if verbose:
        print(f"LP  MRR={result['MRR']:.4f}  "
              f"H@1={result['Hits@1']:.4f}  "
              f"H@3={result['Hits@3']:.4f}  "
              f"H@10={result['Hits@10']:.4f}")

    model.train()
    return result
