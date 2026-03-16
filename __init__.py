"""
mln_box_embedding
=================
MLN-Enhanced Box Embedding for Complex Query Answering.

Public API:
    MLNBoxEmbedding   – the main model class
    RuleMiner         – Horn rule extractor
    MLNBuilder        – MLN subgraph builder
    VEMInference      – variational EM MLN inference
    CQADataset        – query dataset
    evaluate_model    – CQA evaluation
"""

from .box_embedding   import (Box, EntityEmbedding, RelationBoxEmbedding,
                               ProjectionOperator, IntersectionOperator,
                               NegationOperator, entity_to_box_distance,
                               training_loss, union_via_dnf)
from .vem_inference   import VEMInference
from .mln_builder     import RuleMiner, MLNBuilder
from .model           import MLNBoxEmbedding
from .query_dataset   import CQADataset, ALL_QUERY_TYPES
from .evaluation_cqa  import evaluate_model, evaluate_query

__version__ = "1.0.0"
__all__ = [
    "Box", "EntityEmbedding", "RelationBoxEmbedding",
    "ProjectionOperator", "IntersectionOperator",
    "NegationOperator", "entity_to_box_distance",
    "training_loss", "union_via_dnf",
    "VEMInference",
    "RuleMiner", "MLNBuilder",
    "MLNBoxEmbedding",
    "CQADataset", "ALL_QUERY_TYPES",
    "evaluate_model", "evaluate_query",
]
