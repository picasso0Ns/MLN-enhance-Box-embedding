"""
train.py
========
Two-stage training for the MLN-Enhanced Box Embedding model.

Stage 1 – Embedding pre-training:
  Train entity and relation box embeddings using link prediction loss
  (1p queries).  This mirrors the TransH pre-training used in the
  provided py files.

Stage 2 – Joint MLN+Box training:
  Train the full model (embeddings + VEM GCNs) on all query types
  simultaneously.  The VEM modules provide MLN formula weights and
  membership probabilities on-the-fly during training.

Usage
-----
  python train.py --dataset FB15k237 --dim 400 --epochs 200 --stage both

  # Pre-train only:
  python train.py --dataset FB15k237 --dim 400 --epochs 100 --stage pretrain

  # Fine-tune from checkpoint:
  python train.py --dataset FB15k237 --dim 400 --epochs 100 --stage joint \\
                  --ckpt checkpoints/FB15k237_pretrain.pt
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
from dataset_loader import KnowledgeGraphDataset
from query_dataset  import CQADataset, ALL_QUERY_TYPES
from model          import MLNBoxEmbedding
from mln_builder    import RuleMiner, MLNBuilder
from evaluation_cqa import evaluate_model, evaluate_link_prediction
from box_embedding  import Box


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Train MLN-Enhanced Box Embedding for CQA')

    # Dataset
    p.add_argument('--dataset', type=str, required=True,
                   help='Dataset name (FB15k, FB15k237, NELL995)')
    p.add_argument('--data_dir', type=str, default='.',
                   help='Root directory that contains dataset folders')

    # Model hyperparameters
    p.add_argument('--dim',         type=int,   default=400)
    p.add_argument('--gamma',       type=float, default=12.0)
    p.add_argument('--alpha',       type=float, default=0.02)
    p.add_argument('--gcn_hidden',  type=int,   default=64)
    p.add_argument('--num_formulas',type=int,   default=20)
    p.add_argument('--max_vem_iter',type=int,   default=5)
    p.add_argument('--kl_threshold',type=float, default=0.01)

    # Training
    p.add_argument('--stage',      type=str, default='both',
                   choices=['pretrain', 'joint', 'both'])
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--batch_size', type=int,   default=512)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--neg_ratio',  type=int,   default=128)
    p.add_argument('--max_queries',type=int,   default=50_000)
    p.add_argument('--eval_every', type=int,   default=10)
    p.add_argument('--save_every', type=int,   default=20)

    # Rule mining
    p.add_argument('--min_support',   type=int,   default=3)
    p.add_argument('--min_conf',      type=float, default=0.2)
    p.add_argument('--max_rules',     type=int,   default=200)
    p.add_argument('--max_rule_len',  type=int,   default=2)

    # I/O
    p.add_argument('--ckpt',        type=str, default=None,
                   help='Load checkpoint before training')
    p.add_argument('--save_dir',    type=str, default='checkpoints')
    p.add_argument('--query_file',  type=str, default=None,
                   help='Pre-built query pickle file')
    p.add_argument('--device',      type=str, default='auto')
    p.add_argument('--seed',        type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(pref: str) -> torch.device:
    if pref == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(pref)


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0)


# ---------------------------------------------------------------------------
# Stage 1 – Embedding pre-training (link prediction / 1p)
# ---------------------------------------------------------------------------

def pretrain_stage(model: MLNBoxEmbedding,
                   dataset: KnowledgeGraphDataset,
                   args,
                   device: torch.device) -> MLNBoxEmbedding:
    """Pre-train entity/relation embeddings with simple 1p loss."""
    print("\n=== Stage 1: Embedding pre-training ===")

    optimizer = optim.Adam(
        list(model.entity_embed.parameters()) +
        list(model.relation_embed.parameters()),
        lr=args.lr
    )

    train_loader = dataset.get_train_dataloader(
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio
    )

    best_mrr = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_pos, batch_neg in train_loader:
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for idx in range(batch_pos.size(0)):
                h, r, t = batch_pos[idx].tolist()
                neg_row  = batch_neg[idx * args.neg_ratio :
                                     (idx + 1) * args.neg_ratio]

                # Build 1p box
                q_box = model.answer_query('1p', {
                    'anchor':    h,
                    'relations': [r],
                })

                # Positive entity
                pos_e = torch.tensor([t], device=device)
                neg_e = neg_row[:, 2]    # tail entities of neg triples

                loss = model.training_step(pos_e, neg_e, q_box)
                total_loss = total_loss + loss

            total_loss = total_loss / max(batch_pos.size(0), 1)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed  = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"t={elapsed:.1f}s")

        if epoch % args.eval_every == 0:
            all_set = {(int(h), int(r), int(t))
                       for h, r, t in dataset.train_triples}
            all_set.update({(int(h), int(r), int(t))
                            for h, r, t in dataset.test_triples})
            m = evaluate_link_prediction(
                model, dataset.test_triples,
                dataset.num_entities, all_set,
                verbose=False
            )
            print(f"  [eval] MRR={m['MRR']:.4f}  H@10={m['Hits@10']:.4f}")
            if m['MRR'] > best_mrr:
                best_mrr = m['MRR']
                save_checkpoint(model, optimizer, epoch,
                                os.path.join(args.save_dir,
                                             f'{args.dataset}_pretrain_best.pt'))

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(args.save_dir,
                                         f'{args.dataset}_pretrain_ep{epoch}.pt'))

    print(f"  Pre-training done. Best LP MRR = {best_mrr:.4f}")
    return model


# ---------------------------------------------------------------------------
# Stage 2 – Joint MLN + Box training
# ---------------------------------------------------------------------------

def joint_train_stage(model:   MLNBoxEmbedding,
                      dataset: KnowledgeGraphDataset,
                      args,
                      device:  torch.device) -> MLNBoxEmbedding:
    """Full joint training on all query types with MLN weights."""
    print("\n=== Stage 2: Joint MLN+Box training ===")

    # Build query dataset
    cqa_ds = CQADataset(
        triples       = dataset.train_triples,
        num_entities  = dataset.num_entities,
        num_relations = dataset.num_relations,
        query_types   = ALL_QUERY_TYPES,
        query_file    = args.query_file,
        max_queries   = args.max_queries,
        neg_ratio     = args.neg_ratio,
    )

    loader = DataLoader(
        cqa_ds,
        batch_size  = 1,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = CQADataset.collate_fn,
    )

    # Test CQA dataset for evaluation
    cqa_test = CQADataset(
        triples       = np.concatenate([dataset.train_triples,
                                        dataset.test_triples], axis=0),
        num_entities  = dataset.num_entities,
        num_relations = dataset.num_relations,
        query_types   = ALL_QUERY_TYPES,
        max_queries   = min(5000, args.max_queries // 10),
        neg_ratio     = args.neg_ratio,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_mrr  = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch_list in loader:
            sample = batch_list[0]
            qt     = sample['query_type']
            struct = sample['structure']
            pos_e  = torch.tensor([sample['pos_answer']], device=device)
            neg_e  = torch.tensor(sample['neg_answers'], device=device)

            optimizer.zero_grad()
            try:
                q_box = model.answer_query(qt, struct)

                # Handle union (multi-box) queries
                if isinstance(q_box, list):
                    loss = torch.tensor(0.0, device=device)
                    for b in q_box:
                        loss = loss + model.training_step(pos_e, neg_e, b)
                    loss = loss / len(q_box)
                else:
                    loss = model.training_step(pos_e, neg_e, q_box)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            except Exception as e:
                # Skip problematic queries silently
                pass

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"batches={n_batches}  t={elapsed:.1f}s")

        if epoch % args.eval_every == 0:
            test_sample = [cqa_test[i] for i in range(min(500, len(cqa_test)))]
            m = evaluate_model(model, test_sample, verbose=False)
            ov = m['overall']
            print(f"  [eval] MRR={ov['MRR']:.4f}  "
                  f"H@1={ov['Hits@1']:.4f}  "
                  f"H@3={ov['Hits@3']:.4f}  "
                  f"H@10={ov['Hits@10']:.4f}")

            if ov['MRR'] > best_mrr:
                best_mrr = ov['MRR']
                save_checkpoint(model, optimizer, epoch,
                                os.path.join(args.save_dir,
                                             f'{args.dataset}_joint_best.pt'))

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(args.save_dir,
                                         f'{args.dataset}_joint_ep{epoch}.pt'))

    print(f"  Joint training done. Best overall MRR = {best_mrr:.4f}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Dataset : {args.dataset}")
    print(f"Device  : {device}")
    print(f"Dim     : {args.dim}")

    # ---- Load KG dataset ----
    dataset = KnowledgeGraphDataset(args.dataset)

    # ---- Mine rules ----
    print("\nMining Horn rules …")
    miner = RuleMiner(
        min_support    = args.min_support,
        min_confidence = args.min_conf,
        max_rules      = args.max_rules,
        max_rule_len   = args.max_rule_len,
    )
    all_triples = np.concatenate([dataset.train_triples,
                                  dataset.test_triples], axis=0)
    rules, rule_weights = miner.mine(dataset.train_triples,
                                     dataset.num_relations)
    print(f"Found {len(rules)} rules")

    # ---- Build MLN structure builder ----
    builder = MLNBuilder(
        rules          = rules,
        rule_weights   = rule_weights,
        all_triples    = all_triples,
        train_triples  = dataset.train_triples,
        max_nodes      = 1000,
        max_nodes_per_rel = 100,
    )

    # ---- Build model ----
    model = MLNBoxEmbedding(
        num_entities   = dataset.num_entities,
        num_relations  = dataset.num_relations,
        dim            = args.dim,
        gamma          = args.gamma,
        alpha          = args.alpha,
        gcn_hidden     = args.gcn_hidden,
        max_vem_iter   = args.max_vem_iter,
        kl_threshold   = args.kl_threshold,
        num_formulas   = args.num_formulas,
        device         = device,
    )
    model.set_mln_builder(builder)
    print(f"Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ---- Load checkpoint ----
    start_epoch = 0
    if args.ckpt:
        start_epoch = load_checkpoint(model, None, args.ckpt, device)
        print(f"Resumed from {args.ckpt} (epoch {start_epoch})")

    # ---- Training ----
    if args.stage in ('pretrain', 'both'):
        model = pretrain_stage(model, dataset, args, device)

    if args.stage in ('joint', 'both'):
        model = joint_train_stage(model, dataset, args, device)

    # ---- Final evaluation ----
    print("\n=== Final Evaluation ===")
    cqa_final = CQADataset(
        triples       = all_triples,
        num_entities  = dataset.num_entities,
        num_relations = dataset.num_relations,
        query_types   = ALL_QUERY_TYPES,
        max_queries   = 10_000,
        neg_ratio     = args.neg_ratio,
    )
    test_data = [cqa_final[i] for i in range(len(cqa_final))]
    evaluate_model(model, test_data, verbose=True)

    # Save final model
    final_path = os.path.join(args.save_dir, f'{args.dataset}_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == '__main__':
    main()
