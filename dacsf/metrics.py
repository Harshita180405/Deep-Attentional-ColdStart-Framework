# ─────────────────────────────────────────────────────────────────────────────
#  metrics.py  —  Evaluation functions
#
#  compute_mae_rmse()  — rating prediction quality (for ADLRS comparison)
#  compute_ndcg()      — ranking quality: Precision, Recall, NDCG, HR at K
# ─────────────────────────────────────────────────────────────────────────────

import math
import random
from collections import defaultdict

import numpy as np
import torch

from config import CFG
from features import item_feats, user_feats


def compute_mae_rmse(model, loader, data, hist_dict, device):
    """
    Computes Mean Absolute Error and Root Mean Squared Error on a
    RatingDataset loader. Predicted score (0-1) is rescaled to [1, 5].
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for users, items, ratings in loader:
            u_np, i_np = users.numpy(), items.numpy()
            uf = user_feats(u_np, data, hist_dict, device)
            if_ = item_feats(i_np, data, device)
            s, _, _ = model(uf, if_)
            # Rescale sigmoid output from (0,1) to rating scale (1,5)
            preds.extend((1 + s.cpu().numpy() * 4).tolist())
            trues.extend(ratings.numpy().tolist())

    p, t = np.array(preds), np.array(trues)
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return mae, rmse


def compute_ndcg(model, test_df, train_df, data, hist_dict, device,
                 k=10, n_sample_users=150):
    """
    Computes Precision@K, Recall@K, NDCG@K and HR@K.

    Evaluation follows the standard leave-one-out / sampled protocol:
    for each sampled user, score all unseen items and rank them.

    Args:
        n_sample_users: number of users to evaluate (random subset for speed).

    Returns:
        (precision, recall, ndcg, hit_rate)
    """
    model.eval()

    # Build ground-truth positive sets
    test_pos   = defaultdict(set)
    for _, r in test_df.iterrows():
        test_pos[r["user_idx"]].add(r["item_idx"])

    train_seen = defaultdict(set)
    for _, r in train_df.iterrows():
        train_seen[r["user_idx"]].add(r["item_idx"])

    eval_users = random.sample(
        [u for u in test_pos if test_pos[u]],
        min(n_sample_users, len(test_pos))
    )
    n_items = data["n_items"]
    precs, recs, ndcgs, hrs = [], [], [], []

    with torch.no_grad():
        for u in eval_users:
            # Score all items not seen during training
            cands = np.array([i for i in range(n_items) if i not in train_seen[u]])
            if not len(cands):
                continue

            uf    = user_feats(np.array([u] * len(cands)), data, hist_dict, device)
            if_   = item_feats(cands, data, device)
            s, _, _ = model(uf, if_)
            scores    = s.cpu().numpy()

            top_idx   = scores.argsort()[::-1][:k]
            top_k_set = set(cands[top_idx])
            rel       = test_pos[u]
            hits      = top_k_set & rel

            precs.append(len(hits) / k)
            recs.append(len(hits) / len(rel) if rel else 0.0)
            hrs.append(1.0 if hits else 0.0)

            # DCG
            dcg  = sum(1 / math.log2(r + 2)
                       for r, item in enumerate(cands[top_idx]) if item in rel)
            idcg = sum(1 / math.log2(r + 2) for r in range(min(len(rel), k)))
            ndcgs.append(dcg / idcg if idcg else 0.0)

    return (
        float(np.mean(precs)),
        float(np.mean(recs)),
        float(np.mean(ndcgs)),
        float(np.mean(hrs)),
    )
