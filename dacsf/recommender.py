# ─────────────────────────────────────────────────────────────────────────────
#  recommender.py  —  Inference: generate Top-K recommendations for a user
#
#  recommend_for_user(user_idx, app_state, top_k)
#      → list of dicts with rank, item_id, title, score, attention weights
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
from collections import defaultdict

from config import CFG
from features import item_feats, user_feats, build_hist_dict


def recommend_for_user(user_idx: int, app_state: dict, top_k: int = 10):
    """
    Generates Top-K personalized recommendations for a given user.

    For cold-start users (no training history) the model relies entirely on
    demographic features; the history modality receives a near-zero attention
    weight from AFFN automatically.

    Returns:
        List of dicts:
        [
          {
            "rank": 1,
            "item_id": 42,
            "title": "Toy Story (1995)",
            "score": 0.8741,
            "attention": {"text": 0.41, "image": 0.35, "category": 0.24}
          }, ...
        ]
    """
    model = app_state.get("model")
    data  = app_state.get("data")

    if model is None or data is None:
        return []

    device = torch.device(CFG.get_device())
    model.eval()

    # Build history lookup from training data
    hist = build_hist_dict(data["train_df"])

    # Items the user has already seen → exclude from recommendations
    seen  = set(hist.get(user_idx, []))
    cands = np.array([i for i in range(data["n_items"]) if i not in seen])

    if not len(cands):
        return []

    with torch.no_grad():
        uf    = user_feats(np.array([user_idx] * len(cands)), data, hist, device)
        if_   = item_feats(cands, data, device)
        scores, _, alphas = model(uf, if_)

        scores_np = scores.detach().cpu().numpy()
        alphas_np = alphas.detach().cpu().numpy()   # [n_cands, 3]

    # Rank by descending score
    top_idx = scores_np.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        item_id = int(cands[idx])
        alp     = alphas_np[idx]
        results.append(dict(
            rank      = rank + 1,
            item_id   = item_id,
            title     = data["title_map"].get(item_id, f"Item {item_id}"),
            score     = round(float(scores_np[idx]), 4),
            attention = dict(
                text     = round(float(alp[0]), 3),
                image    = round(float(alp[1]), 3),
                category = round(float(alp[2]), 3),
            )
        ))

    return results
