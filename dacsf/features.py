# ─────────────────────────────────────────────────────────────────────────────
#  features.py  —  Batch tensor construction helpers
#
#  These functions convert raw numpy arrays from `data` into device tensors
#  ready to be fed into DACSF encoders.
#
#  item_feats(idxs, data, device)
#      → (text_tensor, img_tensor, cat_tensor)
#
#  user_feats(idxs, data, hist_dict, device, max_hist=20)
#      → (demo_tensor, hist_embs_tensor, hist_mask_tensor)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch

from config import CFG


def item_feats(idxs, data, device):
    """
    Gathers text / image / category feature tensors for a batch of items.

    Args:
        idxs:   array-like of item indices
        data:   dict returned by data_loader.load_data()
        device: torch device

    Returns:
        tuple of three float32 tensors each [len(idxs), feature_dim]
    """
    t = torch.tensor(data["text_mat"][idxs],  dtype=torch.float32).to(device)
    v = torch.tensor(data["img_mat"][idxs],   dtype=torch.float32).to(device)
    c = torch.tensor(data["genre_mat"][idxs], dtype=torch.float32).to(device)
    return t, v, c


def user_feats(idxs, data, hist_dict, device, max_hist=20):
    """
    Gathers demographic + history feature tensors for a batch of users.

    History is encoded as mean-pooled text embeddings of previously interacted
    items (capped at max_hist). Users with no history (cold-start) receive an
    all-zero history tensor and mask — AFFN will suppress this modality.

    Args:
        idxs:      array-like of user indices
        data:      dict returned by data_loader.load_data()
        hist_dict: dict mapping user_idx → list of interacted item indices
        device:    torch device
        max_hist:  maximum number of historical items to consider

    Returns:
        (demo [B, 3],  hist_embs [B, max_hist, d],  hist_mask [B, max_hist])
    """
    B = len(idxs)
    d = CFG.latent_dim

    demo      = torch.tensor(data["demo_mat"][idxs], dtype=torch.float32).to(device)
    hist_embs = torch.zeros(B, max_hist, d, device=device)
    hist_mask = torch.zeros(B, max_hist,    device=device)

    for b, u in enumerate(idxs):
        h = list(hist_dict.get(u, []))[:max_hist]
        if not h:
            continue  # cold-start user — leave zeros, AFFN handles it
        # Use text features as a proxy for item identity in the history encoder
        feat_dim = min(data["text_mat"].shape[1], d)
        hf = torch.tensor(data["text_mat"][h], dtype=torch.float32).to(device)
        hist_embs[b, :len(h), :feat_dim] = hf[:, :feat_dim]
        hist_mask[b, :len(h)] = 1.0

    return demo, hist_embs, hist_mask


def build_hist_dict(train_df):
    """
    Pre-computes the user → [item list] mapping from the training DataFrame.
    Call once after data splitting; pass the result to user_feats().
    """
    from collections import defaultdict
    hist = defaultdict(list)
    for _, row in train_df.iterrows():
        hist[int(row["user_idx"])].append(int(row["item_idx"]))
    return hist
