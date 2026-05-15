# ─────────────────────────────────────────────────────────────────────────────
#  datasets.py  —  PyTorch Dataset wrappers
#
#  BPRDataset     — yields (user, positive_item, negative_item) triples
#                   for BPR pairwise ranking training
#  RatingDataset  — yields (user, item, rating) for MAE / RMSE evaluation
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from config import CFG


class BPRDataset(Dataset):
    """
    Popularity-biased negative sampling:
        P(j as negative) ∝ freq(j)^0.75
    This avoids trivially easy negatives (rare items) while still
    not always sampling the most popular ones.
    """

    def __init__(self, df, n_items, item_freqs):
        self.users     = df["user_idx"].values
        self.pos_items = df["item_idx"].values
        self.n_items   = n_items

        # Map each user → set of interacted items (for rejection sampling)
        self.user_pos = defaultdict(set)
        for u, i in zip(self.users, self.pos_items):
            self.user_pos[u].add(i)

        # Compute sampling distribution once
        w = np.power(item_freqs + 1, CFG.neg_sample_pow)
        self.weights = w / w.sum()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u   = self.users[idx]
        pos = self.pos_items[idx]
        # Rejection sample a negative item not seen by this user
        while True:
            neg = np.random.choice(self.n_items, p=self.weights)
            if neg not in self.user_pos[u]:
                break
        return u, pos, neg


class RatingDataset(Dataset):
    """
    Simple dataset for (user, item, rating) used during MAE / RMSE
    computation on val / test sets.
    """

    def __init__(self, df):
        self.users   = df["user_idx"].values
        self.items   = df["item_idx"].values
        self.ratings = df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
