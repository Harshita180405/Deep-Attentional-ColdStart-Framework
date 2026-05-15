# ─────────────────────────────────────────────────────────────────────────────
#  data_loader.py  —  Download, parse and split MovieLens 100K
#
#  Public API
#  ----------
#  load_data()   → dict with keys: ratings, n_users, n_items, demo_mat,
#                   text_mat, img_mat, genre_mat, n_genres, title_map,
#                   u2idx, i2idx
#  split_data()  → train_df, val_df, test_df, cold_users, cold_items,
#                   item_freqs
# ─────────────────────────────────────────────────────────────────────────────

import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CFG


# ─── Download helper ──────────────────────────────────────────────────────────

def _download_movielens(log_fn=print):
    url  = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    dest = "./data/ml-100k.zip"
    if not os.path.exists(CFG.data_dir):
        os.makedirs("./data", exist_ok=True)
        log_fn("Downloading MovieLens 100K …")
        urllib.request.urlretrieve(url, dest)
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall("./data")
        os.remove(dest)
        log_fn("Download complete.")


# ─── Main data loader ─────────────────────────────────────────────────────────

def load_data(log_fn=print):
    """
    Returns a dict with all arrays needed by the model.

    Simulated features (text_mat, img_mat) stand in for real BERT / ResNet
    embeddings. Swap these out for actual encoder outputs when integrating
    the full pipeline.
    """
    _download_movielens(log_fn)

    # ── Ratings ──────────────────────────────────────────────────────────────
    ratings = pd.read_csv(
        f"{CFG.data_dir}/u.data", sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    user_ids = sorted(ratings["user_id"].unique())
    item_ids = sorted(ratings["item_id"].unique())
    u2idx = {u: i for i, u in enumerate(user_ids)}
    i2idx = {it: i for i, it in enumerate(item_ids)}
    ratings["user_idx"] = ratings["user_id"].map(u2idx)
    ratings["item_idx"] = ratings["item_id"].map(i2idx)
    n_users, n_items = len(user_ids), len(item_ids)

    # ── User demographics ────────────────────────────────────────────────────
    users = pd.read_csv(
        f"{CFG.data_dir}/u.user", sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    users["user_idx"] = users["user_id"].map(u2idx)
    users = users.dropna(subset=["user_idx"])
    users["user_idx"]   = users["user_idx"].astype(int)
    users["age_norm"]   = (users["age"] - users["age"].mean()) / (users["age"].std() + 1e-8)
    users["gender_bin"] = (users["gender"] == "M").astype(float)
    occ2idx = {o: i for i, o in enumerate(users["occupation"].unique())}
    users["occ_idx"] = users["occupation"].map(occ2idx)

    demo_mat = np.zeros((n_users, 3), dtype=np.float32)
    for _, row in users.iterrows():
        demo_mat[int(row["user_idx"])] = [row["age_norm"], row["gender_bin"], row["occ_idx"]]

    # ── Item features ────────────────────────────────────────────────────────
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    items = pd.read_csv(
        f"{CFG.data_dir}/u.item", sep="|", encoding="latin-1",
        names=["item_id", "title", "release_date", "video_date", "imdb_url"] + genre_names
    )
    items["item_idx"] = items["item_id"].map(i2idx)
    items = items.dropna(subset=["item_idx"])
    items["item_idx"] = items["item_idx"].astype(int)

    genre_mat = np.zeros((n_items, len(genre_names)), dtype=np.float32)
    title_map = {}
    for _, row in items.iterrows():
        idx = int(row["item_idx"])
        genre_mat[idx] = row[genre_names].values.astype(np.float32)
        title_map[idx] = row["title"]

    # ── Simulated BERT / ResNet features ─────────────────────────────────────
    #    Replace with real encoder outputs for the full pipeline.
    rng      = np.random.RandomState(0)
    text_mat = rng.randn(n_items, CFG.text_dim).astype(np.float32)
    text_mat /= np.linalg.norm(text_mat, axis=1, keepdims=True) + 1e-8
    img_mat  = rng.randn(n_items, CFG.image_dim).astype(np.float32)
    img_mat  /= np.linalg.norm(img_mat,  axis=1, keepdims=True) + 1e-8

    log_fn(f"Loaded: {n_users} users | {n_items} items | {len(ratings)} ratings")
    log_fn(f"Sparsity: {1 - len(ratings) / (n_users * n_items):.2%}")

    return dict(
        ratings=ratings,
        n_users=n_users, n_items=n_items,
        demo_mat=demo_mat,
        text_mat=text_mat, img_mat=img_mat,
        genre_mat=genre_mat, n_genres=len(genre_names),
        title_map=title_map,
        u2idx=u2idx, i2idx=i2idx,
    )


# ─── Train / val / test split ────────────────────────────────────────────────

def split_data(ratings, n_items, log_fn=print):
    """
    Stratified 80 / 10 / 10 split.
    Also identifies cold-start users and items (< tau interactions in train).
    """
    user_counts = ratings.groupby("user_idx").size()
    item_counts = ratings.groupby("item_idx").size()
    cold_users  = set(user_counts[user_counts < CFG.cold_start_tau].index)
    cold_items  = set(item_counts[item_counts < CFG.cold_start_tau].index)

    train_df, temp_df = train_test_split(ratings, test_size=0.2,  random_state=CFG.seed)
    val_df,   test_df = train_test_split(temp_df,  test_size=0.5, random_state=CFG.seed)

    # Popularity weights used for negative sampling
    item_freqs = np.zeros(n_items, dtype=np.float32)
    for i in train_df["item_idx"]:
        item_freqs[i] += 1

    log_fn(f"Split → train:{len(train_df)} | val:{len(val_df)} | test:{len(test_df)}")
    log_fn(f"Cold users: {len(cold_users)} | Cold items: {len(cold_items)}")

    return train_df, val_df, test_df, cold_users, cold_items, item_freqs
