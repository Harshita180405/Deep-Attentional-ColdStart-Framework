# ─────────────────────────────────────────────────────────────────────────────
#  model.py  —  DACSF architecture
#
#  Modules
#  ───────
#  TextEncoder    — linear projection of (simulated) BERT output → R^d
#  ImageEncoder   — linear projection of (simulated) ResNet output → R^d
#  CatEncoder     — genre one-hot → MLP → R^d
#  DemoEncoder    — demographic features → MLP → R^d
#  HistEncoder    — mean-pool of interacted-item text embeddings → R^d
#  AFFN           — Attentional Feature Fusion Network (core contribution)
#  DACSF          — full model tying all encoders + AFFN + scoring
# ─────────────────────────────────────────────────────────────────────────────

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# ─── Individual encoders ──────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """Projects simulated BERT features into the shared latent space R^d."""
    def __init__(self, text_dim, d, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d),
        )

    def forward(self, x):
        return self.proj(x)


class ImageEncoder(nn.Module):
    """Projects simulated ResNet features into the shared latent space R^d."""
    def __init__(self, image_dim, d, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d),
        )

    def forward(self, x):
        return self.proj(x)


class CatEncoder(nn.Module):
    """Encodes genre / categorical one-hot vectors → R^d via a small MLP."""
    def __init__(self, n_genres, d, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_genres, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d),
        )

    def forward(self, x):
        return self.proj(x)


class DemoEncoder(nn.Module):
    """
    Encodes structured demographic features (age, gender, occupation) via a
    multi-layer perceptron with ReLU activations and dropout regularisation.
    Hidden layer sizes are controlled by CFG.demo_hidden.
    """
    def __init__(self, demo_dim, hidden_sizes, d, dropout):
        super().__init__()
        layers, in_d = [], demo_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, d))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class HistEncoder(nn.Module):
    """
    Aggregates a user's interaction history via masked mean-pooling of
    item text embeddings. Returns a zero vector for new users (cold-start).
    """
    def forward(self, hist_embs, hist_mask):
        # hist_embs: [B, max_hist, d]
        # hist_mask: [B, max_hist]
        count = hist_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (hist_embs * hist_mask.unsqueeze(-1)).sum(dim=1) / count


# ─── Attentional Feature Fusion Network ──────────────────────────────────────

class AFFN(nn.Module):
    """
    Dynamically weights modality-specific embeddings via scaled dot-product
    attention with a learnable context vector.

    For each modality embedding e_k:
        s_k  = (e_k · v) / sqrt(d)       alignment score
        α_k  = softmax({s_k})             attention weight
        e    = Σ α_k e_k                  fused embedding

    Cold-start behaviour: if a modality is absent, its embedding is set to
    zeros, which yields a near-zero dot product and thus near-zero α_k —
    automatically suppressing the missing modality without any masking logic.
    """

    def __init__(self, d, n_modalities):
        super().__init__()
        self.d       = d
        self.context = nn.Parameter(torch.randn(d))  # learnable context vector v

    def forward(self, embeddings):
        """
        Args:
            embeddings: list of K tensors, each [B, d]
        Returns:
            fused:  [B, d]
            alphas: [B, K]  (for interpretability / logging)
        """
        E     = torch.stack(embeddings, dim=1)          # [B, K, d]
        v     = self.context / (self.context.norm() + 1e-8)
        s     = torch.einsum("bkd,d->bk", E, v) / math.sqrt(self.d)
        alpha = F.softmax(s, dim=-1)                    # [B, K]
        fused = torch.einsum("bk,bkd->bd", alpha, E)   # [B, d]
        return fused, alpha


# ─── Full DACSF model ─────────────────────────────────────────────────────────

class DACSF(nn.Module):
    """
    Deep Attentional Cold-Start Framework.

    Item pipeline: text → TextEncoder ]
                   img  → ImageEncoder ] → AFFN_item → e_i ∈ R^d
                   cat  → CatEncoder  ]

    User pipeline: demo → DemoEncoder  ]
                   hist → HistEncoder  ] → AFFN_user → e_u ∈ R^d

    Scoring: R̂_{u,i} = σ( e_u^T W e_i + b )
    """

    def __init__(self, text_dim, image_dim, n_genres, demo_dim, cfg):
        super().__init__()
        d, dr = cfg.latent_dim, cfg.dropout

        # Encoders
        self.text_enc  = TextEncoder(text_dim,  d, dr)
        self.img_enc   = ImageEncoder(image_dim, d, dr)
        self.cat_enc   = CatEncoder(n_genres,    d, dr)
        self.demo_enc  = DemoEncoder(demo_dim, cfg.demo_hidden, d, dr)
        self.hist_enc  = HistEncoder()

        # Attentional fusion networks (separate context vectors for item / user)
        self.affn_item = AFFN(d, cfg.num_item_mod)
        self.affn_user = AFFN(d, cfg.num_user_mod)

        # Bilinear scoring
        self.W    = nn.Parameter(torch.randn(d, d) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

        self.drop = nn.Dropout(dr)

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def encode_item(self, text_feat, img_feat, cat_feat):
        """Returns fused item embedding + attention weights."""
        et = self.drop(self.text_enc(text_feat))
        ei = self.drop(self.img_enc(img_feat))
        ec = self.drop(self.cat_enc(cat_feat))
        return self.affn_item([et, ei, ec])       # fused [B,d], alphas [B,3]

    def encode_user(self, demo, hist_embs, hist_mask):
        """Returns fused user embedding + attention weights."""
        ed = self.drop(self.demo_enc(demo))
        eh = self.hist_enc(hist_embs, hist_mask)
        return self.affn_user([ed, eh])            # fused [B,d], alphas [B,2]

    # ── Preference scoring ────────────────────────────────────────────────────

    def score(self, eu, ei):
        """
        Bilinear preference score:  σ( e_u^T W e_i + b )  ∈ (0, 1)
        """
        return torch.sigmoid((eu @ self.W * ei).sum(dim=-1) + self.bias)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, user_feats, item_pos_feats, item_neg_feats=None):
        """
        Args:
            user_feats:      (demo, hist_embs, hist_mask)
            item_pos_feats:  (text, img, cat) for positive items
            item_neg_feats:  (text, img, cat) for negative items (BPR only)
        Returns:
            score_pos, score_neg (or None), item_attention_weights
        """
        demo, hist_embs, hist_mask = user_feats
        eu, _           = self.encode_user(demo, hist_embs, hist_mask)
        ei_p, alpha_i   = self.encode_item(*item_pos_feats)
        s_pos           = self.score(eu, ei_p)

        s_neg = None
        if item_neg_feats is not None:
            ei_n, _ = self.encode_item(*item_neg_feats)
            s_neg   = self.score(eu, ei_n)

        return s_pos, s_neg, alpha_i
